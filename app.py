import os
import re
import sys

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from groq import Groq
from dotenv import load_dotenv
from langchain_chroma import Chroma

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from rag_config import (
    CHROMA_PATH,
    FINAL_TOP_K,
    GROQ_MODEL,
    MAX_SEMANTIC_DISTANCE,
    RETRIEVE_TOP_K,
    SEMANTIC_TOP_K,
)
from rag_embeddings import get_embeddings
from rag_routing import (
    filter_relevant_chunks,
    get_primary_doc_titles,
    load_document_registry,
    route_to_documents,
)

app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

sessions = {}

print("🧠 Загрузка модели эмбеддингов и ChromaDB...")
embeddings = get_embeddings()
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
print("✅ RAG готов к работе.")


def rewrite_query_with_context(user_history, user_text):
    """Turn a follow-up into a standalone search query."""
    if not user_history:
        return user_text
    try:
        recent = user_history[-4:]
        history_text = "\n".join(
            f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content'][:300]}"
            for m in recent
        )
        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": (
                    "Перепиши последний вопрос пользователя как самостоятельный поисковый запрос "
                    "для поиска в корпоративной документации. Учти контекст диалога. "
                    "Ответь ТОЛЬКО одной строкой — готовым запросом, без пояснений.\n\n"
                    f"Диалог:\n{history_text}\n\n"
                    f"Последний вопрос: {user_text}"
                ),
            }],
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=150,
        )
        rewritten = response.choices[0].message.content.strip().strip('"')
        if rewritten:
            return rewritten
    except Exception as e:
        print(f"⚠️ Ошибка переписывания запроса: {e}")
    return user_text


def expand_query_ai(search_query):
    """Generate extra search terms via Groq."""
    try:
        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": (
                    "Для поиска в корпоративной документации (процессы, Jira, Slack-каналы, приемка) "
                    "предложи 3-5 ключевых слов или коротких фраз на русском и/или английском. "
                    "Только через запятую, без нумерации и пояснений.\n\n"
                    f"Запрос: {search_query}"
                ),
            }],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        return [v.strip().strip(".") for v in raw.split(",") if v.strip()]
    except Exception as e:
        print(f"⚠️ Ошибка расширения запроса: {e}")
        return []


def _make_doc(page_content, metadata):
    return type("Doc", (object,), {"page_content": page_content, "metadata": metadata})()


def retrieve_chunks(search_query, source_filter=None, top_k=RETRIEVE_TOP_K):
    """Hybrid retrieval with fresh DB connection and safe filtering."""
    # 1. Инициализируем подключение заново, чтобы видеть изменения от sync_engine
    current_vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )

    clean_query = re.sub(r"[*_~`]", "", search_query).strip()

    # 2. Безопасное формирование фильтра
    chroma_filter = None
    if source_filter and isinstance(source_filter, list) and len(source_filter) > 0:
        if len(source_filter) == 1:
            chroma_filter = {"source": str(source_filter[0])}
        else:
            chroma_filter = {"source": {"$in": [str(s) for s in source_filter]}}

    scored = {}

    # 3. Семантический поиск (с обработкой ошибок)
    try:
        semantic_hits = current_vectorstore.similarity_search_with_score(
            clean_query, k=SEMANTIC_TOP_K, filter=chroma_filter,
        )
        for doc, distance in semantic_hits:
            if distance <= MAX_SEMANTIC_DISTANCE:
                scored[doc.page_content] = (doc, distance)
    except Exception as e:
        print(f"⚠️ Ошибка семантического поиска: {e}")
        # Если поиск с фильтром упал, пробуем без него как fallback
        if chroma_filter:
            return retrieve_chunks(search_query, source_filter=None, top_k=top_k)
        return []

    # 4. Подготовка ключевых слов
    extra_terms = expand_query_ai(clean_query)
    all_keywords = set()
    for word in clean_query.split():
        w = word.strip(".,?!:;()[]\"'").lower()
        if len(w) >= 2:
            all_keywords.add(w)
            all_keywords.add(w.capitalize())
    for term in extra_terms:
        t = term.strip()
        if len(t) >= 2:
            all_keywords.add(t.lower())
            all_keywords.add(t)

    # 5. Поиск по ключевым словам (через текущую коллекцию)
    keyword_hits = {}
    for word in all_keywords:
        try:
            kwargs = {"where_document": {"$contains": word}}
            if chroma_filter:
                kwargs["where"] = chroma_filter

            # Используем коллекцию текущего подключения
            found = current_vectorstore._collection.get(**kwargs)

            for i in range(len(found["documents"])):
                content = found["documents"][i]
                meta = found["metadatas"][i]
                keyword_hits[content] = keyword_hits.get(content, 0) + 1
                if content not in scored:
                    scored[content] = (_make_doc(content, meta), 0.45)
        except Exception:
            pass

    # 6. Ранжирование
    ranked = sorted(
        scored.values(),
        key=lambda x: (x[1] - keyword_hits.get(x[0].page_content, 0) * 0.08),
    )

    return [doc for doc, _ in ranked[:top_k]]


def build_context(chunks):
    """Format retrieved chunks and build url map for citations."""
    context_parts = []
    url_vault = {}

    for i, res in enumerate(chunks, 1):
        raw_title = res.metadata.get("title", "Документ")
        display_name = raw_title.replace(".md", "").replace("_", " ").strip()
        url_vault[display_name] = res.metadata.get("url", "#")

        header = " > ".join(
            res.metadata.get(f"Header {j}", "")
            for j in range(1, 5)
            if res.metadata.get(f"Header {j}")
        )
        label = f"[ФРАГМЕНТ {i} | {display_name}"
        if header:
            label += f" | {header}"
        label += "]"
        context_parts.append(f"{label}\n{res.page_content}")

    return "\n\n---\n\n".join(context_parts), url_vault


SYSTEM_PROMPT = """Ты — корпоративный ассистент по внутренним процессам компании.

РАЗРЕШЕНИЕ КОНФЛИКТОВ МЕЖДУ ДОКУМЕНТАМИ (КРИТИЧЕСКИ ВАЖНО):
1. Если указаны ПЕРВИЧНЫЕ ДОКУМЕНТЫ — отвечай ТОЛЬКО на их основе.
2. Если фрагменты из разных документов описывают разные процессы (например, «приемка эпика» vs «шаблон постановки на дизайн») — используй ТОЛЬКО фрагменты, соответствующие вопросу.
3. Совпадение слов (канал, приемка, результаты) НЕ означает, что фрагмент релевантен — смотри на контекст процесса.
4. Если фрагмент начинается с «Постановка на дизайн», «Постановка на тестовые стримы» — это НЕ про приемку эпиков (To Verify).
5. Если фрагмент про To Verify / Epic / FO / E2E acceptance — это НЕ про шаблоны создания запросов.

ПРАВИЛА ОТВЕТА:
1. Отвечай ТОЛЬКО на основе отфильтрованного контекста.
2. Если ответа нет — напиши: «В предоставленной документации эта информация отсутствует.»
3. Не выдумывай каналы, статусы, роли и процедуры.
4. Структурируй ответ списками, выделяй ключевое *жирным*.
5. В конце добавь: SOURCES_USED: [точные названия документов из меток фрагментов]"""


def get_ai_response(user_id, user_text):
    if user_id not in sessions:
        sessions[user_id] = []
    user_history = sessions[user_id]

    registry = load_document_registry()
    search_query = rewrite_query_with_context(user_history, user_text)

    # Этап 1: маршрутизация — какой документ первичный для этого вопроса
    primary_sources = route_to_documents(client, user_text, registry)
    primary_titles = get_primary_doc_titles(primary_sources, registry)

    # Этап 2: retrieval с фильтром по документу (fallback без фильтра)
    chunks = retrieve_chunks(search_query, source_filter=primary_sources or None)
    if not chunks and primary_sources:
        chunks = retrieve_chunks(search_query)
    if not chunks:
        return (
            "В базе знаний не найдено релевантных фрагментов по вашему вопросу. "
            "Попробуйте переформулировать или проверьте, что документация синхронизирована."
        )

    # Этап 3: LLM-фильтрация чанков — убираем семантический шум
    chunks = filter_relevant_chunks(client, user_text, chunks)
    if not chunks:
        chunks = retrieve_chunks(search_query, source_filter=primary_sources or None)[:FINAL_TOP_K]
    else:
        chunks = chunks[:FINAL_TOP_K]

    context_text, url_vault = build_context(chunks)

    primary_hint = ""
    if primary_titles:
        primary_hint = f"ПЕРВИЧНЫЕ ДОКУМЕНТЫ (authoritative): {', '.join(primary_titles)}\n\n"

    user_message = (
        f"{primary_hint}"
        f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ (уже отфильтрован):\n\n{context_text}\n\n"
        f"---\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{user_text}"
    )

    messages_for_ai = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + user_history
        + [{"role": "user", "content": user_message}]
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=messages_for_ai,
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=1024,
        )
        ai_answer = chat_completion.choices[0].message.content
    except Exception as e:
        ai_answer = f"❌ Ошибка LLM: {e}"

    final_links = []
    if "SOURCES_USED:" in ai_answer:
        parts = ai_answer.split("SOURCES_USED:")
        main_text = parts[0].strip()
        sources_part = parts[1].strip().replace("[", "").replace("]", "")

        for name in [s.strip() for s in sources_part.split(",")]:
            if name in url_vault:
                final_links.append(f"<{url_vault[name]}|{name}>")
            else:
                for vault_name, url in url_vault.items():
                    if name.lower() in vault_name.lower() or vault_name.lower() in name.lower():
                        final_links.append(f"<{url}|{vault_name}>")
                        break

        ai_answer = main_text

    user_history.append({"role": "user", "content": user_text})
    user_history.append({"role": "assistant", "content": ai_answer})
    sessions[user_id] = user_history[-10:]

    if final_links:
        ai_answer += f"\n\n_Информация извлечена из:_ :page_facing_up: {', '.join(dict.fromkeys(final_links))}"

    return ai_answer


@app.event("message")
def handle_message(event, say):
    if event.get("bot_id") or event.get("subtype") or "text" not in event:
        return

    text = event["text"].lower().strip()
    if text in ["сброс", "reset"]:
        sessions[event["user"]] = []
        say("✅ Память очищена.")
        return

    response = get_ai_response(event["user"], event["text"])
    say(response)


if __name__ == "__main__":
    print("⚡️ Бот-аналитик с гибридным поиском запущен!")
    handler = SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN"))
    handler.start()
