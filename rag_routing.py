"""Document routing and chunk filtering via Groq."""

import json
import re
from pathlib import Path

from rag_config import GROQ_MODEL

REGISTRY_PATH = Path(__file__).parent / "documents_registry.json"
SYNC_STATE_PATH = Path(__file__).parent / "sync_state.json"


def load_document_registry() -> list[dict]:
    """Load document catalog with scope descriptions."""
    docs = []
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            docs = json.load(f).get("documents", [])

    meta = {}
    if SYNC_STATE_PATH.exists():
        with open(SYNC_STATE_PATH, encoding="utf-8") as f:
            meta = json.load(f).get("meta", {})

    for doc in docs:
        file_meta = meta.get(doc["source"], {})
        doc["url"] = file_meta.get("url", doc.get("url", ""))
        if file_meta.get("title"):
            doc["title"] = file_meta["title"]

    return docs


def _parse_documents_line(text: str) -> list[str]:
    match = re.search(r"DOCUMENTS?\s*:\s*(.+)", text, re.I)
    if not match:
        return []
    raw = match.group(1).strip()
    if raw.upper() == "NONE":
        return []
    return [t.strip().strip('"').strip("'") for t in raw.split(",") if t.strip()]


def _parse_relevant_line(text: str) -> list[int] | None:
    match = re.search(r"RELEVANT\s*:\s*(.+)", text, re.I)
    if not match:
        return None
    raw = match.group(1).strip()
    if raw.upper() == "NONE":
        return []
    nums = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            nums.append(int(part))
    return nums


def route_to_documents(client, question: str, registry: list[dict]) -> list[str]:
    """
    Pick 1-2 primary source filenames for the question.
    Returns list of 'source' fields (e.g. 'Need Single ... .md').
    """
    if not registry:
        return []

    catalog_lines = []
    title_to_source = {}
    for doc in registry:
        title = doc["title"]
        title_to_source[title.lower()] = doc["source"]
        catalog_lines.append(
            f"• «{title}»\n"
            f"  О чём: {doc['scope']}\n"
            f"  НЕ о чём: {doc.get('not_about', '—')}"
        )

    prompt = (
        "Ты маршрутизатор запросов по корпоративной документации.\n"
        "Разные документы используют похожие слова (каналы, приемка, результаты), "
        "но описывают РАЗНЫЕ процессы. Выбери только первичный документ.\n\n"
        f"ДОКУМЕНТЫ:\n\n" + "\n\n".join(catalog_lines) + "\n\n"
        f"ВОПРОС: {question}\n\n"
        "Правила:\n"
        "— Вопрос про приемку ЭПИКА / To Verify / готовность эпика → документ про Acceptance\n"
        "— Вопрос про шаблон постановки задачи на дизайн/стрим/требования → Шаблоны\n"
        "— Вопрос про заполнение инициативы / PD / child-задачи → Заполнение инициативы\n"
        "— Вопрос про discovery / estimation / migration → Common process\n\n"
        "Ответь СТРОГО одной строкой:\n"
        "DOCUMENTS: Точное название документа"
    )

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=120,
        )
        raw = response.choices[0].message.content.strip()
        selected_titles = _parse_documents_line(raw)

        sources = []
        for title in selected_titles:
            src = title_to_source.get(title.lower())
            if src:
                sources.append(src)
            else:
                for doc in registry:
                    if title.lower() in doc["title"].lower() or doc["title"].lower() in title.lower():
                        sources.append(doc["source"])
                        break

        return list(dict.fromkeys(sources))[:2]
    except Exception as e:
        print(f"⚠️ Ошибка маршрутизации документов: {e}")
        return []


def filter_relevant_chunks(client, question: str, chunks: list) -> list:
    """LLM pre-filter: keep only chunks that directly answer the question."""
    if len(chunks) <= 2:
        return chunks

    summaries = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.metadata.get("title", "Документ")
        header = " > ".join(
            chunk.metadata.get(f"Header {j}", "")
            for j in range(1, 5)
            if chunk.metadata.get(f"Header {j}")
        )
        label = f"[{i}] «{title}»"
        if header:
            label += f" / {header}"
        preview = chunk.page_content[:700]
        summaries.append(f"{label}\n{preview}")

    prompt = (
        "Ты фильтруешь фрагменты документации перед ответом пользователю.\n"
        "Несколько фрагментов могут содержать похожие термины (каналы, приемка), "
        "но относиться к РАЗНЫМ процессам. Оставь только те, что ПРЯМО отвечают на вопрос.\n\n"
        f"ВОПРОС: {question}\n\n"
        f"ФРАГМЕНТЫ:\n\n" + "\n\n---\n\n".join(summaries) + "\n\n"
        "Ответь СТРОГО одной строкой:\n"
        "RELEVANT: номера через запятую (например 1,3) или NONE"
    )

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=60,
        )
        raw = response.choices[0].message.content.strip()
        indices = _parse_relevant_line(raw)
        if indices is None:
            return chunks[:3]
        if not indices:
            return []
        filtered = [chunks[i - 1] for i in indices if 0 < i <= len(chunks)]
        return filtered if filtered else chunks[:2]
    except Exception as e:
        print(f"⚠️ Ошибка фильтрации чанков: {e}")
        return chunks[:3]


def get_primary_doc_titles(sources: list[str], registry: list[dict]) -> list[str]:
    source_to_title = {d["source"]: d["title"] for d in registry}
    return [source_to_title.get(s, s) for s in sources]
