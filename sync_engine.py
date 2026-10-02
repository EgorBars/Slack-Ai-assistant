import os
import json
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from notion_client import Client
from apscheduler.schedulers.blocking import BlockingScheduler

from notion_export import fetch_page_markdown
from rag_config import DATA_PATH, CHROMA_PATH
from rag_utils import run_indexing, chroma_is_empty

NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
RAW_IDS = os.environ.get("NOTION_PAGE_IDS", "").split(",")
PAGE_IDS = [i.strip().split("-")[-1] for i in RAW_IDS if i.strip()]

DATA_PATH = Path(DATA_PATH)
STATE_FILE = "sync_state.json"

notion = Client(auth=NOTION_TOKEN)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"pages": {}, "meta": {}}
    return {"pages": {}, "meta": {}}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=4)


def sync_knowledge(force=False):
    print(f"[{datetime.now()}] 🔍 Проверка {len(PAGE_IDS)} страниц Notion...")
    if not DATA_PATH.exists():
        DATA_PATH.mkdir()

    state = load_state()
    any_changes = False

    for p_id in PAGE_IDS:
        try:
            page_meta = notion.pages.retrieve(page_id=p_id)
            last_time = page_meta.get("last_edited_time")

            props = page_meta.get("properties", {})
            title_obj = props.get("title") or props.get("Name")
            page_title = title_obj.get("title", [{}])[0].get("plain_text", "Untitled")

            filename = "".join(x for x in page_title if x.isalnum() or x in " _-").strip() + ".md"
            file_path = DATA_PATH / filename
            notion_url = f"https://www.notion.so/{p_id.replace('-', '')}"

            if not force and state["pages"].get(p_id) == last_time and file_path.exists():
                print(f"✅ Страница '{page_title}' актуальна.")
                continue

            print(f"🔄 Изменения найдены! Качаю '{page_title}'...")
            md_content, _ = fetch_page_markdown(notion, p_id)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            state["pages"][p_id] = last_time
            state["meta"][filename] = {"title": page_title, "url": notion_url}
            any_changes = True
            print(f"   💾 Сохранено: {filename} ({len(md_content)} символов)")

        except Exception as e:
            print(f"❌ Ошибка при обработке страницы {p_id}: {e}")

    if any_changes:
        save_state(state)
        run_indexing(state["meta"])
    elif chroma_is_empty():
        print("⚠️ ChromaDB пуста — запускаю индексацию...")
        run_indexing(state.get("meta", {}))
    else:
        print("😴 Обновлений в Notion нет.")


scheduler = BlockingScheduler()
scheduler.add_job(sync_knowledge, "interval", minutes=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Синхронизация Notion → data/ → ChromaDB")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Полная переиндексация из data/ перед запуском мониторинга.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Один цикл синхронизации и выход (без постоянного мониторинга).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перекачать все страницы из Notion, даже если они не менялись.",
    )
    args = parser.parse_args()

    if args.reindex:
        state = load_state()
        if Path(CHROMA_PATH).exists():
            shutil.rmtree(CHROMA_PATH)
            print("🧹 Старая ChromaDB удалена.")
        run_indexing(state.get("meta", {}))

    if not args.reindex or not args.once:
        sync_knowledge(force=args.force)

    if args.once:
        print("✅ Одноразовая синхронизация завершена.")
    else:
        print("⏰ Мониторинг запущен — проверка каждые 2 мин. Ctrl+C для остановки.")
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            print("🛑 Синхронизация остановлена.")
