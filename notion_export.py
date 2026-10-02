"""Export Notion pages via the official Markdown API."""

from __future__ import annotations

from notion_client import Client


def fetch_page_markdown(notion: Client, page_id: str) -> tuple[str, bool]:
    """
    Download page body as Notion-flavored Markdown.

    Returns (markdown_text, was_truncated).
    """
    response = notion.pages.retrieve_markdown(page_id=page_id)
    markdown = response.get("markdown", "") or ""
    truncated = bool(response.get("truncated", False))
    unknown = response.get("unknown_block_ids") or []

    if truncated:
        print(f"   ⚠️ Страница {page_id}: markdown обрезан Notion API (слишком большая страница).")

    if unknown:
        print(f"   ⚠️ Страница {page_id}: неподдерживаемые блоки: {len(unknown)}")

    return markdown.strip(), truncated
