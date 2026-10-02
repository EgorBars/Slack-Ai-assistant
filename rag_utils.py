import os
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from markdown_normalize import normalize_markdown
from rag_config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    HEADERS_TO_SPLIT,
    INDEX_BATCH_SIZE,
)
from rag_embeddings import get_embeddings


def _header_path(metadata: dict) -> str:
    parts = [
        metadata.get(f"Header {i}", "")
        for i in range(1, 5)
        if metadata.get(f"Header {i}")
    ]
    return " > ".join(parts)


def _base_metadata(source_file: str, file_meta: dict) -> dict:
    return {
        "source": source_file,
        "title": file_meta.get("title", source_file.replace(".md", "")),
        "url": file_meta.get("url", ""),
    }


def _attach_chunk_context(chunk: Document, title: str) -> Document:
    """Embed searchable context into chunk text for better retrieval."""
    path = _header_path(chunk.metadata)
    prefix = f"[Документ: {title}]"
    if path:
        prefix += f" [Раздел: {path}]"
    if not chunk.page_content.startswith(prefix):
        chunk.page_content = f"{prefix}\n{chunk.page_content}"
    if path:
        chunk.metadata["section"] = path
    return chunk


def chunk_documents(documents, metadata_map=None):
    """Split documents into chunks preserving header metadata."""
    metadata_map = metadata_map or {}

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    all_chunks: list[Document] = []

    for doc in documents:
        source_file = os.path.basename(doc.metadata["source"])
        file_meta = metadata_map.get(
            source_file,
            {"title": source_file.replace(".md", ""), "url": ""},
        )
        title = file_meta["title"]
        base_meta = _base_metadata(source_file, file_meta)

        content = normalize_markdown(doc.page_content, title)
        sections = header_splitter.split_text(content)

        doc_chunks: list[Document] = []

        if not sections:
            doc_chunks = section_splitter.create_documents([content], metadatas=[base_meta.copy()])
        elif len(sections) == 1 and len(sections[0].page_content) > CHUNK_SIZE:
            # Single large section without sub-headers — split but keep metadata
            meta = {**base_meta, **sections[0].metadata}
            doc_chunks = section_splitter.create_documents(
                [sections[0].page_content],
                metadatas=[meta],
            )
        else:
            for section in sections:
                section_meta = {**base_meta, **section.metadata}
                if len(section.page_content) <= CHUNK_SIZE:
                    doc_chunks.append(
                        Document(page_content=section.page_content, metadata=section_meta)
                    )
                else:
                    sub_chunks = section_splitter.create_documents(
                        [section.page_content],
                        metadatas=[section_meta],
                    )
                    doc_chunks.extend(sub_chunks)

        for i, chunk in enumerate(doc_chunks):
            chunk.metadata.update(base_meta)
            chunk.metadata["chunk_index"] = i
            all_chunks.append(_attach_chunk_context(chunk, title))

    return all_chunks


def run_indexing(metadata_map=None, embeddings=None):
    """Index all markdown files from data/ into Chroma."""
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        print("❓ Папка data/ не найдена.")
        return 0

    loader = DirectoryLoader(
        str(data_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    if not documents:
        print("❓ Файлы для индексации не найдены.")
        return 0

    embeddings = embeddings or get_embeddings()
    all_chunks = chunk_documents(documents, metadata_map)
    print(f"📦 Создано {len(all_chunks)} фрагментов, начинаю векторизацию...")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    unique_sources = list({c.metadata["source"] for c in all_chunks})
    for source in unique_sources:
        try:
            db.delete(where={"source": source})
        except Exception:
            pass

    for i in range(0, len(all_chunks), INDEX_BATCH_SIZE):
        batch = all_chunks[i : i + INDEX_BATCH_SIZE]
        db.add_documents(batch)
        print(f"   ... {min(i + INDEX_BATCH_SIZE, len(all_chunks))}/{len(all_chunks)}")

    print(f"🚀 База знаний обновлена! Всего {len(all_chunks)} фрагментов.")
    return len(all_chunks)


def chroma_is_empty() -> bool:
    """Check Chroma without loading the embedding model."""
    chroma_path = Path(CHROMA_PATH)
    if not chroma_path.exists():
        return True
    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collections = client.list_collections()
        if not collections:
            return True
        return client.get_collection(collections[0].name).count() == 0
    except Exception:
        return True
