"""Ingestion pipeline: PDF → text → chunks → embeddings → MariaDB."""

import os
import re
import sys
import argparse
from collections import Counter

import fitz  # pymupdf

from config import (
    CHUNK_TARGET_SIZE, CHUNK_MIN_SIZE, CHUNK_MAX_SIZE, INSERT_BATCH_SIZE,
)
import db
import embedder


# ---------------------------------------------------------------------------
# 1) PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path, max_pages=None, progress_callback=None):
    """Extract text from PDF, filtering headers/footers.
    Returns list of (page_number, page_text) tuples (1-indexed).
    Also writes a .txt file next to the PDF for debugging.
    """
    doc = fitz.open(pdf_path)
    total = min(len(doc), max_pages) if max_pages else len(doc)

    # First pass: collect candidate header/footer lines
    top_lines = Counter()
    bottom_lines = Counter()
    for i in range(total):
        page = doc[i]
        blocks = [b for b in page.get_text("blocks") if b[6] == 0]  # text blocks only
        if blocks:
            # Sort by vertical position
            blocks.sort(key=lambda b: b[1])
            page_height = page.rect.height
            for b in blocks:
                text = b[4].strip()
                if not text:
                    continue
                y_top = b[1]
                y_bottom = b[3]
                if y_top < page_height * 0.08:
                    top_lines[text] += 1
                if y_bottom > page_height * 0.92:
                    bottom_lines[text] += 1

    # Lines appearing on >50% of pages are likely headers/footers
    threshold = total * 0.5
    skip_lines = set()
    for text, count in top_lines.items():
        if count >= threshold:
            skip_lines.add(text)
    for text, count in bottom_lines.items():
        if count >= threshold:
            skip_lines.add(text)

    # Second pass: extract text
    pages = []
    for i in range(total):
        page = doc[i]
        blocks = [b for b in page.get_text("blocks") if b[6] == 0]
        blocks.sort(key=lambda b: b[1])  # top to bottom
        texts = []
        for b in blocks:
            text = b[4].strip()
            if text and text not in skip_lines:
                texts.append(text)
        page_text = "\n\n".join(texts)
        # Normalize whitespace and merge hyphen breaks
        page_text = re.sub(r"-\n\s*", "", page_text)
        page_text = re.sub(r"\n(?!\n)", " ", page_text)
        page_text = re.sub(r"[ \t]+", " ", page_text)
        pages.append((i + 1, page_text))
        if progress_callback:
            progress_callback(i + 1, total)

    doc.close()

    # Write debug .txt file
    txt_path = os.path.splitext(pdf_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for page_num, text in pages:
            f.write(f"--- Sida {page_num} ---\n{text}\n\n")

    return pages, total


# ---------------------------------------------------------------------------
# 2) Chunking
# ---------------------------------------------------------------------------

def _word_count(text):
    return len(text.split())


def chunk_pages(pages):
    """Split pages into chunks with context_before/after.
    Returns list of dicts: {page_number, paragraph_index, chunk_text, context_before, context_after}.
    """
    # Build raw paragraph list
    raw = []  # list of (page_number, text)
    for page_num, page_text in pages:
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        for p in paragraphs:
            raw.append((page_num, p))

    # Merge short paragraphs and split long ones
    processed = []
    buffer_text = ""
    buffer_page = None
    for page_num, text in raw:
        if buffer_text:
            wc = _word_count(buffer_text)
            if wc < CHUNK_MIN_SIZE:
                # Merge with next paragraph if same page or next
                buffer_text += " " + text
                continue
            else:
                processed.append((buffer_page, buffer_text))
                buffer_text = ""
                buffer_page = None

        wc = _word_count(text)
        if wc < CHUNK_MIN_SIZE:
            buffer_text = text
            buffer_page = page_num
        elif wc > CHUNK_MAX_SIZE:
            # Split at sentence boundaries
            sentences = re.split(r"(?<=\. )", text)
            current = ""
            for sent in sentences:
                if _word_count(current + sent) > CHUNK_TARGET_SIZE and current:
                    processed.append((page_num, current.strip()))
                    current = sent
                else:
                    current += sent
            if current.strip():
                processed.append((page_num, current.strip()))
        else:
            processed.append((page_num, text))

    # Flush remaining buffer
    if buffer_text:
        processed.append((buffer_page, buffer_text))

    # Build chunks with context and paragraph_index
    chunks = []
    page_para_counter = {}
    for i, (page_num, text) in enumerate(processed):
        if page_num not in page_para_counter:
            page_para_counter[page_num] = 0
        para_idx = page_para_counter[page_num]
        page_para_counter[page_num] += 1

        # context_before: last sentence of previous chunk
        context_before = None
        if i > 0:
            prev_text = processed[i - 1][1]
            sentences = re.split(r"(?<=\.) ", prev_text)
            context_before = sentences[-1].strip() if sentences else None

        # context_after: first sentence of next chunk
        context_after = None
        if i < len(processed) - 1:
            next_text = processed[i + 1][1]
            sentences = re.split(r"(?<=\.) ", next_text)
            context_after = sentences[0].strip() if sentences else None

        chunks.append({
            "page_number": page_num,
            "paragraph_index": para_idx,
            "chunk_text": text,
            "context_before": context_before,
            "context_after": context_after,
        })

    return chunks


# ---------------------------------------------------------------------------
# 3-4) Embedding + Storage
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path, title=None, max_pages=None, progress_callback=None):
    """Full ingestion pipeline. progress_callback(stage, detail) for UI updates."""
    if title is None:
        title = os.path.splitext(os.path.basename(pdf_path))[0]

    def notify(stage, detail=""):
        if progress_callback:
            progress_callback(stage, detail)

    # Extract text
    notify("extract", "Startar textextraktion...")
    pages, total_pages = extract_text_from_pdf(
        pdf_path, max_pages=max_pages,
        progress_callback=lambda cur, tot: notify("extract", f"Extraherar text... sida {cur}/{tot}"),
    )

    # Create book record
    book_id = db.create_book(title, os.path.basename(pdf_path), total_pages)

    try:
        # Chunk
        notify("chunk", "Skapar chunks...")
        chunks = chunk_pages(pages)
        notify("chunk", f"{len(chunks)} chunks skapade")

        # Embed
        texts = [c["chunk_text"] for c in chunks]
        embeddings = embedder.embed_texts(
            texts,
            progress_callback=lambda cur, tot: notify("embed", f"Embeddar chunk {cur}/{tot}..."),
        )

        # Store in batches
        notify("store", "Sparar i MariaDB...")
        batch = []
        for i, chunk in enumerate(chunks):
            batch.append({
                "book_id": book_id,
                "page_number": chunk["page_number"],
                "paragraph_index": chunk["paragraph_index"],
                "chunk_text": chunk["chunk_text"],
                "context_before": chunk["context_before"],
                "context_after": chunk["context_after"],
                "embedding": embeddings[i],
            })
            if len(batch) >= INSERT_BATCH_SIZE:
                db.insert_chunks_batch(batch)
                batch = []
        if batch:
            db.insert_chunks_batch(batch)

        db.update_book_chunk_count(book_id)
        notify("done", f"Klar! {len(chunks)} chunks lagrade.")
        return book_id

    except Exception as e:
        # Cleanup on failure
        db.delete_book(book_id)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into Bok-RAG")
    parser.add_argument("--file", required=True, help="Path to PDF file")
    parser.add_argument("--title", default=None, help="Book title (default: filename)")
    parser.add_argument("--max-pages", type=int, default=None, help="Only process first N pages")
    args = parser.parse_args()

    def cli_progress(stage, detail):
        print(f"[{stage}] {detail}")

    book_id = ingest_pdf(args.file, title=args.title, max_pages=args.max_pages, progress_callback=cli_progress)
    print(f"Ingestion complete. Book ID: {book_id}")


if __name__ == "__main__":
    # Allow imports from src directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
