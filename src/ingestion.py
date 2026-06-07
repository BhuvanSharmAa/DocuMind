"""
src/ingestion.py — PDF loading + smart text chunking (Step 2)
"""
import fitz  # PyMuPDF
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR


def load_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str) -> list:
    """
    Split text into overlapping chunks for embedding.

    RecursiveCharacterTextSplitter tries to split on:
      ['\n\n', '\n', ' ', ''] — in order of preference.
    This preserves paragraph structure wherever possible.

    Chunk size 500 chars ≈ 120 tokens — within MiniLM's 256-token limit.
    Overlap 50 chars prevents answers straddling chunk boundaries from being lost.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks


def ingest_pdf(pdf_path: str) -> list:
    """Full pipeline: PDF → raw text → chunks. Returns list of strings."""
    raw_text = load_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    print(f"[Ingestion] {os.path.basename(pdf_path)}: {len(chunks)} chunks from {len(raw_text):,} chars")
    return chunks
