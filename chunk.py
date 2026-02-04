# chunk.py
from __future__ import annotations
from pathlib import Path
from typing import List

import chardet
import PyPDF2
import docx

DATA_PATH = Path(__file__).parent / "data.txt"


def detect_encoding(path: Path) -> str:
    """Detect file encoding using chardet."""
    with open(path, "rb") as f:
        raw = f.read(2048)
    result = chardet.detect(raw)
    return result["encoding"] or "utf-8"


def read_data(path: Path = None) -> str:
    """
    Read text from TXT, PDF, or DOCX files with automatic encoding detection.

    Args:
        path: File path. Defaults to DATA_PATH if None.

    Returns:
        The full text content of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnicodeDecodeError: If the file cannot be decoded.
    """
    if path is None:
        path = DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages)

    if suffix == ".docx":
        doc = docx.Document(path)
        return "\n".join(para.text for para in doc.paragraphs)

    # Plain text – try UTF-8 first, then chardet fallback
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(path)
        return path.read_text(encoding=encoding)


def get_chunks(text: str) -> List[str]:
    """
    Split text into chunks on blank lines while *preserving* the most recent
    header (lines starting with '#') by prepending it to each following chunk.

    Rules:
      - One or more header lines (starting with '#') set the current header block.
      - A blank line separates chunks.
      - Non-header paragraphs become chunks; the latest header block is prepended.
    """
    # Normalize newlines and split on double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    chunks: List[str] = []

    current_header_lines: List[str] = []
    for para in paragraphs:
        if not para:
            continue

        lines = para.splitlines()
        if all(line.lstrip().startswith("#") for line in lines):
            current_header_lines = [ln.rstrip() for ln in lines]
            continue

        if current_header_lines:
            chunk = "\n\n".join(["\n".join(current_header_lines), para])
        else:
            chunk = para
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    text = read_data()
    chunks = get_chunks(text)
    for i, ch in enumerate(chunks, start=1):
        print(f"[Chunk {i}]")
        print(ch)
        print("-" * 30)
