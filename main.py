# main.py
from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import chromadb

from google import genai
from google.genai import types

from chunk import read_data, get_chunks


# ==== Configuration ====
LLM_MODEL = "gemini-2.5-flash-preview-05-20"
EMBEDDING_MODEL = "gemini-embedding-001"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_chunks"


def _get_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment."
        )
    return key


# Initialize Google GenAI client
google_client = genai.Client(api_key=_get_api_key())


# Initialize Chroma persistent client & collection
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},  # cosine works well with embeddings
)


def embed(text: str, *, store: bool) -> List[float]:
    """
    Convert text to an embedding vector. Uses RETRIEVAL_DOCUMENT when storing
    and RETRIEVAL_QUERY when embedding a user query.
    """
    # Using the new GenAI SDK's EmbedContentConfig per official docs.
    task = "RETRIEVAL_DOCUMENT" if store else "RETRIEVAL_QUERY"
    result = google_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task),
    )
    return result.embeddings[0].values


def _chunk_id(file_path: Path, index: int) -> str:
    """Deterministic ID from source file + chunk index so re-runs upsert, not duplicate."""
    key = f"{file_path}::{index}"
    return hashlib.sha1(key.encode()).hexdigest()


def create_db(file_path: Optional[Path] = None, chunks: Optional[List[str]] = None) -> None:
    """
    Build or resume the Chroma collection.
    Uses deterministic chunk IDs so re-runs skip already-embedded chunks
    and safely resume after hitting a daily quota limit.
    """
    if file_path is None:
        file_path = Path("data.txt")

    if chunks is None:
        text = read_data(file_path)
        chunks = get_chunks(text)

    if not chunks:
        print("No chunks to index.")
        return

    # Find which IDs already exist so we can skip them
    all_ids = [_chunk_id(file_path, i) for i in range(len(chunks))]
    existing = set(collection.get(ids=all_ids, include=[])["ids"])
    pending = [(i, cid, chunks[i]) for i, cid in enumerate(all_ids) if cid not in existing]

    if not pending:
        print(f"✅ All {len(chunks)} chunks already indexed. Nothing to do.")
        return

    print(f"Resuming: {len(existing)} already done, {len(pending)} remaining.")

    total = 0
    for idx, cid, text in pending:
        embedding = embed(text, store=True)
        collection.upsert(
            ids=[cid],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"source": str(file_path), "index": idx}],
        )
        total += 1
        time.sleep(0.65)  # stay under 100 req/min free-tier limit
        if total % 50 == 0:
            print(f"  {len(existing) + total}/{len(chunks)} indexed...")

    count = collection.count()
    print(f"✅ Indexed {total} new chunks. Collection now has {count}/{len(chunks)} total items.")


def query_db(query: str, top_k: int = 7) -> List[str]:
    """
    Embed the query and retrieve the top_k most similar documents from Chroma.
    Returns a list of matching document strings.
    """
    q_emb = embed(query, store=False)
    res = collection.query(query_embeddings=[q_emb], n_results=top_k)
    # Chroma returns lists per query; we only issued one query
    docs = res.get("documents", [[]])[0]
    return docs


PROMPT_TEMPLATE = """You are a helpful assistant. Use the provided context to answer the user's question.
Don't search beyond the user chosen file. If the answer is not in the context, say you don't know.

# Question
{question}

# Retrieved Context
{context}

Provide a concise, direct answer first, then any brief supporting details.
"""


def answer_with_llm(question: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant context found)"
    prompt = PROMPT_TEMPLATE.format(question=question, context=context)

    resp = google_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    # The GenAI SDK returns a structured response; text lives in .text
    return getattr(resp, "text", "").strip()


# ---- Example CLI usage ----
@dataclass
class Args:
    build: bool = False
    ask: Optional[str] = None


def parse_args(argv: List[str]) -> Args:
    """
    Minimal CLI:
      python main.py --build                -> (re)build the DB from data.txt
      python main.py --ask "your question"  -> query and answer
    """
    build = "--build" in argv
    ask = None
    if "--ask" in argv:
        try:
            idx = argv.index("--ask")
            ask = argv[idx + 1]
        except (ValueError, IndexError):
            raise SystemExit("Usage: python main.py --ask \"your question\"")
    return Args(build=build, ask=ask)


def main() -> None:
    args = parse_args(sys.argv[1:])

    if args.build:
        print("Building (or refreshing) the vector DB from data.txt ...")
        create_db()

    if args.ask:
        print(f"🔎 Query: {args.ask}")
        retrieved = query_db(args.ask, top_k=5)
        print("\nTop matches:")
        for i, d in enumerate(retrieved, 1):
            print(f"\n[{i}] {d[:300]}{'...' if len(d) > 300 else ''}")

        print("\n🤖 Answer:")
        print(answer_with_llm(args.ask, retrieved))

    if not args.build and not args.ask:
        print("Nothing to do. Try one of:")
        print("  python main.py --build")
        print('  python main.py --ask "What does section X say?"')


if __name__ == "__main__":
    main()
