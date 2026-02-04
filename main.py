# main.py
from __future__ import annotations

import os
import sys
import uuid
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


def create_db(file_path: Optional[Path] = None, chunks: Optional[List[str]] = None) -> None:
    """
    Create / refresh the Chroma collection by embedding and upserting chunks.
    If file_path is None, read from data.txt and compute chunks.
    """
    if file_path is None:
        # If no file path is provided, default to "data.txt"
        file_path = Path("data.txt")

    if chunks is None:
        # Read the file and chunk the content
        text = read_data(file_path)
        chunks = get_chunks(text)

    if not chunks:
        print("No chunks to index.")
        return

    # Upsert in small batches to avoid long single requests
    BATCH = 32
    total = 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i: i + BATCH]
        ids = [str(uuid.uuid4()) for _ in batch]
        embeddings = [embed(c, store=True) for c in batch]
        metadatas = [{"source": str(file_path), "index": i + j} for j, _ in enumerate(batch)]
        collection.upsert(ids=ids, embeddings=embeddings, documents=batch, metadatas=metadatas)
        total += len(batch)

    count = collection.count()
    print(f"✅ Indexed {total} chunks. Collection now has {count} total items.")


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
