"""
Core RAG engine.

Three jobs:
  1. Turn text into vectors (Google Gemini embeddings)
  2. Store / search those vectors (Pinecone)
  3. Answer a question using only the retrieved text (Gemini chat)
"""

import time
import uuid

from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec

EMBED_MODEL = "gemini-embedding-001"

# Google retires model names over time. If you ever see a 404 saying a model is
# "no longer available", the error message names the replacement - put it here.
CHAT_MODEL = "gemini-3.6-flash"
VISION_MODEL = "gemini-3.6-flash"

# gemini-embedding-001 can output 768 / 1536 / 3072 dimensions.
# 768 keeps the Pinecone free tier comfortable and is plenty accurate.
EMBED_DIM = 768

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------

def gemini_client(api_key: str):
    return genai.Client(api_key=api_key)


# Google occasionally returns 503 ("high demand") or 429 (rate limit). Those are
# temporary, so wait a moment and try again rather than failing in front of
# whoever is watching.
RETRY_CODES = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "500", "INTERNAL")
MAX_ATTEMPTS = 4


def with_retry(call):
    """Run call(); retry a few times on temporary Google errors."""
    delay = 1.5
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            return call()
        except Exception as error:
            message = str(error)
            if not any(code in message for code in RETRY_CODES):
                raise
            last_error = error
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(delay)
                delay *= 2

    raise last_error


def _is_ready(status):
    """Index status is an object in some SDK versions and a dict in others."""
    if status is None:
        return False
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(getattr(status, "ready", False))


def pinecone_index(api_key: str, index_name: str):
    """Return a Pinecone index, creating it on first run."""
    pc = Pinecone(api_key=api_key)

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    # Wait for the index to finish provisioning before using it.
    for _ in range(90):
        try:
            if _is_ready(pc.describe_index(index_name).status):
                break
        except Exception:
            pass
        time.sleep(1)

    return pc.Index(index_name)


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def embed(client, texts, task_type):
    """
    Turn a list of strings into a list of vectors.

    task_type is "RETRIEVAL_DOCUMENT" when storing, "RETRIEVAL_QUERY" when asking.
    Gemini uses it to tune the vector for that purpose.
    """
    vectors = []
    batch = 50

    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        result = with_retry(lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=chunk,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBED_DIM,
            ),
        ))
        vectors.extend([e.values for e in result.embeddings])

    return vectors


# --------------------------------------------------------------------------
# Store / search
# --------------------------------------------------------------------------

def store_chunks(client, index, namespace, chunks):
    """
    chunks: list of {"text": str, "source": str}
    Returns how many pieces were saved.
    """
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    vectors = embed(client, texts, "RETRIEVAL_DOCUMENT")

    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append({
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {
                "text": chunk["text"][:8000],
                "source": chunk["source"],
            },
        })

    for start in range(0, len(records), 100):
        index.upsert(vectors=records[start:start + 100], namespace=namespace)

    return len(records)


def search(client, index, namespace, question, top_k=6):
    """Find the most relevant stored pieces for a question."""
    query_vector = embed(client, [question], "RETRIEVAL_QUERY")[0]

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )

    hits = []
    for match in result.get("matches", []):
        meta = match.get("metadata") or {}
        hits.append({
            "text": meta.get("text", ""),
            "source": meta.get("source", "unknown"),
            "score": match.get("score", 0.0),
        })
    return hits


def wipe(index, namespace):
    """Delete everything in one knowledge base."""
    try:
        index.delete(delete_all=True, namespace=namespace)
        return True
    except Exception:
        # Pinecone raises if the namespace does not exist yet — that is fine.
        return False


def stats(index, namespace):
    """How many pieces are stored in this knowledge base."""
    try:
        info = index.describe_index_stats()
        namespaces = info.get("namespaces") or {}
        summary = namespaces.get(namespace)
        if summary is None:
            return 0
        # Field name differs across SDK versions.
        for key in ("vector_count", "record_count"):
            if isinstance(summary, dict):
                if summary.get(key) is not None:
                    return summary[key]
            else:
                value = getattr(summary, key, None)
                if value is not None:
                    return value
        return 0
    except Exception:
        return 0


# --------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------

REFUSAL = "I could not find that in the information you gave me."

SYSTEM_RULES = """You are a helpful assistant that answers ONLY from the reference \
material provided below.

Rules:
- If the answer is in the reference material, answer clearly and directly.
- If the reference material does not contain the answer, say exactly:
  "I could not find that in the information you gave me."
  Do not guess and do not use outside knowledge.
- Keep answers short and plain. Use bullet points when listing things.
- Answer in the same language the question was asked in.
"""


def answer(client, question, hits, history=None):
    """Ask Gemini the question, giving it only the retrieved text."""
    if not hits:
        return REFUSAL, []

    blocks = []
    for i, hit in enumerate(hits, 1):
        blocks.append(f"[{i}] (from: {hit['source']})\n{hit['text']}")
    reference = "\n\n".join(blocks)

    conversation = ""
    if history:
        recent = history[-6:]
        lines = [f"{role.upper()}: {text}" for role, text in recent]
        conversation = "Earlier in this conversation:\n" + "\n".join(lines) + "\n\n"

    prompt = (
        f"{SYSTEM_RULES}\n\n"
        f"{conversation}"
        f"REFERENCE MATERIAL:\n{reference}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

    response = with_retry(lambda: client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    ))
    text = (response.text or "").strip()

    # When it refuses, nothing was actually used - showing sources would imply
    # the answer came from somewhere, which is the opposite of what happened.
    if REFUSAL.lower() in text.lower():
        return text, []

    sources = sorted({hit["source"] for hit in hits})
    return text, sources
