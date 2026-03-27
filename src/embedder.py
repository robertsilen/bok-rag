import requests
from config import OLLAMA_BASE_URL, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

# nomic-embed-text has an 8192 token context window.
# Progressively truncate if embedding fails.
MAX_WORDS = 6000


def _clean(text):
    """Normalize text to reduce token count: collapse repeated punctuation."""
    import re
    # Collapse long runs of dots, dashes, asterisks etc. (e.g. table of contents)
    text = re.sub(r'[.\s]{5,}', ' ... ', text)
    text = re.sub(r'[-*=_]{5,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _truncate(text, max_words=MAX_WORDS):
    text = _clean(text)
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def _embed_one(text):
    """Embed a single text, truncating if needed."""
    truncated = _truncate(text)
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": [truncated], "options": {"num_ctx": 8192}},
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding failed: {resp.status_code} {resp.text[:300]}")
    return resp.json()["embeddings"][0]


def embed_texts(texts, progress_callback=None):
    """Embed a list of texts using Ollama. Returns list of embedding vectors."""
    all_embeddings = []
    total = len(texts)
    for i in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = [_truncate(t) for t in texts[i : i + EMBEDDING_BATCH_SIZE]]
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": batch, "options": {"num_ctx": 8192}},
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            if "embeddings" in data and len(data["embeddings"]) == len(batch):
                all_embeddings.extend(data["embeddings"])
            else:
                # Batch returned unexpected data — fall back to one-by-one
                for t in batch:
                    all_embeddings.append(_embed_one(t))
        else:
            # Batch failed — fall back to one-by-one
            for t in batch:
                all_embeddings.append(_embed_one(t))
        if progress_callback:
            progress_callback(min(i + len(batch), total), total)
    return all_embeddings


def embed_single(text):
    """Embed a single text. Returns one embedding vector."""
    return _embed_one(text)
