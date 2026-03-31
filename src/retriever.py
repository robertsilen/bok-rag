"""Hybrid retrieval with Reciprocal Rank Fusion (RRF)."""

import time
from config import VECTOR_SEARCH_LIMIT, FULLTEXT_SEARCH_LIMIT, RRF_K, RRF_TOP_N
import db
import embedder


def hybrid_search(book_id, query_text, progress_callback=None):
    """Perform hybrid vector + fulltext search with RRF merge.
    Returns (top_chunks, diagnostics_dict).
    """
    diagnostics = {}

    def _sorted_pages(rows):
        return sorted({
            int(r["page_number"])
            for r in rows
            if r.get("page_number") is not None
        })

    def _notify(stage, payload):
        if progress_callback:
            try:
                progress_callback(stage, payload)
            except Exception:
                # Progress updates should never break retrieval.
                pass

    # Step 1: Embed query
    t0 = time.perf_counter()
    query_vec = embedder.embed_single(query_text)
    diagnostics["embed_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    _notify("embed", {"embed_time_ms": diagnostics["embed_time_ms"]})

    # Step 2: Vector search
    t0 = time.perf_counter()
    vec_results, vec_sql = db.vector_search(book_id, query_vec, VECTOR_SEARCH_LIMIT)
    diagnostics["vec_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    diagnostics["vec_sql"] = vec_sql
    diagnostics["vec_count"] = len(vec_results)
    diagnostics["vec_pages"] = _sorted_pages(vec_results)
    _notify(
        "vector",
        {
            "vec_time_ms": diagnostics["vec_time_ms"],
            "vec_count": diagnostics["vec_count"],
            "vec_pages": diagnostics["vec_pages"],
        },
    )

    # Step 3: Fulltext search
    t0 = time.perf_counter()
    ft_results, ft_sql = db.fulltext_search(book_id, query_text, FULLTEXT_SEARCH_LIMIT)
    diagnostics["ft_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    diagnostics["ft_sql"] = ft_sql
    diagnostics["ft_count"] = len(ft_results)
    diagnostics["ft_pages"] = _sorted_pages(ft_results)
    _notify(
        "fulltext",
        {
            "ft_time_ms": diagnostics["ft_time_ms"],
            "ft_count": diagnostics["ft_count"],
            "ft_pages": diagnostics["ft_pages"],
        },
    )

    # Step 4: RRF merge
    t0 = time.perf_counter()
    chunk_map = {}  # chunk_id -> chunk data
    rrf_scores = {}  # chunk_id -> score

    for rank, row in enumerate(vec_results):
        cid = row["id"]
        chunk_map[cid] = row
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    for rank, row in enumerate(ft_results):
        cid = row["id"]
        chunk_map[cid] = row
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    top_ids = sorted_ids[:RRF_TOP_N]

    top_chunks = []
    for cid in top_ids:
        chunk = chunk_map[cid].copy()
        chunk["rrf_score"] = rrf_scores[cid]
        # Add vec_dist if available
        vec_row = next((r for r in vec_results if r["id"] == cid), None)
        chunk["vec_dist"] = vec_row["vec_dist"] if vec_row else None
        top_chunks.append(chunk)

    diagnostics["rrf_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    diagnostics["rrf_count"] = len(top_chunks)

    # Count consensus hits
    vec_ids = {r["id"] for r in vec_results}
    ft_ids = {r["id"] for r in ft_results}
    consensus_ids = vec_ids & ft_ids
    diagnostics["consensus_count"] = len(consensus_ids)
    diagnostics["consensus_pages"] = sorted({
        int(chunk_map[cid]["page_number"])
        for cid in consensus_ids
        if chunk_map.get(cid) and chunk_map[cid].get("page_number") is not None
    })
    diagnostics["llm_pages"] = sorted({
        int(c["page_number"])
        for c in top_chunks
        if c.get("page_number") is not None
    })
    _notify(
        "rrf",
        {
            "rrf_time_ms": diagnostics["rrf_time_ms"],
            "rrf_count": diagnostics["rrf_count"],
            "consensus_count": diagnostics["consensus_count"],
            "consensus_pages": diagnostics["consensus_pages"],
            "llm_pages": diagnostics["llm_pages"],
        },
    )

    return top_chunks, diagnostics
