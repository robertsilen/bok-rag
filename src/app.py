"""Bok-RAG — Streamlit UI."""

import os
import sys
import time
import threading

import streamlit as st

# Ensure src is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db
import ingest
import retriever
import llm
from config import EMBEDDING_DIM

st.set_page_config(page_title="Bok-RAG", layout="wide")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingestion_running" not in st.session_state:
    st.session_state.ingestion_running = False
if "show_upload" not in st.session_state:
    st.session_state.show_upload = False


# ---------------------------------------------------------------------------
# Helper: diagnostics expander
# ---------------------------------------------------------------------------
def _show_diagnostics(diag, top_chunks=None):
    with st.expander("🔍 MariaDB under huven"):
        if "search_query" in diag:
            st.markdown(f"**Omskriven sökfråga:** {diag['search_query']}")

        st.markdown("**SQL-frågor som kördes**")
        if "vec_sql" in diag:
            st.code(diag["vec_sql"], language="sql")
        if "ft_sql" in diag:
            st.code(diag["ft_sql"], language="sql")

        st.markdown("**Söktider**")
        timing = (
            f"```\n"
            f"Vektorsökning:    {diag.get('vec_time_ms', '?')} ms\n"
            f"Fulltextsökning:  {diag.get('ft_time_ms', '?')} ms\n"
            f"RRF-merge:        {diag.get('rrf_time_ms', '?')} ms\n"
            f"LLM-svar:         {diag.get('llm_time_ms', '?')} ms\n"
            f"Totalt:           {diag.get('total_time_ms', '?')} ms\n"
            f"```"
        )
        st.markdown(timing)

        st.markdown("**Retrieval-resultat**")
        st.text(
            f"Vektorsökning: {diag.get('vec_count', '?')} träffar\n"
            f"Fulltextsökning: {diag.get('ft_count', '?')} träffar\n"
            f"Efter RRF: {diag.get('rrf_count', '?')} skickade till LLM\n"
            f"Konsensus-träffar: {diag.get('consensus_count', '?')}"
        )

        if top_chunks:
            st.markdown("**Top 3 chunks**")
            for i, c in enumerate(top_chunks[:3], 1):
                vec_dist_str = f"{c['vec_dist']:.4f}" if c.get('vec_dist') is not None else "—"
                st.text(
                    f"{i}. Chunk #{c['id']} | Sida {c['page_number']} | "
                    f"Cosinus-avstånd: {vec_dist_str} | RRF: {c['rrf_score']:.5f}\n"
                    f"   {c['chunk_text'][:100]}..."
                )


# ---------------------------------------------------------------------------
# Helper: get the current (single) active book
# ---------------------------------------------------------------------------
def _get_active_book():
    """Return the most recently ingested book with chunks, or None."""
    books = db.get_books()
    for b in books:
        if b["chunk_count"] > 0:
            return b
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Bok-RAG")

    active_book = _get_active_book()

    if active_book and not st.session_state.show_upload:
        # A book is loaded — show info
        st.success(f"📖 {active_book['title']}")
        st.caption(f"{active_book['total_pages']} sidor · {active_book['chunk_count']} chunks")

        st.divider()

        # Database statistics
        st.subheader("Databasstatistik")
        st.metric("Chunks i databasen", active_book["chunk_count"])
        st.text(f"Vektorindex: {EMBEDDING_DIM} dimensioner, cosine, M=8")
        stats = db.get_db_stats(active_book["id"])
        total_mb = stats["total_size"] / (1024 * 1024)
        st.text(f"Databasstorlek (chunks): {total_mb:.1f} MB")

        st.divider()

        # Replace book button
        if st.button("📁 Ladda ny bok"):
            st.session_state.show_upload = True
            st.rerun()

    else:
        # No book loaded, or user wants to upload — show upload form
        if active_book:
            st.info(f"Nuvarande bok: {active_book['title']}")
            st.warning("Att ladda en ny bok ersätter den nuvarande.")
            if st.button("← Avbryt"):
                st.session_state.show_upload = False
                st.rerun()
            st.divider()

        st.subheader("Ladda upp en bok")
        uploaded_file = st.file_uploader("Välj PDF-fil (max 300 MB)", type=["pdf"])
        max_pages = st.number_input("Antal sidor att bearbeta (0 = alla)", min_value=0, value=0, step=1)
        if max_pages == 0:
            max_pages = None

        if st.button("Ladda upp & Bearbeta", disabled=uploaded_file is None or st.session_state.ingestion_running):
            if uploaded_file is not None:
                save_path = os.path.join("/data", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                title = os.path.splitext(uploaded_file.name)[0]

                # Delete existing books first (single-book model)
                if active_book:
                    db.delete_book(active_book["id"])

                _progress = {"status": "Startar bearbetning...", "pct": 0.0, "running": True}
                st.session_state.ingestion_running = True

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def run_ingestion():
                    def progress_cb(stage, detail):
                        _progress["status"] = detail
                        if stage == "extract" and "/" in detail:
                            try:
                                parts = detail.split()
                                frac = parts[-1]
                                cur, tot = frac.split("/")
                                _progress["pct"] = 0.30 * int(cur) / int(tot)
                            except (ValueError, IndexError, ZeroDivisionError):
                                pass
                        elif stage == "chunk":
                            _progress["pct"] = 0.33
                        elif stage == "embed" and "/" in detail:
                            try:
                                nums = detail.replace("...", "").split()
                                frac = nums[-1]
                                cur, tot = frac.split("/")
                                _progress["pct"] = 0.35 + 0.55 * int(cur) / int(tot)
                            except (ValueError, IndexError, ZeroDivisionError):
                                pass
                        elif stage == "store":
                            _progress["pct"] = 0.92
                        elif stage == "done":
                            _progress["pct"] = 1.0
                    try:
                        ingest.ingest_pdf(save_path, title=title, max_pages=max_pages, progress_callback=progress_cb)
                        _progress["status"] = "✅ Klar!"
                        _progress["pct"] = 1.0
                    except Exception as e:
                        _progress["status"] = f"❌ Fel: {e}"
                        _progress["error"] = True
                    finally:
                        _progress["running"] = False

                thread = threading.Thread(target=run_ingestion, daemon=True)
                thread.start()

                while thread.is_alive():
                    progress_bar.progress(min(_progress["pct"], 1.0))
                    status_text.info(_progress["status"])
                    time.sleep(0.3)
                progress_bar.progress(min(_progress["pct"], 1.0))
                status_text.info(_progress["status"])

                st.session_state.ingestion_running = False

                if _progress.get("error"):
                    st.error(_progress["status"])
                else:
                    st.session_state.show_upload = False
                    st.session_state.chat_history = []
                    st.rerun()


# ---------------------------------------------------------------------------
# Main area: Chat
# ---------------------------------------------------------------------------

active_book = _get_active_book()
book_ready = active_book is not None

# New chat button
if book_ready and st.session_state.chat_history:
    if st.button("🔄 Ny chatt"):
        st.session_state.chat_history = []
        st.rerun()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "pages" in msg:
            pages_str = ", ".join(str(p) for p in msg["pages"])
            st.caption(f"📄 Sidor: {pages_str}")
            if "diagnostics" in msg:
                _show_diagnostics(msg["diagnostics"], msg.get("top_chunks"))

# Chat input
if prompt := st.chat_input("Ställ en fråga om boken...", disabled=not book_ready):
    if not book_ready:
        st.warning("Ladda upp en bok först.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Söker och genererar svar..."):
                t_total_start = time.perf_counter()
                try:
                    # Rewrite query if there's conversation history
                    llm_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history[:-1]
                        if m["role"] in ("user", "assistant")
                    ]
                    search_query = llm.rewrite_query(prompt, llm_history)

                    top_chunks, diagnostics = retriever.hybrid_search(active_book["id"], search_query)
                    if search_query != prompt:
                        diagnostics["search_query"] = search_query

                    if not top_chunks:
                        st.warning("Inga relevanta stycken hittades.")
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "Inga relevanta stycken hittades i boken.",
                        })
                    else:
                        t_llm_start = time.perf_counter()
                        answer, page_numbers = llm.chat(prompt, top_chunks, history=llm_history)
                        diagnostics["llm_time_ms"] = round((time.perf_counter() - t_llm_start) * 1000, 1)
                        diagnostics["total_time_ms"] = round((time.perf_counter() - t_total_start) * 1000, 1)

                        st.markdown(answer)
                        pages_str = ", ".join(str(p) for p in page_numbers)
                        st.caption(f"📄 Sidor: {pages_str}")
                        _show_diagnostics(diagnostics, top_chunks)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "pages": page_numbers,
                            "diagnostics": diagnostics,
                            "top_chunks": top_chunks,
                        })

                except ValueError as e:
                    st.error(str(e))
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": str(e),
                    })
                except Exception as e:
                    st.error(f"Ett fel uppstod: {e}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Ett fel uppstod: {e}",
                    })
