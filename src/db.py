import mariadb
import json
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME


def get_connection():
    return mariadb.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )


def get_books():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, title, filename, total_pages, chunk_count, ingested_at FROM books ORDER BY ingested_at DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_book(book_id):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM books WHERE id = ?", (book_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def create_book(title, filename, total_pages):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO books (title, filename, total_pages, chunk_count) VALUES (?, ?, ?, 0)",
        (title, filename, total_pages),
    )
    conn.commit()
    book_id = cur.lastrowid
    cur.close()
    conn.close()
    return book_id


def insert_chunks_batch(chunks):
    """Insert a batch of chunks. Each chunk is a dict with keys:
    book_id, page_number, paragraph_index, chunk_text, context_before, context_after, embedding
    """
    if not chunks:
        return
    conn = get_connection()
    cur = conn.cursor()
    sql = (
        "INSERT INTO chunks (book_id, page_number, paragraph_index, chunk_text, "
        "context_before, context_after, embedding) VALUES "
    )
    placeholders = []
    params = []
    for c in chunks:
        vec_str = json.dumps(c["embedding"])
        placeholders.append("(?, ?, ?, ?, ?, ?, VEC_FromText(?))")
        params.extend([
            c["book_id"], c["page_number"], c["paragraph_index"],
            c["chunk_text"], c["context_before"], c["context_after"], vec_str,
        ])
    sql += ", ".join(placeholders)
    cur.execute(sql, tuple(params))
    conn.commit()
    cur.close()
    conn.close()


def update_book_chunk_count(book_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE books SET chunk_count = (SELECT COUNT(*) FROM chunks WHERE book_id = ?) WHERE id = ?", (book_id, book_id))
    conn.commit()
    cur.close()
    conn.close()


def delete_book(book_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM books WHERE id = ?", (book_id,))
    conn.commit()
    cur.close()
    conn.close()


def vector_search(book_id, query_vec, limit=20):
    vec_str = json.dumps(query_vec)
    sql = (
        "SELECT id, book_id, page_number, chunk_text, context_before, context_after, "
        "VEC_DISTANCE_COSINE(embedding, VEC_FromText(?)) AS vec_dist "
        "FROM chunks WHERE book_id = ? "
        "ORDER BY vec_dist ASC LIMIT ?"
    )
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(sql, (vec_str, book_id, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows, sql.replace("?", "{}").format(repr(vec_str[:40] + "..."), book_id, limit)


def fulltext_search(book_id, query_text, limit=20):
    sql = (
        "SELECT id, book_id, page_number, chunk_text, context_before, context_after, "
        "MATCH(chunk_text) AGAINST(?) AS ft_score "
        "FROM chunks WHERE book_id = ? "
        "AND MATCH(chunk_text) AGAINST(?) "
        "ORDER BY ft_score DESC LIMIT ?"
    )
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(sql, (query_text, book_id, query_text, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows, sql.replace("?", "{}").format(repr(query_text), book_id, repr(query_text), limit)


def get_db_stats(book_id):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT data_length, index_length FROM information_schema.TABLES "
        "WHERE table_schema = ? AND table_name = 'chunks'",
        (DB_NAME,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return {
            "data_size": row["data_length"],
            "index_size": row["index_length"],
            "total_size": row["data_length"] + row["index_length"],
        }
    return {"data_size": 0, "index_size": 0, "total_size": 0}
