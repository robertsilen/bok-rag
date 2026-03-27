CREATE DATABASE IF NOT EXISTS bookrag;
USE bookrag;

CREATE TABLE IF NOT EXISTS books (
  id INT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(500) NOT NULL,
  filename VARCHAR(500) NOT NULL,
  total_pages INT NOT NULL,
  chunk_count INT DEFAULT 0,
  ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
  id INT AUTO_INCREMENT PRIMARY KEY,
  book_id INT NOT NULL,
  page_number INT NOT NULL,
  paragraph_index INT NOT NULL,
  chunk_text TEXT NOT NULL,
  context_before TEXT,
  context_after TEXT,
  embedding VECTOR(768) NOT NULL,
  VECTOR INDEX (embedding) M=8 DISTANCE=cosine,
  FOREIGN KEY (book_id) REFERENCES books(id) ON DELETE CASCADE,
  INDEX idx_book_page (book_id, page_number),
  FULLTEXT INDEX idx_fulltext (chunk_text)
) ENGINE=InnoDB;
