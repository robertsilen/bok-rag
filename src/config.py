import os

# Database
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "bookrag")
DB_PASSWORD = os.getenv("DB_PASSWORD", "bookrag_pass")
DB_NAME = os.getenv("DB_NAME", "bookrag")

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "host.docker.internal")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
EMBEDDING_MODEL = "nomic-embed-text:latest"
EMBEDDING_DIM = 768
EMBEDDING_BATCH_SIZE = 1

# Chunking
CHUNK_TARGET_SIZE = int(os.getenv("CHUNK_TARGET_SIZE", "500"))
CHUNK_MIN_SIZE = 50
CHUNK_MAX_SIZE = int(os.getenv("CHUNK_MAX_SIZE", "1000"))

# Retrieval
VECTOR_SEARCH_LIMIT = 20
FULLTEXT_SEARCH_LIMIT = 20
RRF_K = 60
RRF_TOP_N = 8

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = (
    "Du är en hjälpsam assistent som svarar på frågor om en bok. "
    "Svara på svenska. Referera alltid till sidnummer. "
    "Basera svaret enbart på den medskickade kontexten. "
    "Om kontexten inte räcker, säg det."
)

# Ingestion
INSERT_BATCH_SIZE = 100
MAX_UPLOAD_SIZE_MB = 300
