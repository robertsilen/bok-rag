# Bok-RAG

En lokal, Docker-baserad RAG-applikation för frågor mot PDF-böcker som visar MariaDBs VECTOR-funktionalitet.

## Funktioner

- **Hybrid sökning**: kombinerar vektorsökning (HNSW) och fulltextsökning med Reciprocal Rank Fusion (RRF)
- **Query rewriting**: uppföljningsfrågor skrivs om automatiskt via Claude Haiku för bättre sökträffar
- **MariaDB 11.8 LTS**: VECTOR-datatyp, VECTOR INDEX med cosinus-avstånd
- **Lokala embeddings**: nomic-embed-text via Ollama (768 dimensioner) med Apple Silicon GPU-acceleration
- **Claude API**: för intelligenta chattsvar med sidreferenser
- **Transparenspanel**: visar SQL-frågor, söktider och retrieval-statistik

## Förutsättningar

```bash
# Docker
brew install colima docker docker-compose
colima start

# Ollama (nativt för GPU-acceleration)
brew install ollama
OLLAMA_HOST=0.0.0.0:11434 ollama serve
ollama pull nomic-embed-text:latest
```

**Viktigt**: Ollama körs nativt på hosten (inte i Docker) för att utnyttja Apple Silicons Metal GPU. `OLLAMA_HOST=0.0.0.0:11434` krävs för att Docker-containern ska kunna nå Ollama via `host.docker.internal`.

## Kom igång

1. Skapa `.env` med din API-nyckel:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

2. Se till att Ollama körs: `OLLAMA_HOST=0.0.0.0:11434 ollama serve`

3. Starta:
   ```bash
   docker compose up app
   ```

4. Öppna http://localhost:8501

5. Ladda upp en PDF och börja chatta!

Första uppstarten tar några minuter (image build).

## Konfiguration

Valfria variabler i `.env`:

| Variabel | Standard | Beskrivning |
|----------|----------|-------------|
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude-modell (Haiku → Sonnet → Opus: snabbare/billigare → mer kapabel) |
| `CHUNK_TARGET_SIZE` | `500` | Målstorlek per chunk (ord) |
| `CHUNK_MAX_SIZE` | `1000` | Max storlek per chunk (ord) |

## CLI-ingestion

```bash
docker compose exec app python /app/src/ingest.py \
  --file "/data/book.pdf" \
  --title "Valfri titel" \
  --max-pages 30
```

## Arkitektur

```
Mac (host)
 ├─ Ollama (nativt, Metal GPU)
 └─ Docker/Colima
     ├─ bookrag-db      (MariaDB 11.8 LTS)
     └─ bookrag-app     (Streamlit + ingestion + retrieval + chat)
```

## Felsökning

- **Ollama unreachable**: kontrollera att `OLLAMA_HOST=0.0.0.0:11434 ollama serve` körs (utan `0.0.0.0` lyssnar Ollama bara på localhost, oåtkomligt från Docker). Testa med `curl http://localhost:11434/api/tags`
- **Modell saknas**: `ollama pull nomic-embed-text:latest` (dras även automatiskt vid uppstart)
- **Ogiltig API-nyckel**: chatten visar felmeddelande, ingestion fungerar utan nyckel
- **Stor PDF (230+ MB)**: kan ta några minuter med GPU, progress visas i UI

## Referenser

- [MariaDB Vector Overview](https://mariadb.com/kb/en/vectors/) — VECTOR-datatyp och VECTOR INDEX
- [MariaDB Vector Functions](https://mariadb.com/kb/en/vector-functions/) — VEC_DISTANCE, VEC_ToText m.fl.
- [MariaDB Full-Text Search](https://mariadb.com/kb/en/full-text-index-overview/) — MATCH AGAINST med FULLTEXT INDEX
- [Hybrid Search with MariaDB](https://mariadb.com/resources/blog/how-to-hybrid-search-with-mariadb/) — Reciprocal Rank Fusion (RRF) med vektor + fulltext
