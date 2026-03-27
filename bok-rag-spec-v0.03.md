# Bok-RAG — Technical Specification (v0.03)

## Översikt

**Bok-RAG** är en lokal, Docker-baserad RAG-applikation för frågor mot PDF-böcker som visar MariaDBs VECTOR-funktionalitet.

Systemet:
1. extraherar text från PDF,
2. delar texten i chunks,
3. skapar embeddings med lokal Ollama,
4. lagrar metadata + chunks + vektorer i MariaDB 11.8 LTS,
5. använder Claude API för chattsvar baserat på hämtade chunks,
6. visar transparens kring vektorsökning och databasoperationer i UI.

---

## Scope

Ingår:
- Svenskt UI i Streamlit.
- Chatt med sidreferenser.
- Hybrid retrieval (vektor + fulltext) med Reciprocal Rank Fusion (RRF).
- MariaDB-transparenspanel: visar SQL-frågor, söktider, lagringsstatistik.
- Ingestion via UI och CLI.
- Testläge: importera endast första N sidor.
- README.md som beskriver vad appen är, one-command startup (`docker compose up app`) och hur användaren kommer igång.

Ingår inte (v0.03):
- Entity index med NER och Wikidata/Wikipedia-länkning.
- Flerbokssökning (en bok aktiv åt gången).

---

## Namn och språk

- Appens namn: **Bok-RAG**.
- All text i UI: **svenska**.

---

## Arkitektur

```text
Mac (host)
 ├─ Ollama (nativt, port 11434) — kör med Metal GPU-acceleration på Apple Silicon
 └─ Docker/Colima
     ├─ bookrag-db      (MariaDB 11.8 LTS)
     └─ bookrag-app     (Streamlit + ingestion + retrieval + chat)
```

**Ollama körs nativt på hosten** (inte i Docker) för att utnyttja Apple Silicons Metal GPU-acceleration. Docker/Colima kör en Linux-VM utan GPU-tillgång, vilket gör embedding markant långsammare. Appen i Docker når hostens Ollama via `host.docker.internal:11434`.

Compose-nätverk: db och app delar ett internt nätverk. Appen refererar till `db:3306` via service-namn och `host.docker.internal:11434` för Ollama.

---

## Databas (MariaDB 11.8 LTS)

Använder VECTOR-datatyp och VECTOR INDEX med HNSW-algoritm.

```sql
CREATE DATABASE IF NOT EXISTS bookrag;
USE bookrag;

CREATE TABLE books (
  id INT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(500) NOT NULL,
  filename VARCHAR(500) NOT NULL,
  total_pages INT NOT NULL,
  chunk_count INT DEFAULT 0,
  ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chunks (
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
```

Kommentarer:
- `VECTOR(768)` matchar nomic-embed-text:s dimensionalitet.
- `VECTOR INDEX ... M=8 DISTANCE=cosine` skapar ett HNSW-index med M=8 grannar och cosinusavstånd. Syntaxen är verifierad mot MariaDB 11.8 GA-dokumentation.
- `chunk_count` i `books` uppdateras efter avslutad ingestion för snabb statistik i UI.
- Det kan bara finnas **ett** VECTOR INDEX per tabell. Den indexerade kolumnen måste vara NOT NULL.

---

## Ingestion-pipeline

### 1) Extraktion + textdump
- Bibliotek: `pymupdf` (`fitz`).
- Filtrera bort bildblock (behåll enbart textblock).
- Heuristik för header/footer: ignorera text som upprepas identiskt på varje sida (inom rimlig marginal) nära sidans övre/nedre kant.
- Normalisera whitespace + merge av hyphen-breaks (t.ex. "konsti-\ntution" → "konstitution").
- Skriv alltid `.txt` bredvid PDF i data-katalogen (för felsökning).

### 2) Chunking

Parametrar (konfigurerbara i `config.py`):

| Parameter | Standardvärde | Beskrivning |
|-----------|---------------|-------------|
| `CHUNK_TARGET_SIZE` | 500 tokens | Målstorlek per chunk |
| `CHUNK_MIN_SIZE` | 50 tokens | Under detta slås ihop med föregående |
| `CHUNK_MAX_SIZE` | 1000 tokens | Över detta delas vid meningsgräns |

Strategi:
- Paragrafbaserat: varje paragraf (dubbel radbrytning) är en kandidat-chunk.
- Korta stycken (< `CHUNK_MIN_SIZE`) slås ihop med nästa stycke inom samma sida.
- Långa stycken (> `CHUNK_MAX_SIZE`) delas vid meningsgräns (punkt + mellanslag).
- `context_before`: sista meningen i föregående chunk (samma bok).
- `context_after`: första meningen i nästa chunk (samma bok).
- `page_number`: varje chunk tillhör den sida där den börjar.
- `paragraph_index`: löpande index per sida, börjar på 0.

Tokenräkning: approximera med `len(text.split())` — tillräckligt för chunking-beslut, undviker tokenizer-beroende.

### 3) Embedding
- Modell: `nomic-embed-text:latest` via Ollama.
- Embeddar varje chunks text (inte context_before/after).
- Batch-storlek: 1 chunk per Ollama-anrop (undviker att flera texter delar samma kontextfönster).
- **Kontextgräns**: nomic-embed-text har 8192 tokens kontextfönster. Skicka `"options": {"num_ctx": 8192}` explicit i embed-anropet. Texter trunkeras till ~6000 ord innan embedding. Om batch-anrop misslyckas faller koden tillbaka på en-och-en.
- **Textnormalisering före embedding**: innehållsförteckningar och liknande med långa sekvenser av punkter (`......`) tokeniseras till enorma mängder tokens trots få ord. Före embedding: kollapsa `[.\s]{5,}` till `...`, kollapsa `[-*=_]{5,}` till mellanslag, normalisera whitespace.
- Realtidsprogress i UI: "Embeddar chunk 47 / 312..."
- Ingestion sker server-side. Om användaren stänger fliken fortsätter processen. Vid nästa besök visas boken som "bearbetar..." tills den är klar.

### 4) Lagring
- Skapa rad i `books` (med `chunk_count=0`).
- Batch-inserts i `chunks` (100 rader per INSERT-sats).
- Embeddings konverteras med `VEC_FromText('[0.1, 0.2, ...]')` till binärt VECTOR-format.
- Efter avslutad ingestion: `UPDATE books SET chunk_count = (SELECT COUNT(*) FROM chunks WHERE book_id = ...)`.

### 5) Felhantering vid ingestion
- Om processen misslyckas: `DELETE FROM books WHERE id = ...` (CASCADE raderar chunks).
- Visa felmeddelande i UI med detaljer.

---

## Retrieval + Chat

### Hybrid sökning med Reciprocal Rank Fusion (RRF)

Vid varje fråga:

**Steg 1 — Skapa embedding** för frågan via Ollama (samma modell som vid ingestion).

**Steg 2 — Vektorsökning** (top 20 chunks):
```sql
SELECT id, book_id, page_number, chunk_text, context_before, context_after,
       VEC_DISTANCE_COSINE(embedding, VEC_FromText(@query_vec)) AS vec_dist
FROM chunks
WHERE book_id = @active_book
ORDER BY vec_dist ASC
LIMIT 20
```

**Steg 3 — Fulltextsökning** (top 20 chunks):
```sql
SELECT id, book_id, page_number, chunk_text, context_before, context_after,
       MATCH(chunk_text) AGAINST(@query_text) AS ft_score
FROM chunks
WHERE book_id = @active_book
  AND MATCH(chunk_text) AGAINST(@query_text)
ORDER BY ft_score DESC
LIMIT 20
```

**Steg 4 — RRF-merge:**
- Varje chunk tilldelas en partiell RRF-poäng per källa: `1 / (k + rank)`.
- `k = 60` (standard från forskningen, rekommenderat av MariaDB-dokumentationen).
- Summera poäng per chunk-ID. Chunks som finns i båda listorna får högre totalpoäng.
- Sortera fallande efter totalpoäng.
- Ta top 8 chunks som kontext till LLM.

RRF-implementationen följer mönstret i MariaDBs officiella dokumentation ("Optimizing Hybrid Search Query with Reciprocal Rank Fusion").

**Steg 5 — Skicka till Claude:**
- Systemmeddelande: "Du är en hjälpsam assistent som svarar på frågor om en bok. Svara på svenska. Referera alltid till sidnummer. Basera svaret enbart på den medskickade kontexten. Om kontexten inte räcker, säg det."
- Kontext: de 8 bästa chunksarna, var och en markerad med sidnummer, med `context_before` och `context_after` inkluderade för bättre sammanhang.
- Användarens fråga.

**Steg 6 — Visa svar i UI:**
- Claudes svar (markdown-formaterat).
- Under svaret: "📄 Sidor: 12, 45, 46, 112" — sidnummer som ingick i kontexten.
- Expanderbar transparenssektion (se nedan).

### LLM-konfiguration
- Standardmodell: `claude-sonnet-4-6` (styrs via `ANTHROPIC_MODEL` i `.env`).
- Kommentar i README: Haiku → Sonnet → Opus (snabbare/billigare → mer kapabel).
- Vid ogiltig/saknad API-nyckel: visa felmeddelande i chatten: "Ogiltig eller saknad ANTHROPIC_API_KEY. Kontrollera din .env-fil."
- Vid rate limit: visa meddelande och föreslå att vänta.

---

## MariaDB-transparenspanel

Syfte: Bok-RAG ska vara en showcase för MariaDBs vektorfunktioner — inte bara en app som råkar använda MariaDB. Transparenspanelen gör de underliggande databasoperationerna synliga.

### Per chattsvar: expanderbar sektion

Titel: **"🔍 MariaDB under huven"** (expander, default stängd).

Innehåll:

1. **SQL-frågor som kördes**
   - Den faktiska vektorsöknings-SQL:en (med parametrar).
   - Den faktiska fulltextsöknings-SQL:en.
   - Formaterat med syntax highlighting.

2. **Söktider**
   ```
   Vektorsökning:    12.3 ms
   Fulltextsökning:   4.1 ms
   RRF-merge:         0.2 ms
   LLM-svar:        1.4 s
   Totalt:           1.42 s
   ```

3. **Retrieval-resultat**
   - Antal träffar: vektor (20) / fulltext (14) / efter RRF (8 skickade till LLM).
   - Antal chunks som förekom i båda sökningarna (konsensus-träffar).

4. **Top 3 chunks med detaljer**
   - Chunk-ID, sidnummer, cosinus-avstånd, RRF-poäng.
   - Kort utdrag (första 100 tecken av chunk_text).

### Sidebar: databas-statistik

Under bokvalet i sidomenyn:

- **Chunks i databasen**: totalt antal för vald bok.
- **Vektorindex**: dimensioner (768), avståndsfunktion (cosine), M-värde (8).
- **Databasstorlek**: hämta från `information_schema.TABLES` (data_length + index_length).

---

## Streamlit-UI (svenska)

### Sidomenyn

- Titel: **Bok-RAG**
- **En bok åt gången** — ingen dropdown. Enkel modell:
  - Om ingen bok är laddad: visa uppladdningssektionen.
  - Om en bok är laddad: visa bokens titel, databasstatistik, och knapp **Ladda ny bok** (ersätter den befintliga — bekräftelsedialog: "Detta ersätter den nuvarande boken. Fortsätt?").
- Sektion **Ladda upp en bok** (visas när ingen bok finns, eller efter "Ladda ny bok"):
  - Fält: **Välj PDF-fil (max 300 MB)**.
  - Fält: Antal sidor att bearbeta (standard: alla).
  - Knapp: **Ladda upp & Bearbeta**.
  - Ingestion triggas **enbart** av knappen — aldrig automatiskt.
  - Max storlek: 300 MB. Tydligt felmeddelande vid större filer.
  - Titel härledas från filnamn (ingen manuell inmatning).
- Under bearbetning — realtidsstatus med **progressbar** och statustext (uppdateras var 0.3s i UI-tråden):
  - "Extraherar text... sida 47/412" (0–30%)
  - "312 chunks skapade" (33%)
  - "Embeddar chunk 47/312..." (35–90%) — detta är det långsammaste steget, viktigast att visa i realtid
  - "Sparar i MariaDB..." (92%)
  - "✅ Klar!" (100%)
- Sektion **Databasstatistik** (se transparenspanel ovan).

### Huvudyta

- Enda vy: **Chatt**.
- Chattfältet ligger fixerat längst ner.
- Nya svar skjuter upp äldre meddelanden.
- Varje assistentsvar:
  - Markdown-formaterat svar.
  - Sidnummer som användes.
  - Expanderbar **"🔍 MariaDB under huven"**.
- Chatthistorik: nollställs vid ny bok.
- Knapp **"🔄 Ny chatt"** ovanför chatthistoriken (visas bara när det finns historik). Nollställer chatten utan att påverka boken.

---

## Docker/Colima: One-command startup

### Förutsättningar

```bash
# Docker
brew install colima docker docker-compose
colima start

# Ollama (nativt på hosten för GPU-acceleration)
brew install ollama
OLLAMA_HOST=0.0.0.0:11434 ollama serve   # MÅSTE lyssna på 0.0.0.0, annars kan Docker inte nå Ollama
ollama pull nomic-embed-text:latest       # ladda ner embedding-modellen
```

**Viktigt**: Ollama måste köras nativt på hosten (inte i Docker) för att utnyttja Apple Silicons Metal GPU. Docker/Colima kör en Linux-VM utan tillgång till Metal, vilket gör embedding ~10x långsammare. `OLLAMA_HOST=0.0.0.0:11434` krävs — utan detta lyssnar Ollama bara på `127.0.0.1` som är oåtkomligt från Docker-containern.

### .env (enda manuella config)

```env
ANTHROPIC_API_KEY=sk-ant-...
```

Valfria variabler (med standardvärden i koden):
```env
ANTHROPIC_MODEL=claude-sonnet-4-6
CHUNK_TARGET_SIZE=500
CHUNK_MAX_SIZE=1000
```

DB-credentials sätts internt i compose — exponeras inte i `.env`.

### Starta

```bash
docker compose up app
```

### Docker volumes

```yaml
volumes:
  mariadb-data:    # Persistent DB-data mellan restarts
```

Ollama-modeller lagras nativt på hosten (~270 MB) och hanteras av `ollama` direkt.

### Uppstartsekvens (startup.py)

1. Vänta på MariaDB (retry-loop, max 60s, kontrollerar TCP + `SELECT 1`).
2. Vänta på Ollama på hosten (retry-loop, max 60s, kontrollerar `host.docker.internal:11434/api/tags`).
3. Kontrollera om `nomic-embed-text` finns — om inte, kör `ollama pull` via API.
4. Starta Streamlit.

### Definition of done

På en ren maskin med Docker/Colima och Ollama installerat:
1. Klona repo.
2. Skapa `.env` med `ANTHROPIC_API_KEY`.
3. Starta Ollama: `OLLAMA_HOST=0.0.0.0:11434 ollama serve`.
4. Kör `docker compose up app`.
5. Öppna `http://localhost:8501`.
6. Ladda upp PDF (testat med 230 MB, 400+ sidor) och chatta — utan manuella steg.

---

## CLI (ingest)

```bash
docker compose exec app python /app/src/ingest.py \
  --file "/data/book.pdf" \
  --title "Valfri titel" \
  --max-pages 30
```

Om `--title` utelämnas härledas titeln från filnamnet.

---

## Drift-hints / vanliga fel

1. **`docker compose` saknas** — installera Docker Desktop eller via brew.
2. **`mariadb_config not found` vid build** — Dockerfile måste installera `libmariadb-dev` + build-essentials.
3. **Ollama unreachable** — kontrollera att `OLLAMA_HOST=0.0.0.0:11434 ollama serve` körs nativt på hosten. Utan `0.0.0.0` lyssnar Ollama bara på localhost, oåtkomligt från Docker. Appen i Docker når Ollama via `host.docker.internal:11434`. Kör `curl http://localhost:11434/api/tags` för att verifiera.
4. **Model not found** — kör `ollama pull nomic-embed-text:latest` på hosten. Modellen dras även automatiskt via `startup.py` om den saknas.
5. **Ingestion körs om vid chat** — ingestion är knappstyrd, inte auto.
6. **Första uppstart tar tid** — image build + modellnedladdning kan ta 5–10 minuter.
7. **Ogiltig API-nyckel** — chatten visar felmeddelande. Ingestion fungerar utan API-nyckel.
8. **Stor PDF (230+ MB)** — ingestion kan ta 20–60 minuter beroende på sidantal och hårdvara. Progress visas i UI. Processen körs server-side och avbryts inte vid stängd flik.
9. **`.streamlit/config.toml` ignoreras** — filen måste kopieras explicit i Dockerfile (`COPY .streamlit/ /app/.streamlit/`), annars används Streamlits standardvärden (t.ex. 200 MB upload-gräns istället för 300 MB).

---

## Projektstruktur

```text
book-rag/
├── docker-compose.yml
├── Dockerfile
├── init.sql
├── requirements.txt
├── README.md
├── .env.example
├── data/
├── src/
│   ├── app.py          # Streamlit-UI
│   ├── config.py       # Alla konfigurationsparametrar
│   ├── db.py           # MariaDB-anslutning och queries
│   ├── ingest.py       # Ingestion-pipeline (CLI + importeras av app.py)
│   ├── embedder.py     # Ollama embedding-klient
│   ├── retriever.py    # Hybrid sökning + RRF
│   ├── llm.py          # Claude API-klient
│   └── startup.py      # Healthcheck + modell-pull + Streamlit-start
└── .streamlit/
    └── config.toml     # VIKTIGT: Måste kopieras i Dockerfile (COPY .streamlit/ /app/.streamlit/)
```

---

## Ändringslogg

| Version | Ändringar |
|---------|-----------|
| v0.01 | Ursprungligt utkast. |
| v0.02 | One-command startup, ingestion-regler, drift-hints. |
| v0.03 | Chunking-parametrar specificerade (target 500, min 50, max 1000 tokens). RRF-strategi definierad (k=60, top 8 chunks till LLM). MariaDB-transparenspanel tillagd med SQL-visning, söktider och statistik. Server-side ingestion (fortsätter vid stängd flik). Felhantering för API-nyckel och rate limits. Batch-embedding (10 chunks/anrop). context_before/context_after skickas till LLM. Cleanup vid misslyckad ingestion. VECTOR INDEX-begränsning dokumenterad (en per tabell). Ollama flyttad till nativ host-installation för Apple Silicon Metal GPU-acceleration (Docker-VM saknar GPU-tillgång). Progressbar med realtidsstatus under ingestion. `.streamlit/config.toml` måste kopieras explicit i Dockerfile. |
