# TravelDude — AI-Powered Travel Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20llama3-purple?style=flat-square)](https://ollama.com)
[![APIs](https://img.shields.io/badge/APIs-OpenTripMap%20%7C%20Unsplash-orange?style=flat-square)]()

A content-based travel recommendation engine inspired by [MovieDude](https://github.com/a-partovii/MovieDude), augmented with **live open APIs**, a **local LLM layer (Ollama + llama3)** for natural-language travel planning, and a **RAG pipeline** that grounds itineraries in real traveler reviews and your own personal notes.

> **Fully local & free.** The LLM runs on your machine via Ollama — no API keys, no usage costs, no data leaving your computer.

---

## Architecture Overview

```
User Input (CLI / Jupyter)
        │
        ▼
┌──────────────────┐     ┌─────────────────────┐
│  Preference      │────▶│  Content-Based       │
│  Collector       │     │  Filtering Engine    │
│  free-text / NLP │     │  TF-IDF + cosine     │
└──────────────────┘     └──────────┬──────────┘
                                     │
        ┌────────────────────────────┘
        ▼
┌──────────────────┐     ┌─────────────────────┐
│  OpenTripMap API │────▶│  Destination DB      │
│  (live POI data) │     │  (SQLite)            │
└──────────────────┘     └──────────┬──────────┘
                                     │
        ┌────────────────────────────┘
        ▼
┌─────────────────────────────────────────────────┐
│                  RAG Layer                       │
│                                                  │
│  User query                                      │
│      │                                           │
│      ▼                                           │
│  Embed  (sentence-transformers — local/offline)  │
│      │                                           │
│      ▼                                           │
│  SQLite cosine similarity search                 │
│      │                                           │
│      ├── Top-K traveler review chunks            │
│      └── Top-K personal note chunks             │
│      │                                           │
│      ▼                                           │
│  Inject as <retrieved_context> into prompt       │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│           LLM Layer  (Ollama — llama3)           │
│   Runs 100% locally on your machine             │
│                                                  │
│  • Extract preferences from plain English       │
│  • Explain why destinations match               │
│  • Build grounded multi-day itineraries         │
│  • RAG chat — multi-turn Q&A with context       │
└─────────────────────────────────────────────────┘
        │
        ▼
   Ranked Results + Grounded Narrative Itinerary
```

---

## Open APIs Used

| API | Purpose |
|-----|---------|
| [OpenTripMap](https://opentripmap.io/docs) | POIs, attraction categories, ratings | 
| [Rest Countries](https://restcountries.com/) | Country metadata (language, currency, region) | 
| [Open-Meteo](https://open-meteo.com/) | Climate/weather data per destination | 
| [Unsplash](https://unsplash.com/developers) | Destination photos | 

> All APIs are **free and require no credit card**.  
> The LLM (llama3 via Ollama) and RAG embeddings (`all-MiniLM-L6-v2`) both run **fully offline**.

---

## Prerequisites

### 1. Ollama (local LLM server)

```bash
# Mac
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull llama3 (~4GB, one-time download)
ollama pull llama3

# Start the server (keep this running in a terminal tab)
ollama serve
```

### 2. Python dependencies

```bash
pip install -r requirements.txt
# First run auto-downloads the ~90MB sentence-transformers embedding model
```

---

## Setup & Run

### Option A — Jupyter Notebook (recommended)

```bash
# Clone repo
git clone https://github.com/ctrl-emelin/TravelDude.git
cd TravelDude

# Install dependencies
pip install -r requirements.txt

# Copy env file (no API keys required for core features)
cp .env.example .env

# Launch Jupyter
jupyter lab

# Open TravelDude.ipynb and run cells top to bottom
```

### Option B — CLI

```bash
git clone https://github.com/ctrl-emelin/TravelDude.git
cd TravelDude

pip install -r requirements.txt
cp .env.example .env

# Initialize database
python database_init.py

# Seed RAG index with demo reviews and notes
python rag/ingest.py --seed

# Run
python main.py
```

---

## Workflow

1. **Preference Collection** — Rate travel categories manually or describe your ideal trip in plain English; llama3 extracts scores automatically
2. **Normalization** — Min-Max normalize scores into a preference vector
3. **Destination Retrieval** — Pull live POI data from OpenTripMap for candidate destinations
4. **Content-Based Filtering** — TF-IDF on destination tags + cosine similarity against preference vector
5. **RAG Retrieval** — Embed the user query locally, search `rag_documents` for relevant review and note chunks
6. **LLM Augmentation** — llama3 explains why destinations match and generates a grounded multi-day itinerary
7. **RAG Chat** — Multi-turn chatbot grounded in your indexed reviews and personal notes
8. **Output** — Ranked destination list + narrative itinerary + optional Unsplash photos

---

## RAG Pipeline

The RAG layer grounds every LLM response in real traveler reviews and your own notes, rather than relying solely on llama3's training data.

### Adding your own knowledge

```bash
# Ingest personal travel notes (.md or .txt files)
# Name files by destination for best results: tokyo_tips.md, bali_notes.txt
python rag/ingest.py --notes data/notes/

# Ingest travel review CSVs (TripAdvisor-style, Kaggle datasets, etc.)
python rag/ingest.py --reviews data/reviews/

# Ingest both at once
python rag/ingest.py --all

# Check what's indexed
python rag/ingest.py --stats
```

### Notes file format (optional YAML front-matter)

```markdown
---
destination: Tokyo
title: My Tokyo Tips
---
Your notes here...
```

### Compatible review CSV columns

| Field | Accepted column names |
|---|---|
| Review text | `review`, `review_text`, `text`, `content`, `body` |
| Destination | `destination`, `city`, `location`, `place` |
| Title | `title`, `review_title`, `subject` |
| Rating | `rating`, `stars`, `score` |

---

## Database Structure

```
TravelDude.db
│
├── TABLE: Destinations
│   ├── dest_id: INTEGER (PRIMARY KEY)
│   ├── name, country, region: TEXT
│   ├── categories: TEXT       (comma-separated: beach, culture, food, ...)
│   ├── climate: TEXT
│   ├── avg_cost_level: INTEGER (1–5)
│   ├── safety_index: REAL
│   ├── rating: REAL
│   └── tags: TEXT
│
├── TABLE: Users
│   ├── user_id: VARCHAR(20) (PRIMARY KEY)
│   ├── name: VARCHAR(40)
│   └── created_at: TIMESTAMP
│
├── TABLE: User_Preferences
│   ├── user_id, dest_id: FK references
│   ├── user_rating: REAL
│   ├── visited, wishlist: BOOLEAN
│   └── rated_at: TIMESTAMP
│
├── TABLE: LLM_Logs
│   ├── prompt_hash: TEXT      ← SHA-256 hash only, raw prompt never stored
│   ├── response_summary: TEXT ← ≤200 chars
│   ├── model: TEXT            ← e.g. llama3
│   ├── tokens_used: INTEGER
│   └── created_at: TIMESTAMP  ← purged after 90 days
│
├── TABLE: rag_documents
│   ├── source_type: TEXT      (review | note)
│   ├── destination, title: TEXT
│   ├── content: TEXT          ← ~400-word chunks, 80-word overlap
│   └── metadata: TEXT         ← JSON (rating, reviewer, filename)
│
└── TABLE: rag_embeddings
    ├── doc_id: INTEGER (PRIMARY KEY)
    └── embedding: BLOB        ← 384-dim float32, all-MiniLM-L6-v2
```

---

## Project Structure

```
TravelDude/
├── data/
│   ├── notes/                 ← drop .md / .txt travel notes here
│   └── reviews/               ← drop review CSVs here
├── database/                  ← TravelDude.db (git-ignored, created at runtime)
├── docs/
│   └── privacy_audit.md
├── rag/
│   ├── __init__.py
│   ├── vector_store.py        ← SQLite embedding store + cosine retrieval
│   ├── ingest.py              ← CSV review + notes ingestion pipeline
│   └── rag_llm.py             ← RAG-enriched itinerary + TravelChat class
├── TravelDude.ipynb           ← Jupyter notebook (recommended entry point)
├── main.py                    ← CLI entry point
├── engine.py                  ← TF-IDF content-based filtering
├── llm_layer.py               ← Ollama / llama3 integration
├── api_clients.py             ← OpenTripMap, RestCountries, Open-Meteo, Unsplash
├── database_init.py           ← schema creation + destination seed data
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## Privacy & Data Flow

TBD


---

## License

MIT — see [LICENSE](LICENSE)
