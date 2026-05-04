# TravelDude — AI-Powered Travel Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![APIs](https://img.shields.io/badge/APIs-OpenTripMap%20%7C%20Unsplash%20%7C%20Anthropic-orange?style=flat-square)]()

A content-based travel recommendation engine inspired by [MovieDude](https://github.com/a-partovii/MovieDude), augmented with **live open APIs**, an **LLM layer (Claude)** for natural-language travel planning, and a **RAG pipeline** that grounds itineraries in real traveler reviews and your own personal notes.

---

## Architecture Overview

```
User Input (CLI / Web UI)
        │
        ▼
┌──────────────────┐     ┌─────────────────────┐
│  Preference      │────▶│  Content-Based       │
│  Collector       │     │  Filtering Engine    │
│  (quasi-NLP)     │     │  (TF-IDF + cosine)   │
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
┌──────────────────────────────────────────────┐
│              RAG Layer  (v1.1)                │
│                                               │
│  User query                                   │
│      │                                        │
│      ▼                                        │
│  Embed (sentence-transformers, local/offline) │
│      │                                        │
│      ▼                                        │
│  SQLite cosine similarity search              │
│      │                                        │
│      ├── Top-K review chunks                  │
│      └── Top-K personal note chunks           │
│      │                                        │
│      ▼                                        │
│  Inject as <retrieved_context> into prompt    │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│          LLM Layer (Anthropic Claude)         │
│  • Explain why destinations match            │
│  • Build grounded multi-day itineraries      │
│  • RAG chat — multi-turn Q&A with context    │
└──────────────────────────────────────────────┘
        │
        ▼
   Ranked Results + Grounded Narrative Itinerary
```

---

## Open APIs Used

| API | Purpose | Free Tier |
|-----|---------|-----------|
| [OpenTripMap](https://opentripmap.io/docs) | POIs, attraction categories, ratings | 1000 req/day |
| [Rest Countries](https://restcountries.com/) | Country metadata (language, currency, region) | Unlimited |
| [Open-Meteo](https://open-meteo.com/) | Climate/weather data per destination | Unlimited |
| [Unsplash](https://unsplash.com/developers) | Destination photos | 50 req/hr |
| [Anthropic Claude API](https://docs.anthropic.com/) | LLM narration + itinerary generation | Pay-per-token |

> All APIs except Anthropic are **free and require no credit card**.  
> The RAG embedding model (`all-MiniLM-L6-v2`) runs **fully offline** — no API key needed.

---

## Setup

```bash
# Clone repo
git clone https://github.com/yourname/TravelDude.git
cd TravelDude

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Note: first run downloads the ~90MB sentence-transformers model automatically

# Set your API keys
cp .env.example .env
# Edit .env and add:
#   OPENTRIPMAP_API_KEY=your_key
#   UNSPLASH_ACCESS_KEY=your_key
#   ANTHROPIC_API_KEY=your_key

# Initialize the database
python src/database_init.py

# Seed the RAG index with built-in demo reviews and notes
python src/rag/ingest.py --seed

# Run
python src/main.py
```

---

## Workflow

1. **User Profile Collection** — User rates travel categories (beaches, culture, food, adventure, etc.) or describes their ideal trip in plain English
2. **Normalization** — Min-Max normalize scores to build a preference vector
3. **Destination Retrieval** — Pull live POI data from OpenTripMap for candidate destinations
4. **Content-Based Filtering** — TF-IDF on destination tags + cosine similarity against preference vector
5. **RAG Retrieval** — Embed the user query locally, search `rag_documents` for relevant review and note chunks
6. **LLM Augmentation** — Claude explains matches and generates a grounded multi-day itinerary using retrieved context
7. **RAG Chat** — Multi-turn chatbot grounded in your indexed reviews and personal notes
8. **Output** — Ranked list with narrative itinerary + optional Unsplash photos

---

## Database Structure

```
TravelDude.db
│
├── TABLE: Destinations
│   ├── dest_id: INTEGER (PRIMARY KEY)
│   ├── name: TEXT
│   ├── country: TEXT
│   ├── region: TEXT
│   ├── categories: TEXT  (comma-separated: beach, culture, food, ...)
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
│   ├── user_id: VARCHAR(20)
│   ├── dest_id: INTEGER
│   ├── user_rating: REAL
│   ├── visited: BOOLEAN
│   └── wishlist: BOOLEAN
│
├── TABLE: LLM_Logs
│   ├── log_id: INTEGER (PRIMARY KEY)
│   ├── user_id: VARCHAR(20)
│   ├── prompt_hash: TEXT        ← SHA-256 hash only, never raw prompt
│   ├── response_summary: TEXT   ← ≤200 chars
│   ├── model: TEXT
│   ├── tokens_used: INTEGER
│   └── created_at: TIMESTAMP    ← purged after 90 days
│
├── TABLE: rag_documents          ← RAG layer
│   ├── doc_id: INTEGER (PRIMARY KEY)
│   ├── doc_hash: TEXT UNIQUE
│   ├── source_type: TEXT         (review | note)
│   ├── destination: TEXT
│   ├── title: TEXT
│   ├── content: TEXT             ← chunked text (~400 words, 80-word overlap)
│   ├── chunk_index: INTEGER
│   └── metadata: TEXT            ← JSON (rating, reviewer, filename)
│
└── TABLE: rag_embeddings          ← RAG layer
    ├── doc_id: INTEGER (PRIMARY KEY)
    └── embedding: BLOB            ← 384-dim float32, all-MiniLM-L6-v2
```

---

## RAG Pipeline

The RAG layer grounds every LLM response in real traveler reviews and your own notes, rather than relying solely on Claude's training data.

### Adding your own knowledge

```bash
# Ingest personal travel notes (.md or .txt)
# Drop files into data/notes/ — name them by destination for best results
# e.g. tokyo_tips.md, bali_hidden_gems.txt
python src/rag/ingest.py --notes data/notes/

# Ingest travel review CSVs (TripAdvisor-style, Kaggle datasets, etc.)
python src/rag/ingest.py --reviews data/reviews/

# Ingest everything at once
python src/rag/ingest.py --all

# Check what's indexed
python src/rag/ingest.py --stats
```

### Notes file format (optional front-matter)

```markdown
---
destination: Tokyo
title: My Tokyo Tips
---
Your notes here...
```

### Compatible review CSV columns

| Canonical field | Accepted column names |
|---|---|
| Review text | `review`, `review_text`, `text`, `content`, `body` |
| Destination | `destination`, `city`, `location`, `place` |
| Title | `title`, `review_title`, `subject` |
| Rating | `rating`, `stars`, `score` |

---

## Privacy & Data Flow

TBD

---

## Project Structure

```
TravelDude/
├── data/
│   ├── notes/              ← drop your .md / .txt travel notes here
│   └── reviews/            ← drop review CSVs here
├── database/               ← TravelDude.db lives here (git-ignored)
├── docs/
│   └── privacy_audit.md
├── src/
│   ├── main.py             ← CLI entry point
│   ├── engine.py           ← TF-IDF content-based filtering
│   ├── llm_layer.py        ← Anthropic Claude integration
│   ├── api_clients.py      ← OpenTripMap, RestCountries, Open-Meteo, Unsplash
│   ├── database_init.py    ← schema creation + destination seed data
│   └── rag/
│       ├── __init__.py
│       ├── vector_store.py ← SQLite embedding store + cosine retrieval
│       ├── ingest.py       ← CSV review + notes ingestion pipeline
│       └── rag_llm.py      ← RAG-enriched itinerary + TravelChat class
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## License

MIT — see [LICENSE](LICENSE)
