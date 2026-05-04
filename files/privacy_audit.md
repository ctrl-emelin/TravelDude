# AI Data Flows Privacy Audit: TravelDude

| | |
|---|---|
| **System** | TravelDude v1.2.0 (Ollama + RAG) |
| **Audit Date** | 2026-05-03 |
| **Prepared by** | TravelDude Engineering |
| **Scope** | Recommendation engine · Ollama LLM layer · RAG pipeline · Open API integrations |

---

## 1. System Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           USER LAYER  (CLI / Jupyter)                         │
│                                                                                │
│   User input: preferences, free-text queries, destination choices             │
│   No PII required — username is user-chosen, no email/phone needed            │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │  Local process (no network for LLM)
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER  (Python)                           │
│                                                                                │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────────────┐  │
│  │  Preference     │   │  Content-Based   │   │  LLM Layer               │  │
│  │  Collector      │──▶│  Filtering       │──▶│  (Ollama — llama3)       │  │
│  │  free-text /    │   │  Engine          │   │                          │  │
│  │  manual sliders │   │  TF-IDF + cosine │   │  • Preference extraction │  │
│  └─────────────────┘   └────────┬─────────┘   │  • Narratives            │  │
│                                  │             │  • Itinerary generation  │  │
│  ┌─────────────────┐             │             │  • Q&A chat              │  │
│  │  Open API       │             │             └────────────┬─────────────┘  │
│  │  Clients        │             │                          │ localhost only  │
│  │                 │             │             ┌────────────▼─────────────┐  │
│  │  OpenTripMap    │             │             │  Ollama Server           │  │
│  │  RestCountries  │             │             │  localhost:11434         │  │
│  │  Open-Meteo     │             │             │  ✅ No data leaves       │  │
│  │  Unsplash       │             │             │     your machine         │  │
│  └─────────────────┘             │             └──────────────────────────┘  │
│                                  │                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  RAG Layer                                                            │    │
│  │                                                                        │    │
│  │  User query ──▶ Embed (sentence-transformers, local) ──▶ SQLite       │    │
│  │                 cosine similarity search                               │    │
│  │                       │                                                │    │
│  │          ┌────────────┴──────────────┐                                │    │
│  │          ▼                           ▼                                 │    │
│  │   Top-K review chunks       Top-K note chunks                         │    │
│  │          └────────────┬──────────────┘                                │    │
│  │                        ▼                                               │    │
│  │              Inject as <retrieved_context>                             │    │
│  │              into Ollama prompt  ──▶  LLM Layer                       │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    DATA LAYER  (SQLite — single file: TravelDude.db)          │
│                                                                                │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────────────────────┐    │
│  │ Destinations │  │ Users         │  │ LLM_Logs                       │    │
│  │ (static)     │  │ user_id, name │  │ prompt_HASH · summary · model  │    │
│  └──────────────┘  └───────────────┘  │ 90-day TTL — no raw prompts    │    │
│                                        └────────────────────────────────┘    │
│  ┌──────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │ User_Preferences             │  │ rag_documents + rag_embeddings      │  │
│  │ ratings · visited · wishlist │  │ chunked text · 384-dim vectors      │  │
│  │ user-deletable on request    │  │ source: reviews / personal notes    │  │
│  └──────────────────────────────┘  └─────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow Analysis

| # | Data Flow | Source | Destination | Encrypted? | Logged? | Priority |
|---|-----------|--------|-------------|------------|---------|---------|
| 1 | User preference input | CLI / Jupyter | App (local) | Local only | No | **High** |
| 2 | Free-text query → Ollama prompt | App | Ollama (localhost) | Local only | Prompt hash only | **High** |
| 3 | Ollama response | localhost:11434 | App | Local only | Summary ≤200 chars | Medium |
| 4 | LLM log metadata | App | SQLite | At-rest (AES-256 rec.) | Yes — 90-day TTL | Medium |
| 5 | User ratings & wishlist | CLI / Jupyter | SQLite | At-rest (AES-256 rec.) | Yes | **High** |
| 6 | RAG query embedding | App | SQLite (local) | Local only | No | Low |
| 7 | Review / note chunks ingested | Local files | SQLite | At-rest (AES-256 rec.) | On ingest | Medium |
| 8 | Retrieved context → Ollama prompt | SQLite | localhost:11434 | Local only | Prompt hash only | Medium |
| 9 | OpenTripMap API call | App | OpenTripMap (HTTPS) | In-transit (TLS) | No | Low |
| 10 | RestCountries / Open-Meteo calls | App | Public APIs (HTTPS) | In-transit (TLS) | No | Low |
| 11 | Unsplash photo fetch | App | Unsplash (HTTPS) | In-transit (TLS) | No | Low |
| 12 | Final recommendations | App | User (display) | Local only | No | Low |

> **Key difference from cloud LLM architectures:** flows 1–3 and 8 are all localhost — no user data is transmitted to any external LLM provider.

---

## 3. Privacy Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Personal notes in RAG index contain sensitive information | Medium | Medium | Notes stay in local SQLite; only sent to localhost Ollama — never to an external server |
| SQLite database unencrypted at rest | High | Medium | Use SQLCipher (`pip install sqlcipher3`) or PostgreSQL with disk encryption in production |
| API keys for OpenTripMap / Unsplash exposed in `.env` | Medium | Low | Keys only control optional enrichment features; `.env` is in `.gitignore`; use a secrets manager in production |
| RAG review CSV contains user PII (emails, names) | Low | Medium | Strip PII columns before ingestion; `ingest.py` stores only content + destination metadata |
| LLM_Logs table grows unbounded | Low | Low | 90-day automated TTL purge; only prompt hash + summary stored, never raw text |
| Ollama server exposed on network interface | Low | Medium | Default binds to localhost only; do not expose port 11434 publicly |

---

## 4. RAG-Specific Privacy Notes

**Embeddings (`rag_embeddings`)** are generated locally by `sentence-transformers` (all-MiniLM-L6-v2). No data leaves the machine during indexing. Embeddings are binary blobs — not directly reversible to source text, though source chunks are stored alongside in `rag_documents`.

**Retrieved context → Ollama** — relevant chunks are injected into prompts sent to the local Ollama server at `localhost:11434`. This is a local process; no data is transmitted externally. Ensure that content ingested into the RAG index does not contain sensitive personal details you would not want processed by the model.

**Ingested sources:**
- `data/reviews/*.csv` — strip PII columns (email, full name) before ingesting
- `data/notes/*.md` or `*.txt` — review before indexing; these go into the local SQLite DB

---

## 5. Data Minimization Principles Applied

- **Fully local LLM** — Ollama runs on `localhost`; no prompts transmitted externally under any circumstances
- **Local embeddings** — `sentence-transformers` runs offline; RAG indexing never requires a network call
- **No raw prompts stored** — only a 16-char SHA-256 hash is persisted in `LLM_Logs`
- **No device location** — coordinates derived from city names chosen by the user, not GPS
- **No account required** — `user_id` is a user-chosen string, not email or phone
- **User data deletion** — all five tables support deletion by `user_id`; `rag_documents` supports deletion by source file

---

## 6. Recommended Production Hardening

1. **Encrypt SQLite at rest** — use SQLCipher or migrate to PostgreSQL with `pgcrypto`
2. **Secrets management** — replace `.env` with AWS Secrets Manager or HashiCorp Vault for the optional travel API keys
3. **Bind Ollama to localhost only** — ensure `OLLAMA_HOST=http://localhost:11434` and port 11434 is not exposed in firewall rules
4. **User authentication** — hash passwords with `bcrypt` or `argon2` if adding multi-user support
5. **GDPR deletion endpoint** — implement `DELETE /users/{user_id}/data` across all five tables
6. **90-day log purge** — scheduled job: `DELETE FROM LLM_Logs WHERE created_at < datetime('now', '-90 days')`
7. **RAG content review** — audit `data/notes/` and `data/reviews/` before ingestion; strip columns containing email, phone, or full name from CSVs
