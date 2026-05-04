"""
database_init.py
Initializes TravelDude.db with schema and seeds it with a starter
dataset pulled from the RestCountries + OpenTripMap free APIs.
"""

import sqlite3
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "TravelDude.db")
OTM_KEY = os.getenv("OPENTRIPMAP_API_KEY", "")

# ── Starter destinations (seeded offline for zero-API demo) ──────────────────
SEED_DESTINATIONS = [
    ("Paris",        "France",      "Europe",      "culture,food,art,history,romance",     "temperate",  3, 0.82, 4.7),
    ("Bali",         "Indonesia",   "Asia",        "beach,nature,spirituality,food,yoga",   "tropical",   2, 0.74, 4.8),
    ("Tokyo",        "Japan",       "Asia",        "food,technology,culture,urban,anime",   "temperate",  3, 0.95, 4.9),
    ("Patagonia",    "Argentina",   "S. America",  "adventure,nature,hiking,wildlife",      "cold",       2, 0.79, 4.6),
    ("Marrakech",    "Morocco",     "Africa",      "culture,food,markets,history,desert",   "hot",        1, 0.71, 4.5),
    ("Kyoto",        "Japan",       "Asia",        "culture,history,temples,nature,art",    "temperate",  3, 0.94, 4.8),
    ("Santorini",    "Greece",      "Europe",      "beach,romance,food,architecture,wine",  "mediterranean", 4, 0.88, 4.7),
    ("Cape Town",    "South Africa","Africa",      "nature,adventure,food,culture,beach",   "mediterranean", 2, 0.68, 4.6),
    ("New York",     "USA",         "N. America",  "urban,food,art,culture,nightlife",      "temperate",  5, 0.80, 4.7),
    ("Queenstown",   "New Zealand", "Oceania",     "adventure,nature,skiing,bungee,hiking", "temperate",  4, 0.92, 4.8),
    ("Barcelona",    "Spain",       "Europe",      "architecture,food,beach,art,nightlife", "mediterranean", 3, 0.85, 4.7),
    ("Chiang Mai",   "Thailand",    "Asia",        "culture,food,temples,nature,budget",    "tropical",   1, 0.78, 4.6),
    ("Iceland",      "Iceland",     "Europe",      "nature,adventure,aurora,hiking,unique", "cold",       4, 0.93, 4.8),
    ("Havana",       "Cuba",        "Caribbean",   "culture,music,history,food,vintage",    "tropical",   1, 0.72, 4.4),
    ("Maldives",     "Maldives",    "Asia",        "beach,luxury,diving,romance,nature",    "tropical",   5, 0.90, 4.9),
]


def create_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS Destinations (
            dest_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL,
            country         TEXT NOT NULL,
            region          TEXT,
            categories      TEXT,
            climate         TEXT,
            avg_cost_level  INTEGER CHECK(avg_cost_level BETWEEN 1 AND 5),
            safety_index    REAL,
            rating          REAL,
            tags            TEXT
        );

        CREATE TABLE IF NOT EXISTS Users (
            user_id     VARCHAR(20) PRIMARY KEY,
            name        VARCHAR(40),
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS User_Preferences (
            pref_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     VARCHAR(20) REFERENCES Users(user_id),
            dest_id     INTEGER REFERENCES Destinations(dest_id),
            user_rating REAL,
            visited     BOOLEAN DEFAULT FALSE,
            wishlist    BOOLEAN DEFAULT FALSE,
            rated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS LLM_Logs (
            log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     VARCHAR(20),
            prompt_hash TEXT,
            response_summary TEXT,
            model       TEXT,
            tokens_used INTEGER,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    print("[DB] Schema created.")


def seed_destinations(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM Destinations")
    if cur.fetchone()[0] > 0:
        print("[DB] Destinations already seeded — skipping.")
        return

    for row in SEED_DESTINATIONS:
        name, country, region, categories, climate, cost, safety, rating = row
        cur.execute("""
            INSERT INTO Destinations (name, country, region, categories, climate,
                                      avg_cost_level, safety_index, rating, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, country, region, categories, climate, cost, safety, rating, categories))

    conn.commit()
    print(f"[DB] Seeded {len(SEED_DESTINATIONS)} destinations.")


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)
    seed_destinations(conn)
    conn.close()
    print(f"[DB] Database ready at: {DB_PATH}")


if __name__ == "__main__":
    main()
