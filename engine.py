"""
engine.py
Content-based filtering engine for TravelDude.
Mirrors MovieDude's TF-IDF + cosine similarity approach, adapted for travel.
"""

import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict


def load_destinations(db_path: str) -> List[Dict]:
    """Load all destination records from SQLite."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM Destinations")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def build_user_preference_vector(
    db_path: str, user_id: str
) -> Tuple[np.ndarray, TfidfVectorizer, List[Dict]]:
    """
    Build a user preference vector from their ratings in User_Preferences.
    Returns (user_vector, fitted_vectorizer, all_destinations).
    """
    destinations = load_destinations(db_path)
    if not destinations:
        raise ValueError("No destinations in database. Run database_init.py first.")

    # Build TF-IDF matrix over all destination tag strings
    tag_corpus = [d.get("tags", "") for d in destinations]
    vectorizer = TfidfVectorizer(token_pattern=r"[a-zA-Z][a-zA-Z]+")
    tfidf_matrix = vectorizer.fit_transform(tag_corpus)

    # Fetch user ratings
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT dest_id, user_rating FROM User_Preferences WHERE user_id = ?",
        (user_id,),
    )
    ratings = {row["dest_id"]: row["user_rating"] for row in cur.fetchall()}
    conn.close()

    if not ratings:
        # Cold-start: return None to trigger preference questionnaire
        return None, vectorizer, destinations

    # Weighted average of TF-IDF vectors for rated destinations
    dest_id_to_idx = {d["dest_id"]: i for i, d in enumerate(destinations)}
    weighted_sum = np.zeros(tfidf_matrix.shape[1])
    total_weight = 0.0

    for dest_id, rating in ratings.items():
        idx = dest_id_to_idx.get(dest_id)
        if idx is not None:
            weight = rating / 5.0  # normalize to [0,1]
            weighted_sum += weight * tfidf_matrix[idx].toarray().flatten()
            total_weight += weight

    if total_weight == 0:
        return None, vectorizer, destinations

    user_vector = weighted_sum / total_weight
    return user_vector, vectorizer, destinations


def recommend_from_categories(
    category_scores: Dict[str, float],
    db_path: str,
    top_n: int = 5,
    exclude_visited: bool = True,
    user_id: str = None,
    min_rating: float = 4.0,
) -> List[Dict]:
    """
    Cold-start recommendation from explicit category preferences.
    category_scores: dict like {"beach": 0.9, "adventure": 0.7, "food": 0.8, ...}
    """
    destinations = load_destinations(db_path)

    # Build user pseudo-document from category weights
    pseudo_doc = " ".join(
        cat for cat, score in category_scores.items() for _ in range(int(score * 10))
    )

    tag_corpus = [d.get("tags", "") for d in destinations]
    vectorizer = TfidfVectorizer(token_pattern=r"[a-zA-Z][a-zA-Z]+")
    tfidf_matrix = vectorizer.fit_transform(tag_corpus + [pseudo_doc])

    user_vec = tfidf_matrix[-1]
    dest_matrix = tfidf_matrix[:-1]
    similarities = cosine_similarity(user_vec, dest_matrix).flatten()

    # Attach scores
    scored = [
        {**dest, "similarity": float(similarities[i])}
        for i, dest in enumerate(destinations)
    ]

    # Filter visited if user_id provided
    if exclude_visited and user_id:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT dest_id FROM User_Preferences WHERE user_id=? AND visited=1",
            (user_id,),
        )
        visited_ids = {row[0] for row in cur.fetchall()}
        conn.close()
        scored = [d for d in scored if d["dest_id"] not in visited_ids]

    # Filter by minimum rating
    scored = [d for d in scored if d.get("rating", 0) >= min_rating]

    # Sort by similarity score descending
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_n]
