"""
llm_layer.py
LLM augmentation layer using the Anthropic Claude API.
Handles:
  - Preference extraction from natural language
  - Recommendation narrative (why this destination?)
  - Multi-day itinerary generation
  - Follow-up travel Q&A
"""

import os
import hashlib
import sqlite3
from typing import List, Dict, Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "TravelDude.db")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))


# ── Privacy: hash prompts before logging ──────────────────────────────────────

def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _log_llm_call(user_id: str, prompt: str, summary: str, tokens: int) -> None:
    """
    Log LLM interaction metadata to SQLite.
    NOTE: We store ONLY a hash of the prompt (never raw PII) and a
    brief summary of the response. Raw prompts are NOT persisted.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO LLM_Logs (user_id, prompt_hash, response_summary, model, tokens_used)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, _hash_prompt(prompt), summary[:200], MODEL, tokens),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[LLM Log] Could not write log: {e}")


# ── Core LLM calls ────────────────────────────────────────────────────────────

def extract_preferences_from_text(user_input: str) -> Dict[str, float]:
    """
    Use Claude to extract travel category scores from free-text input.
    Returns dict like {"beach": 0.8, "adventure": 0.9, ...}
    """
    prompt = f"""
You are a travel preference extractor. Given this user message, return a JSON object
with scores (0.0–1.0) for these travel categories:
beach, culture, food, adventure, nature, urban, luxury, budget, history, spirituality,
nightlife, romance, family, wildlife, skiing

User message: "{user_input}"

Respond ONLY with valid JSON. Example: {{"beach": 0.9, "culture": 0.4, "food": 0.7}}
"""
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        text = resp.content[0].text.strip()
        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[LLM] Preference extraction failed: {e}")
        return {}


def generate_recommendation_narrative(
    user_preferences: Dict[str, float],
    destinations: List[Dict],
    user_id: str = "anon",
) -> str:
    """
    Generate a natural-language explanation of why each destination matches the user.
    """
    dest_summaries = "\n".join(
        f"- {d['name']}, {d['country']} (tags: {d.get('tags','')}, rating: {d.get('rating','')})"
        for d in destinations
    )
    pref_str = ", ".join(f"{k}: {v:.1f}" for k, v in user_preferences.items())
    prompt = f"""
You are a friendly expert travel advisor. A user has these travel preferences:
{pref_str}

Based on content-based filtering, these top destinations were recommended:
{dest_summaries}

Write a warm, engaging 2–3 paragraph explanation of why these destinations match the user's preferences.
Be specific about each destination's highlights. Do not add new destinations.
"""
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        narrative = resp.content[0].text.strip()
        _log_llm_call(user_id, prompt, narrative[:100], resp.usage.output_tokens)
        return narrative
    except Exception as e:
        return f"[LLM unavailable: {e}]"


def generate_itinerary(
    destination: str,
    days: int,
    preferences: Dict[str, float],
    pois: List[Dict],
    user_id: str = "anon",
) -> str:
    """
    Generate a day-by-day itinerary for a chosen destination.
    """
    poi_list = ", ".join(p["name"] for p in pois[:6]) if pois else "local attractions"
    pref_str = ", ".join(f"{k}: {v:.1f}" for k, v in preferences.items() if v > 0.5)

    prompt = f"""
Create a detailed {days}-day travel itinerary for {destination}.
The traveler enjoys: {pref_str}.
Notable nearby places to include if relevant: {poi_list}.

Format as Day 1, Day 2, etc. with morning/afternoon/evening suggestions.
Include food recommendations and practical tips. Be specific and vivid.
"""
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        itinerary = resp.content[0].text.strip()
        _log_llm_call(
            user_id, prompt, f"Itinerary: {destination} {days}d", resp.usage.output_tokens
        )
        return itinerary
    except Exception as e:
        return f"[LLM unavailable: {e}]"


def answer_travel_question(
    question: str,
    context: Optional[str] = None,
    user_id: str = "anon",
) -> str:
    """
    Answer a follow-up travel question with optional destination context.
    """
    system = (
        "You are TravelDude AI, a helpful and knowledgeable travel assistant. "
        "Give concise, accurate, and practical travel advice."
    )
    messages = []
    if context:
        messages.append({"role": "user", "content": f"Context: {context}"})
        messages.append({"role": "assistant", "content": "Got it, I'll keep that in mind."})
    messages.append({"role": "user", "content": question})

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=system,
            messages=messages,
        )
        answer = resp.content[0].text.strip()
        _log_llm_call(user_id, question, answer[:100], resp.usage.output_tokens)
        return answer
    except Exception as e:
        return f"[LLM unavailable: {e}]"
