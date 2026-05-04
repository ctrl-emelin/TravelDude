"""
llm_layer.py
LLM layer using Ollama (local, free, no API key required).
Default model: llama3

Handles:
  - Preference extraction from natural language
  - Recommendation narrative (why this destination?)
  - Multi-day itinerary generation
  - Follow-up travel Q&A

Ollama must be running locally: https://ollama.com
  brew install ollama        # Mac
  ollama pull llama3         # download the model
  ollama serve               # start the server (runs on localhost:11434)
"""

import os
import json
import hashlib
import sqlite3
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
DB_PATH      = os.path.join(os.path.dirname(__file__), "..", "database", "TravelDude.db")


# ── Ollama client ─────────────────────────────────────────────────────────────

def _ollama_chat(
    messages: List[Dict],
    system: Optional[str] = None,
    temperature: float = 0.7,
) -> str:
    """
    Send a chat request to Ollama and return the response text.
    Uses the /api/chat endpoint (supports multi-turn message history).
    """
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": temperature},
    }
    if system:
        payload["messages"] = [{"role": "system", "content": system}] + messages

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return (
            "[Ollama not running] Start it with: ollama serve\n"
            "Then make sure llama3 is pulled: ollama pull llama3"
        )
    except Exception as e:
        return f"[Ollama error: {e}]"


def _ollama_available() -> bool:
    """Quick health-check — returns True if Ollama is reachable."""
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return True
    except Exception:
        return False


# ── Privacy: hash prompts before logging ──────────────────────────────────────

def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _log_llm_call(user_id: str, prompt: str, summary: str, tokens: int = 0) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT INTO LLM_Logs
               (user_id, prompt_hash, response_summary, model, tokens_used)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, _hash_prompt(prompt), summary[:200], OLLAMA_MODEL, tokens),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[LLM Log] Could not write log: {e}")


# ── Core LLM calls ────────────────────────────────────────────────────────────

def extract_preferences_from_text(user_input: str) -> Dict[str, float]:
    prompt = f"""You are a travel preference extractor. Given this user message, return a JSON object
with scores (0.0 to 1.0) for these travel categories:
beach, culture, food, adventure, nature, urban, luxury, budget, history, spirituality,
nightlife, romance, family, wildlife, skiing

User message: "{user_input}"

Rules:
- Respond ONLY with a valid JSON object, nothing else
- No markdown, no explanation, no code fences
- Example: {{"beach": 0.9, "culture": 0.4, "food": 0.7}}"""

    text = _ollama_chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        return json.loads(text)
    except Exception as e:
        print(f"[LLM] Preference extraction failed: {e}\nRaw: {text}")
        return {}


def generate_recommendation_narrative(
    user_preferences: Dict[str, float],
    destinations: List[Dict],
    user_id: str = "anon",
) -> str:
    dest_summaries = "\n".join(
        f"- {d['name']}, {d['country']} (tags: {d.get('tags','')}, rating: {d.get('rating','')})"
        for d in destinations
    )
    pref_str = ", ".join(f"{k}: {v:.1f}" for k, v in user_preferences.items())

    prompt = f"""A user has these travel preferences: {pref_str}

These top destinations were recommended:
{dest_summaries}

Write a warm, engaging 2-3 paragraph explanation of why these destinations match the user's preferences.
Be specific about each destination's highlights. Do not add new destinations."""

    narrative = _ollama_chat(
        messages=[{"role": "user", "content": prompt}],
        system="You are TravelDude AI, an expert and enthusiastic travel advisor.",
    )
    _log_llm_call(user_id, prompt, narrative[:100])
    return narrative


def generate_itinerary(
    destination: str,
    days: int,
    preferences: Dict[str, float],
    pois: List[Dict],
    user_id: str = "anon",
) -> str:
    poi_list = ", ".join(p["name"] for p in pois[:6]) if pois else "local attractions"
    pref_str = ", ".join(f"{k}: {v:.1f}" for k, v in preferences.items() if v > 0.5)

    prompt = f"""Create a detailed {days}-day travel itinerary for {destination}.
The traveler enjoys: {pref_str}.
Notable nearby places: {poi_list}.

Format as Day 1, Day 2, etc. with Morning / Afternoon / Evening sections.
Include food recommendations and practical tips. Be specific and vivid."""

    itinerary = _ollama_chat(
        messages=[{"role": "user", "content": prompt}],
        system="You are TravelDude AI, an expert travel planner with deep local knowledge.",
    )
    _log_llm_call(user_id, prompt, f"Itinerary: {destination} {days}d")
    return itinerary


def answer_travel_question(
    question: str,
    context: Optional[str] = None,
    user_id: str = "anon",
) -> str:
    messages = []
    if context:
        messages.append({"role": "user",      "content": f"Context: {context}"})
        messages.append({"role": "assistant", "content": "Got it, I'll keep that in mind."})
    messages.append({"role": "user", "content": question})

    answer = _ollama_chat(
        messages=messages,
        system=(
            "You are TravelDude AI, a helpful and knowledgeable travel assistant. "
            "Give concise, accurate, and practical travel advice."
        ),
    )
    _log_llm_call(user_id, question, answer[:100])
    return answer
