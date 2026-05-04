"""
rag/rag_llm.py

RAG-augmented LLM calls for TravelDude.
Wraps the base llm_layer.py functions with retrieval context
injected into every prompt before sending to Claude.

Two main entry points:
  1. rag_generate_itinerary()  — enriched itinerary with grounded reviews + notes
  2. rag_chat()                — multi-turn chatbot with persistent context window
"""

import os
import sys
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic
from dotenv import load_dotenv
from rag.vector_store import retrieve

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

# ── Context builder ───────────────────────────────────────────────────────────

def _build_rag_context(
    query: str,
    destination: Optional[str] = None,
    top_k: int = 6,
    source_types: Optional[List[str]] = None,
) -> str:
    """
    Retrieve relevant chunks and format them as a context block
    to inject into a Claude prompt.
    """
    all_chunks = []

    # Pull from reviews
    if source_types is None or "review" in source_types:
        reviews = retrieve(
            query=query,
            top_k=top_k // 2 + 1,
            destination_filter=destination,
            source_type_filter="review",
        )
        all_chunks.extend(reviews)

    # Pull from personal notes
    if source_types is None or "note" in source_types:
        notes = retrieve(
            query=query,
            top_k=top_k // 2 + 1,
            destination_filter=destination,
            source_type_filter="note",
        )
        all_chunks.extend(notes)

    # Re-rank by score, take top_k total
    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    top = all_chunks[:top_k]

    if not top:
        return ""

    lines = ["<retrieved_context>"]
    for i, chunk in enumerate(top, 1):
        src = f"[{chunk['source_type'].upper()}]"
        dest_label = f" ({chunk['destination']})" if chunk["destination"] else ""
        title_label = f" — {chunk['title']}" if chunk["title"] else ""
        score_label = f" [relevance: {chunk['score']:.2f}]"
        lines.append(
            f"\n--- Source {i}: {src}{dest_label}{title_label}{score_label} ---"
        )
        lines.append(chunk["content"])
    lines.append("\n</retrieved_context>")
    return "\n".join(lines)


# ── RAG Itinerary ─────────────────────────────────────────────────────────────

def rag_generate_itinerary(
    destination: str,
    days: int,
    preferences: Dict[str, float],
    pois: List[Dict],
    user_id: str = "anon",
) -> str:
    """
    Generate a grounded itinerary using retrieved reviews and personal notes
    as context alongside the LLM's own knowledge.
    """
    pref_str = ", ".join(f"{k}: {v:.1f}" for k, v in preferences.items() if v > 0.5)
    poi_list = ", ".join(p["name"] for p in pois[:6]) if pois else "local attractions"
    query = f"travel tips things to do food recommendations {destination}"

    context = _build_rag_context(query=query, destination=destination, top_k=6)

    prompt = f"""You are an expert travel planner creating a personalized itinerary.

{"Use the following real traveler reviews and personal notes to enrich your recommendations:" if context else ""}
{context}

Now create a detailed {days}-day itinerary for {destination}.

Traveler preferences: {pref_str}
Notable POIs to include if relevant: {poi_list}

Instructions:
- Format as Day 1, Day 2, etc. with Morning / Afternoon / Evening sections
- Where the retrieved context above contains specific tips, restaurants, or warnings — reference and use them
- Add practical tips (transport, booking, timing) drawn from the reviews
- Be vivid and specific; avoid generic travel brochure language
- If a tip came from a review or note, weave it in naturally (no need to cite the source explicitly)
"""

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        return f"[RAG itinerary generation failed: {e}]"


# ── RAG Chat ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are TravelDude AI, an expert travel assistant.
You have access to real traveler reviews and personal travel notes that will be provided
as context in each message. Use this grounded information to give specific, accurate advice.
When your context contains relevant tips, prioritize them over generic knowledge.
Be concise, warm, and practical."""


class TravelChat:
    """
    Multi-turn RAG chatbot for travel Q&A.
    Maintains conversation history and retrieves fresh context for every turn.

    Usage:
        chat = TravelChat(destination="Tokyo")
        response = chat.ask("What's the best way to get from the airport?")
        response = chat.ask("And what about the food scene?")
        chat.reset()
    """

    def __init__(self, destination: Optional[str] = None):
        self.destination = destination
        self.history: List[Dict] = []   # {role, content} list for Claude

    def ask(self, question: str, top_k: int = 5) -> str:
        """Send a question, retrieves context, returns Claude's answer."""
        # Retrieve context relevant to this turn
        context = _build_rag_context(
            query=question,
            destination=self.destination,
            top_k=top_k,
        )

        # Inject context into user message
        if context:
            user_content = (
                f"{context}\n\n"
                f"Using the above context where relevant, please answer:\n{question}"
            )
        else:
            user_content = question

        self.history.append({"role": "user", "content": user_content})

        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=self.history,
            )
            answer = resp.content[0].text.strip()
            # Store clean assistant turn (without injected context for readability)
            self.history.append({"role": "assistant", "content": answer})
            # Trim history to last 10 turns to stay within context window
            if len(self.history) > 20:
                self.history = self.history[-20:]
            return answer
        except Exception as e:
            err = f"[Chat error: {e}]"
            self.history.append({"role": "assistant", "content": err})
            return err

    def set_destination(self, destination: str) -> None:
        """Focus retrieval on a specific destination mid-chat."""
        self.destination = destination

    def reset(self) -> None:
        """Clear conversation history."""
        self.history = []

    def context_summary(self) -> str:
        """Return a short summary of what the RAG index knows about this destination."""
        if not self.destination:
            return "No destination set."
        chunks = retrieve(self.destination, top_k=3, destination_filter=self.destination)
        if not chunks:
            return f"No indexed content found for '{self.destination}'."
        preview = chunks[0]["content"][:200]
        return (
            f"Found {len(chunks)}+ chunks for '{self.destination}'. "
            f"Sample: \"{preview}...\""
        )
