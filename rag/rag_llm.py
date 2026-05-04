"""
rag/rag_llm.py

RAG-augmented LLM calls for TravelDude using Ollama (llama3).
Retrieves relevant chunks from SQLite and injects them into
every prompt before sending to the local Ollama server.

Two main entry points:
  1. rag_generate_itinerary()  — enriched itinerary grounded in reviews + notes
  2. TravelChat                — multi-turn chatbot with persistent context
"""

import os
import sys
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from rag.vector_store import retrieve
from llm_layer import _ollama_chat, _log_llm_call

load_dotenv()

SYSTEM_PROMPT = (
    "You are TravelDude AI, an expert travel assistant. "
    "You have access to real traveler reviews and personal travel notes "
    "provided as context. Use this grounded information to give specific, "
    "accurate advice. When context contains relevant tips, prioritize them. "
    "Be concise, warm, and practical."
)


# ── Context builder ───────────────────────────────────────────────────────────

def _build_rag_context(
    query: str,
    destination: Optional[str] = None,
    top_k: int = 6,
    source_types: Optional[List[str]] = None,
) -> str:
    """Retrieve top-K chunks and format as a context block for injection."""
    all_chunks = []

    if source_types is None or "review" in source_types:
        all_chunks.extend(retrieve(
            query=query, top_k=top_k // 2 + 1,
            destination_filter=destination, source_type_filter="review",
        ))

    if source_types is None or "note" in source_types:
        all_chunks.extend(retrieve(
            query=query, top_k=top_k // 2 + 1,
            destination_filter=destination, source_type_filter="note",
        ))

    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    top = all_chunks[:top_k]

    if not top:
        return ""

    lines = ["<retrieved_context>"]
    for i, chunk in enumerate(top, 1):
        src        = f"[{chunk['source_type'].upper()}]"
        dest_label = f" ({chunk['destination']})" if chunk["destination"] else ""
        title_label= f" — {chunk['title']}"       if chunk["title"]       else ""
        score_label= f" [relevance: {chunk['score']:.2f}]"
        lines.append(f"\n--- Source {i}: {src}{dest_label}{title_label}{score_label} ---")
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
    """Generate a grounded itinerary using retrieved reviews and notes."""
    pref_str = ", ".join(f"{k}: {v:.1f}" for k, v in preferences.items() if v > 0.5)
    poi_list = ", ".join(p["name"] for p in pois[:6]) if pois else "local attractions"
    query    = f"travel tips things to do food recommendations {destination}"

    context = _build_rag_context(query=query, destination=destination, top_k=6)

    prompt = f"""{"Use the following real traveler reviews and personal notes to enrich your recommendations:" if context else ""}
{context}

Create a detailed {days}-day itinerary for {destination}.
Traveler preferences: {pref_str}
Notable POIs: {poi_list}

Instructions:
- Format as Day 1, Day 2, etc. with Morning / Afternoon / Evening sections
- Where retrieved context contains specific tips or warnings, use them
- Add practical tips on transport, booking, and timing
- Be vivid and specific; avoid generic travel brochure language"""

    itinerary = _ollama_chat(
        messages=[{"role": "user", "content": prompt}],
        system=SYSTEM_PROMPT,
    )
    _log_llm_call(user_id, prompt, f"RAG Itinerary: {destination} {days}d")
    return itinerary


# ── RAG Chat ──────────────────────────────────────────────────────────────────

class TravelChat:
    """
    Multi-turn RAG chatbot for travel Q&A powered by Ollama.
    Retrieves fresh context for every turn and maintains conversation history.

    Usage:
        chat = TravelChat(destination="Tokyo")
        print(chat.ask("Best area to stay?"))
        print(chat.ask("What about food?"))
        chat.reset()

    CLI commands during chat:
        /dest <name>   — switch destination focus
        /reset         — clear conversation history
        /quit          — exit chat
    """

    def __init__(self, destination: Optional[str] = None):
        self.destination = destination
        self.history: List[Dict] = []

    def ask(self, question: str, top_k: int = 5) -> str:
        """Send a question, retrieve context, return llama3's answer."""
        context = _build_rag_context(
            query=question,
            destination=self.destination,
            top_k=top_k,
        )

        if context:
            user_content = (
                f"{context}\n\n"
                f"Using the above context where relevant, answer:\n{question}"
            )
        else:
            user_content = question

        self.history.append({"role": "user", "content": user_content})

        answer = _ollama_chat(
            messages=self.history,
            system=SYSTEM_PROMPT,
        )

        # Store clean assistant turn for next context window
        self.history.append({"role": "assistant", "content": answer})

        # Keep last 10 turns to avoid bloating the context window
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return answer

    def set_destination(self, destination: str) -> None:
        self.destination = destination

    def reset(self) -> None:
        self.history = []

    def context_summary(self) -> str:
        if not self.destination:
            return "No destination set."
        chunks = retrieve(self.destination, top_k=3, destination_filter=self.destination)
        if not chunks:
            return f"No indexed content found for '{self.destination}'."
        preview = chunks[0]["content"][:200]
        return f"Found {len(chunks)}+ chunks for '{self.destination}'. Sample: \"{preview}...\""
