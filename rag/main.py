"""
main.py
TravelDude — CLI entry point.
Run: python src/main.py
"""

import os
import sys
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, FloatPrompt
from rich import box

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from engine import recommend_from_categories
from llm_layer import (
    extract_preferences_from_text,
    generate_recommendation_narrative,
    answer_travel_question,
)
from api_clients import get_country_info, get_pois_by_city

# RAG layer — graceful fallback if sentence-transformers not installed
try:
    from rag import rag_generate_itinerary, TravelChat, seed_demo_data, get_index_stats
    RAG_AVAILABLE = True
except Exception as _rag_err:
    RAG_AVAILABLE = False
    _RAG_ERR = str(_rag_err)
    # Fall back to base itinerary generator
    from llm_layer import generate_itinerary as _base_itinerary

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "TravelDude.db")
console = Console()

CATEGORIES = ["beach", "adventure", "food", "culture", "nature",
              "urban", "history", "romance", "wildlife", "luxury"]


def collect_preferences_manual() -> dict:
    """Rate travel categories manually 1–10."""
    console.print("\n[bold cyan]Rate each travel category (0–10, 0 = not interested):[/bold cyan]")
    scores = {}
    for cat in CATEGORIES:
        val = FloatPrompt.ask(f"  {cat.capitalize()}", default=5.0)
        scores[cat] = max(0.0, min(10.0, val)) / 10.0
    return scores


def collect_preferences_nlp() -> dict:
    """Extract preferences from natural language using Claude."""
    console.print("\n[bold cyan]Describe your ideal trip in a few sentences:[/bold cyan]")
    user_input = Prompt.ask("  >")
    console.print("[dim]Analyzing your preferences with AI...[/dim]")
    prefs = extract_preferences_from_text(user_input)
    if not prefs:
        console.print("[yellow]Could not parse preferences — switching to manual mode.[/yellow]")
        return collect_preferences_manual()
    console.print(f"[green]Detected preferences:[/green] {prefs}")
    return prefs


def display_recommendations(recs: list) -> None:
    table = Table(title="🌍 Top Destinations for You", box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Destination", style="bold cyan")
    table.add_column("Country", style="green")
    table.add_column("Categories")
    table.add_column("Climate", style="yellow")
    table.add_column("Cost (1–5)", justify="center")
    table.add_column("Rating", justify="center", style="magenta")
    table.add_column("Match %", justify="right", style="bold")

    for i, d in enumerate(recs, 1):
        table.add_row(
            str(i),
            d["name"],
            d["country"],
            d.get("categories", "")[:30],
            d.get("climate", ""),
            "💰" * d.get("avg_cost_level", 1),
            f"⭐ {d.get('rating', 0):.1f}",
            f"{d.get('similarity', 0) * 100:.1f}%",
        )
    console.print(table)


def main():
    rag_badge = "[bold green]+ RAG[/bold green]" if RAG_AVAILABLE else "[dim]RAG unavailable[/dim]"
    console.print(Panel.fit(
        f"[bold green]TravelDude 🌍[/bold green]  {rag_badge}\n"
        "[dim]AI-powered travel recommendations with grounded LLM itineraries[/dim]",
        border_style="green",
    ))

    # Seed RAG demo data on first run if index is empty
    if RAG_AVAILABLE:
        stats = get_index_stats()
        if stats["total_chunks"] == 0:
            console.print("[dim]First run — seeding RAG demo data...[/dim]")
            seed_demo_data()
        else:
            console.print(f"[dim]RAG index: {stats['total_chunks']} chunks across {len(stats['by_type'])} source types[/dim]")
    else:
        console.print(f"[yellow]RAG disabled:[/yellow] {_RAG_ERR if 'sentence-transformers' not in str(globals().get('_RAG_ERR','')) else 'Run: pip install sentence-transformers'}")

    # User ID
    user_id = Prompt.ask("\nEnter your username", default="traveler_01")

    # Preference input mode
    mode = Prompt.ask(
        "How would you like to enter preferences?",
        choices=["text", "manual"],
        default="text",
    )

    if mode == "text":
        preferences = collect_preferences_nlp()
    else:
        preferences = collect_preferences_manual()

    # Options
    exclude_visited = Prompt.ask("Exclude already visited destinations?", choices=["y", "n"], default="y") == "y"
    top_n = int(Prompt.ask("How many recommendations?", default="5"))

    # Run recommendation engine
    console.print("\n[dim]Running recommendation engine...[/dim]")
    recommendations = recommend_from_categories(
        category_scores=preferences,
        db_path=DB_PATH,
        top_n=top_n,
        exclude_visited=exclude_visited,
        user_id=user_id,
    )

    if not recommendations:
        console.print("[red]No recommendations found. Try adjusting your preferences.[/red]")
        return

    display_recommendations(recommendations)

    # LLM narrative
    if Prompt.ask("\nGenerate AI explanation of recommendations?", choices=["y", "n"], default="y") == "y":
        console.print("\n[dim]Generating narrative...[/dim]")
        narrative = generate_recommendation_narrative(preferences, recommendations, user_id)
        console.print(Panel(narrative, title="🤖 TravelDude AI Says", border_style="cyan"))

    # ── Itinerary (RAG-enriched if available) ─────────────────────────────────
    dest_names = [r["name"] for r in recommendations]
    choice = Prompt.ask(
        "\nGenerate itinerary for which destination?",
        choices=dest_names + ["skip"],
        default=dest_names[0],
    )

    if choice != "skip":
        days = int(Prompt.ask("How many days?", default="5"))
        chosen = next(d for d in recommendations if d["name"] == choice)

        console.print(f"[dim]Fetching live POIs for {choice}...[/dim]")
        pois = get_pois_by_city(choice)

        country_info = get_country_info(chosen["country"])
        console.print(
            f"[cyan]{country_info.get('flag','')} {chosen['country']}[/cyan] — "
            f"Capital: {country_info.get('capital','?')} | "
            f"Currency: {', '.join(country_info.get('currencies',['?']))} | "
            f"Language: {', '.join(country_info.get('languages',['?'])[:2])}"
        )

        console.print(f"\n[dim]Generating {days}-day itinerary {'(RAG-enriched)' if RAG_AVAILABLE else ''}...[/dim]")

        if RAG_AVAILABLE:
            itinerary = rag_generate_itinerary(choice, days, preferences, pois, user_id)
            title_label = f"📅 {days}-Day Itinerary: {choice}  [dim](grounded with real reviews)[/dim]"
        else:
            itinerary = _base_itinerary(choice, days, preferences, pois, user_id)
            title_label = f"📅 {days}-Day Itinerary: {choice}"

        console.print(Panel(itinerary, title=title_label, border_style="yellow"))

    # ── RAG Chat interface ────────────────────────────────────────────────────
    if RAG_AVAILABLE:
        if Prompt.ask(
            "\nSwitch to RAG chat mode for deeper Q&A about your destination?",
            choices=["y", "n"],
            default="y" if choice != "skip" else "n",
        ) == "y":
            _run_rag_chat(destination=choice if choice != "skip" else None)
    else:
        # Fallback: basic Q&A without RAG
        while Prompt.ask("\nAsk a follow-up travel question?", choices=["y", "n"], default="n") == "y":
            question = Prompt.ask("Your question")
            answer = answer_travel_question(
                question,
                context=f"User is planning to visit {choice}" if choice != "skip" else None,
                user_id=user_id,
            )
            console.print(Panel(answer, title="🤖 TravelDude AI", border_style="green"))

    console.print("\n[bold green]Happy travels! 🌏[/bold green]")


def _run_rag_chat(destination: Optional[str] = None) -> None:
    """Interactive RAG chat session."""
    chat = TravelChat(destination=destination)

    dest_label = f" about [cyan]{destination}[/cyan]" if destination else ""
    console.print(Panel(
        f"[bold]RAG Travel Chat[/bold]{dest_label}\n"
        "[dim]Ask anything — answers are grounded in real reviews and your notes.\n"
        "Commands: [yellow]/dest <name>[/yellow] change destination  "
        "[yellow]/reset[/yellow] clear history  [yellow]/quit[/yellow] exit[/dim]",
        border_style="magenta",
    ))

    # Show what the RAG index knows
    if destination:
        preview = chat.context_summary()
        console.print(f"[dim]RAG context: {preview}[/dim]\n")

    while True:
        try:
            question = Prompt.ask("[magenta]You[/magenta]")
        except (KeyboardInterrupt, EOFError):
            break

        if not question.strip():
            continue

        if question.strip().lower() == "/quit":
            break
        elif question.strip().lower() == "/reset":
            chat.reset()
            console.print("[dim]Conversation history cleared.[/dim]")
            continue
        elif question.strip().lower().startswith("/dest "):
            new_dest = question.strip()[6:].strip()
            chat.set_destination(new_dest)
            console.print(f"[dim]Destination updated to: {new_dest}[/dim]")
            continue

        console.print("[dim]Retrieving context...[/dim]", end="\r")
        answer = chat.ask(question)
        console.print(Panel(answer, title="🤖 TravelDude RAG", border_style="magenta"))


if __name__ == "__main__":
    main()
