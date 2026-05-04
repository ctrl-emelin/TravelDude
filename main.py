"""
main.py
TravelDude — CLI entry point.
Run: python src/main.py
"""

import os
import sys
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
    generate_itinerary,
    answer_travel_question,
)
from api_clients import get_country_info, get_pois_by_city

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
    console.print(Panel.fit(
        "[bold green]TravelDude 🌍[/bold green]\n"
        "[dim]AI-powered travel recommendations with LLM itinerary generation[/dim]",
        border_style="green",
    ))

    # User ID (simple prototype)
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

    # Run engine
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

    # Itinerary
    dest_names = [r["name"] for r in recommendations]
    choice = Prompt.ask(
        f"\nGenerate itinerary for which destination?",
        choices=dest_names + ["skip"],
        default=dest_names[0],
    )
    if choice != "skip":
        days = int(Prompt.ask("How many days?", default="5"))
        chosen = next(d for d in recommendations if d["name"] == choice)

        # Fetch live POIs
        console.print(f"[dim]Fetching live POIs for {choice}...[/dim]")
        pois = get_pois_by_city(choice)

        # Country info
        country_info = get_country_info(chosen["country"])
        console.print(
            f"[cyan]{country_info.get('flag','')} {chosen['country']}[/cyan] — "
            f"Capital: {country_info.get('capital','?')} | "
            f"Currency: {', '.join(country_info.get('currencies',['?']))} | "
            f"Language: {', '.join(country_info.get('languages',['?'])[:2])}"
        )

        console.print(f"\n[dim]Generating {days}-day itinerary...[/dim]")
        itinerary = generate_itinerary(choice, days, preferences, pois, user_id)
        console.print(Panel(itinerary, title=f"📅 {days}-Day Itinerary: {choice}", border_style="yellow"))

    # Follow-up Q&A
    while Prompt.ask("\nAsk a follow-up travel question?", choices=["y", "n"], default="n") == "y":
        question = Prompt.ask("Your question")
        answer = answer_travel_question(question, context=f"User is planning to visit {choice}", user_id=user_id)
        console.print(Panel(answer, title="🤖 TravelDude AI", border_style="green"))

    console.print("\n[bold green]Happy travels! 🌏[/bold green]")


if __name__ == "__main__":
    main()
