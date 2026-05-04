"""
rag/ingest.py

Ingestion pipeline for TravelDude RAG.
Handles two source types:

  1. USER REVIEWS  — CSV format compatible with open TripAdvisor-style datasets
                     (e.g. Kaggle "Hotel Reviews", "Restaurant Reviews", open dumps)
  2. CUSTOM NOTES  — Plain .txt or .md files you drop into data/notes/

Run standalone:
  python src/rag/ingest.py --reviews data/reviews/
  python src/rag/ingest.py --notes   data/notes/
  python src/rag/ingest.py --all
  python src/rag/ingest.py --stats
"""

import os
import csv
import argparse
import sys
from pathlib import Path
from typing import Optional

# Allow running from project root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.vector_store import index_document, get_index_stats
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich import box

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REVIEWS_DIR  = PROJECT_ROOT / "data" / "reviews"
NOTES_DIR    = PROJECT_ROOT / "data" / "notes"


# ── Review ingestion ──────────────────────────────────────────────────────────

# Expected CSV columns (flexible — mapped below)
REVIEW_COLUMN_ALIASES = {
    "destination": ["destination", "city", "location", "place", "hotel_address"],
    "title":       ["title", "review_title", "subject"],
    "content":     ["review", "review_text", "text", "content", "body", "comments"],
    "rating":      ["rating", "stars", "score", "overall"],
    "reviewer":    ["reviewer", "user", "author", "username"],
}


def _map_columns(headers: list) -> dict:
    """Map actual CSV headers to canonical field names."""
    headers_lower = [h.lower().strip() for h in headers]
    mapping = {}
    for canonical, aliases in REVIEW_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in headers_lower:
                mapping[canonical] = headers_lower.index(alias)
                break
    return mapping


def ingest_reviews_csv(filepath: str, destination_override: Optional[str] = None) -> int:
    """
    Ingest a CSV file of travel reviews into the vector store.

    The CSV must have at minimum a text/review column.
    Destination can be auto-detected from the data or passed explicitly.

    Compatible with common open datasets:
      - Kaggle "515K Hotel Reviews" (Europe)
      - Kaggle "TripAdvisor Hotel Reviews"
      - Any CSV with a review text column
    """
    total = 0
    path = Path(filepath)
    console.print(f"[cyan]Ingesting reviews:[/cyan] {path.name}")

    with open(filepath, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        headers = next(reader)
        col_map = _map_columns(headers)

        if "content" not in col_map:
            console.print(
                f"[red]✗ Could not find review text column in {path.name}[/red]\n"
                f"  Found columns: {headers}\n"
                f"  Expected one of: {REVIEW_COLUMN_ALIASES['content']}"
            )
            return 0

        rows = list(reader)

    for row in track(rows, description=f"  Embedding {path.name}..."):
        if len(row) <= max(col_map.values()):
            continue

        content = row[col_map["content"]].strip()
        if len(content) < 50:          # skip very short reviews
            continue

        destination = destination_override or (
            row[col_map["destination"]].strip() if "destination" in col_map else "Unknown"
        )
        title   = row[col_map["title"]].strip()   if "title"    in col_map else ""
        rating  = row[col_map["rating"]].strip()  if "rating"   in col_map else ""
        reviewer= row[col_map["reviewer"]].strip()if "reviewer" in col_map else ""

        meta = {}
        if rating:   meta["rating"]   = rating
        if reviewer: meta["reviewer"] = reviewer

        n = index_document(
            content=content,
            destination=destination,
            source_type="review",
            title=title,
            metadata=meta,
        )
        total += n

    console.print(f"  [green]✓ Indexed {total} chunks from {len(rows)} reviews[/green]")
    return total


def ingest_reviews_dir(directory: str) -> int:
    """Ingest all .csv files in a directory."""
    total = 0
    csv_files = list(Path(directory).glob("*.csv"))
    if not csv_files:
        console.print(f"[yellow]No .csv files found in {directory}[/yellow]")
        console.print(
            "  Drop any travel review CSV here. Compatible formats:\n"
            "  • Kaggle '515K Hotel Reviews'\n"
            "  • Any CSV with a 'review' or 'text' column\n"
            "  • A 'destination' or 'city' column is helpful but not required\n"
            "    (use --destination flag to set it manually)"
        )
        return 0
    for f in csv_files:
        total += ingest_reviews_csv(str(f))
    return total


# ── Notes ingestion ───────────────────────────────────────────────────────────

def ingest_notes_dir(directory: str) -> int:
    """
    Ingest all .txt and .md files from the notes directory.

    File naming convention (optional but recommended):
      tokyo_food_tips.md          → destination inferred as "tokyo"
      bali_general.txt            → destination inferred as "bali"
      my_paris_trip_notes.md      → destination inferred from content

    Front-matter supported (YAML-style, optional):
      ---
      destination: Paris
      title: My Paris Notes
      ---
      The rest is your note content...
    """
    total = 0
    note_files = list(Path(directory).glob("*.txt")) + list(Path(directory).glob("*.md"))

    if not note_files:
        console.print(f"[yellow]No .txt or .md files found in {directory}[/yellow]")
        console.print(
            "  Drop your personal travel notes here.\n"
            "  Naming tip: start filenames with the destination, e.g. tokyo_tips.md"
        )
        return 0

    for f in track(note_files, description="  Embedding notes..."):
        raw = f.read_text(encoding="utf-8", errors="replace")
        destination, title, content = _parse_note(raw, f.stem)

        n = index_document(
            content=content,
            destination=destination,
            source_type="note",
            title=title or f.stem,
            metadata={"filename": f.name},
        )
        total += n
        console.print(f"  [green]✓[/green] {f.name} → {n} chunks (dest: {destination})")

    return total


def _parse_note(raw: str, filename_stem: str) -> tuple:
    """
    Extract destination, title, and body from a note file.
    Supports optional YAML-style front-matter block.
    """
    destination = ""
    title = ""
    content = raw

    # Try YAML front-matter
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            fm_block = parts[1]
            content = parts[2].strip()
            for line in fm_block.splitlines():
                if line.lower().startswith("destination:"):
                    destination = line.split(":", 1)[1].strip()
                elif line.lower().startswith("title:"):
                    title = line.split(":", 1)[1].strip()

    # Infer destination from filename (e.g. "tokyo_food_tips" → "Tokyo")
    if not destination:
        first_word = filename_stem.split("_")[0].split("-")[0]
        destination = first_word.title()

    return destination, title, content


# ── Seed data (demo content when no files provided) ───────────────────────────

SEED_REVIEWS = [
    ("Bali",      "review", "Amazing spiritual retreat",
     "The rice terraces in Ubud are absolutely breathtaking. We hired a local guide who took us through hidden paths most tourists never see. The food at Warung Babi Guling was incredible — crispy pork skin and rich spices. Make sure to visit early morning to avoid crowds at Tanah Lot temple."),
    ("Tokyo",     "review", "Food paradise",
     "Tsukiji outer market for breakfast sushi is non-negotiable. The ramen at Ichiran in Shibuya — get the solo booth, it's a unique experience. Shinjuku Golden Gai for tiny bars at night. Prepaid Suica card makes transit effortless. Avoid rush hour on the Yamanote line at all costs."),
    ("Paris",     "review", "More than the Eiffel Tower",
     "Skip the Louvre on weekends — Tuesday mornings are quiet. The Marais district has the best falafel in Europe at L'As du Fallafel. Rent a Vélib bike and ride along the Seine. Montmartre at dawn before the tourists arrive is magical. Book Musée d'Orsay tickets weeks in advance."),
    ("Patagonia", "review", "Hiking bucket list complete",
     "Torres del Paine W trek: do it east to west to end at the Towers with morning light. Weather changes every 20 minutes — pack waterproofs even in summer. The refugios book out a year in advance for peak season. Puerto Natales has surprisingly good lamb stew to recover after the trail."),
    ("Marrakech", "review", "Sensory overload in the best way",
     "Get lost in the medina on purpose — you'll find the best stalls that way. Djemaa el-Fna square transforms completely after dark with food stalls and musicians. Hire a guide for the tanneries to get rooftop access. Stay in a riad for the authentic courtyard experience. Negotiate everything."),
    ("Kyoto",     "review", "Best in spring and autumn",
     "Arashiyama bamboo grove at 6am is serene and crowd-free. Fushimi Inari is better explored past the first hour — most tourists turn back. Nishiki market for street food. Rent a kimono in Gion and walk the stone-paved Ninenzaka. The night buses are a great way to temple-hop affordably."),
]

SEED_NOTES = [
    ("Tokyo", "note", "Personal Tokyo Tips",
     "Best conveience store meal: 7-Eleven tamago sando (egg sandwich). The teamLab digital art installations sell out fast — book online. If you only do one day trip, make it Nikko or Kamakura. Pocket wifi from the airport is cheaper than roaming. Google Maps works perfectly for transit here."),
    ("Bali", "note", "Bali Hidden Gems",
     "Munduk waterfall in the north: 2 hour drive but zero tourists and cooler temperatures. Sidemen valley for rice terrace walks without the Ubud crowds. Nusa Penida for the cliffside Angel's Billabong — only reachable by fast boat. Avoid Kuta entirely. Canggu has better surf for beginners than Seminyak."),
    ("Iceland", "note", "Iceland Practical Notes",
     "Northern lights: September to March, need clear skies and solar activity KP3+. Rent a 4WD even in summer for the highland F-roads. Fill up petrol at every station — gaps can be 150km. The Reykjanes peninsula geothermal area is free and spectacular. Blue Lagoon is expensive and crowded — go to Sky Lagoon instead."),
]


def seed_demo_data() -> int:
    """Index built-in demo content so the RAG works immediately without any files."""
    total = 0
    console.print("[dim]Seeding demo review and note data...[/dim]")

    for dest, stype, title, content in SEED_REVIEWS + SEED_NOTES:
        n = index_document(
            content=content,
            destination=dest,
            source_type=stype,
            title=title,
            metadata={"source": "demo_seed"},
        )
        total += n

    console.print(f"[green]✓ Demo data: {total} chunks indexed[/green]")
    return total


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_stats() -> None:
    stats = get_index_stats()
    console.print(f"\n[bold cyan]RAG Index Stats[/bold cyan]")
    console.print(f"  Total chunks: [bold]{stats['total_chunks']}[/bold]")

    t = Table(box=box.SIMPLE, show_header=True)
    t.add_column("Source Type", style="cyan")
    t.add_column("Chunks", justify="right")
    for k, v in stats["by_type"].items():
        t.add_row(k, str(v))
    console.print(t)

    t2 = Table(box=box.SIMPLE, show_header=True, title="Top Destinations")
    t2.add_column("Destination", style="green")
    t2.add_column("Chunks", justify="right")
    for k, v in stats["top_destinations"].items():
        t2.add_row(k, str(v))
    console.print(t2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TravelDude RAG Ingestion Pipeline")
    parser.add_argument("--reviews",     metavar="DIR",  help="Ingest CSV reviews from directory")
    parser.add_argument("--notes",       metavar="DIR",  help="Ingest .txt/.md notes from directory")
    parser.add_argument("--all",         action="store_true", help="Ingest from default data/ directories")
    parser.add_argument("--seed",        action="store_true", help="Index built-in demo data")
    parser.add_argument("--stats",       action="store_true", help="Print index statistics")
    parser.add_argument("--destination", metavar="NAME", help="Override destination for --reviews")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.seed:
        seed_demo_data()
        print_stats()
    elif args.all:
        ingest_reviews_dir(str(REVIEWS_DIR))
        ingest_notes_dir(str(NOTES_DIR))
        print_stats()
    elif args.reviews:
        if os.path.isdir(args.reviews):
            ingest_reviews_dir(args.reviews)
        else:
            ingest_reviews_csv(args.reviews, destination_override=args.destination)
        print_stats()
    elif args.notes:
        ingest_notes_dir(args.notes)
        print_stats()
    else:
        parser.print_help()
