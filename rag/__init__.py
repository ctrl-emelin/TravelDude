"""
TravelDude RAG package.

Quick start:
    from rag.ingest import seed_demo_data, ingest_notes_dir, ingest_reviews_csv
    from rag.vector_store import retrieve, get_index_stats
    from rag.rag_llm import rag_generate_itinerary, TravelChat
"""
from .vector_store import index_document, retrieve, get_index_stats
from .ingest import seed_demo_data, ingest_reviews_csv, ingest_notes_dir
from .rag_llm import rag_generate_itinerary, TravelChat

__all__ = [
    "index_document",
    "retrieve",
    "get_index_stats",
    "seed_demo_data",
    "ingest_reviews_csv",
    "ingest_notes_dir",
    "rag_generate_itinerary",
    "TravelChat",
]
