"""
api_clients.py
Wrappers for all open travel APIs used by TravelDude.
All free-tier friendly. No credit card required (except Anthropic).
"""

import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

OTM_KEY = os.getenv("OPENTRIPMAP_API_KEY", "")
UNSPLASH_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")

# ── OpenTripMap ──────────────────────────────────────────────────────────────

OTM_BASE = "https://api.opentripmap.com/0.1/en"


def get_pois_by_city(city: str, radius_m: int = 5000, limit: int = 10) -> List[Dict]:
    """
    Fetch top points of interest near a city using OpenTripMap.
    Requires a free API key from https://opentripmap.io/product
    """
    if not OTM_KEY:
        print("[API] No OTM key — returning mock POIs.")
        return _mock_pois(city)

    # Step 1: geocode city to lat/lon
    geo_url = f"{OTM_BASE}/places/geoname"
    geo_resp = requests.get(geo_url, params={"name": city, "apikey": OTM_KEY}, timeout=10)
    if geo_resp.status_code != 200:
        return _mock_pois(city)
    geo = geo_resp.json()
    lat, lon = geo.get("lat"), geo.get("lon")
    if not lat:
        return _mock_pois(city)

    # Step 2: fetch POIs in radius
    poi_url = f"{OTM_BASE}/places/radius"
    poi_resp = requests.get(
        poi_url,
        params={
            "radius": radius_m,
            "lon": lon,
            "lat": lat,
            "limit": limit,
            "format": "json",
            "apikey": OTM_KEY,
        },
        timeout=10,
    )
    if poi_resp.status_code != 200:
        return _mock_pois(city)

    pois = poi_resp.json()
    return [
        {
            "name": p.get("name", "Unknown"),
            "kinds": p.get("kinds", ""),
            "dist": p.get("dist", 0),
        }
        for p in pois
        if p.get("name")
    ]


def _mock_pois(city: str) -> List[Dict]:
    """Fallback mock data when no API key is available."""
    return [
        {"name": f"{city} Central Museum", "kinds": "museums,cultural", "dist": 500},
        {"name": f"{city} Old Town", "kinds": "historic,architecture", "dist": 800},
        {"name": f"{city} Botanical Garden", "kinds": "nature,parks", "dist": 1200},
    ]


# ── RestCountries ────────────────────────────────────────────────────────────

RC_BASE = "https://restcountries.com/v3.1"


def get_country_info(country_name: str) -> Dict:
    """Fetch country metadata (currency, language, region) — no API key needed."""
    try:
        resp = requests.get(
            f"{RC_BASE}/name/{country_name}",
            params={"fields": "name,currencies,languages,region,flags,capital"},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data:
                c = data[0]
                currencies = list(c.get("currencies", {}).keys())
                languages = list(c.get("languages", {}).values())
                return {
                    "region": c.get("region", ""),
                    "capital": (c.get("capital") or [""])[0],
                    "currencies": currencies,
                    "languages": languages,
                    "flag": c.get("flags", {}).get("emoji", "🏳"),
                }
    except Exception:
        pass
    return {"region": "", "capital": "", "currencies": [], "languages": [], "flag": "🌍"}


# ── Open-Meteo ───────────────────────────────────────────────────────────────

OM_BASE = "https://api.open-meteo.com/v1"


def get_climate_summary(lat: float, lon: float) -> Dict:
    """Fetch average temperature forecast — no API key needed."""
    try:
        resp = requests.get(
            f"{OM_BASE}/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "forecast_days": 7,
                "timezone": "auto",
            },
            timeout=8,
        )
        if resp.status_code == 200:
            daily = resp.json().get("daily", {})
            temps_max = daily.get("temperature_2m_max", [])
            temps_min = daily.get("temperature_2m_min", [])
            precip = daily.get("precipitation_sum", [])
            if temps_max and temps_min:
                avg_high = round(sum(temps_max) / len(temps_max), 1)
                avg_low = round(sum(temps_min) / len(temps_min), 1)
                total_rain = round(sum(p for p in precip if p), 1)
                return {
                    "avg_high_c": avg_high,
                    "avg_low_c": avg_low,
                    "weekly_precip_mm": total_rain,
                }
    except Exception:
        pass
    return {"avg_high_c": None, "avg_low_c": None, "weekly_precip_mm": None}


# ── Unsplash ─────────────────────────────────────────────────────────────────

UNSPLASH_BASE = "https://api.unsplash.com"


def get_destination_photo_url(destination: str) -> Optional[str]:
    """Fetch a representative photo URL for a destination."""
    if not UNSPLASH_KEY:
        return None
    try:
        resp = requests.get(
            f"{UNSPLASH_BASE}/search/photos",
            params={"query": destination, "per_page": 1, "orientation": "landscape"},
            headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"},
            timeout=8,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                return results[0]["urls"]["regular"]
    except Exception:
        pass
    return None
