"""
fetch_observations.py — iNaturalist observations for Singapore's micro-islands.

Uses the v1 API (returns full records by default). Queries each island
centroid with a small radius, across 2-year date slices to avoid the
10,000-result per-query cap, dedupes by observation ID, assigns each
observation to its nearest island centroid.

Output: data/islands_observations.parquet
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

OUT_PATH = Path("data/islands_observations.parquet")
API_URL = "https://api.inaturalist.org/v1/observations"

START_YEAR = 2018
END_YEAR = date.today().year
SLICE_YEARS = 2          # each query covers this many years

PER_PAGE = 200
RATE_LIMIT_SLEEP = 1.1
MAX_RETRIES = 4
MAX_PAGES_PER_QUERY = 50 # v1 hard cap

SEARCH_RADIUS_KM = 1.5

ISLANDS = {
    "Pulau Ubin":  {"lat": 1.4097, "lon": 103.9586},
    "St. John's":  {"lat": 1.2168, "lon": 103.8478},
    "Kusu":        {"lat": 1.2245, "lon": 103.8605},
    "Lazarus":     {"lat": 1.2260, "lon": 103.8550},
    "Sisters":     {"lat": 1.2118, "lon": 103.8352},
    "Pulau Hantu": {"lat": 1.2258, "lon": 103.7488},
}

MAX_ASSIGN_KM = 2.0

log = logging.getLogger("fetch_observations")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def get_token() -> str:
    load_dotenv()
    token = os.getenv("INATURALIST_TOKEN")
    if not token:
        sys.exit(
            "Missing INATURALIST_TOKEN in .env.\n"
            "Get one at https://www.inaturalist.org/users/api_token "
            "(expires every 24 hours)."
        )
    return token


def date_slices() -> list[tuple[str, str]]:
    """Generate (d1, d2) windows covering START_YEAR → today, SLICE_YEARS each."""
    slices = []
    y = START_YEAR
    today = date.today()
    while y <= END_YEAR:
        d1 = f"{y}-01-01"
        end_y = min(y + SLICE_YEARS - 1, END_YEAR)
        if end_y == today.year:
            d2 = today.isoformat()
        else:
            d2 = f"{end_y}-12-31"
        slices.append((d1, d2))
        y += SLICE_YEARS
    return slices


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def nearest_island(lat: float, lon: float) -> tuple[str | None, float]:
    best_name, best_d = None, float("inf")
    for name, spec in ISLANDS.items():
        d = haversine_km(lat, lon, spec["lat"], spec["lon"])
        if d < best_d:
            best_name, best_d = name, d
    return best_name, best_d


def fetch_page(token: str, island_spec: dict, d1: str, d2: str,
               page: int, per_page: int = PER_PAGE) -> dict:
    params = {
        "lat": island_spec["lat"],
        "lng": island_spec["lon"],
        "radius": SEARCH_RADIUS_KM,
        "d1": d1, "d2": d2,
        "quality_grade": "research",
        "per_page": per_page,
        "page": page,
        "order_by": "observed_on",
        "order": "asc",
    }
    headers = {"Authorization": f"Bearer {token}"}

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(API_URL, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                log.warning("Rate limited (429). Sleeping %ds…", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = 2 ** attempt
            log.warning("Page %d attempt %d failed: %s — retry in %ds",
                        page, attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Page {page}: giving up after {MAX_RETRIES}") from last_err


def parse_record(obs: dict) -> dict | None:
    loc = obs.get("location")
    if not loc:
        return None
    try:
        lat_s, lon_s = loc.split(",")
        lat, lon = float(lat_s), float(lon_s)
    except Exception:  # noqa: BLE001
        return None

    if obs.get("geoprivacy") == "obscured" or obs.get("taxon_geoprivacy") == "obscured":
        return None

    observed_on = obs.get("observed_on")
    if not observed_on:
        return None

    taxon = obs.get("taxon") or {}
    return {
        "obs_id": obs.get("id"),
        "observed_on": observed_on,
        "taxon_name": taxon.get("name"),
        "common_name": taxon.get("preferred_common_name"),
        "iconic_taxon": taxon.get("iconic_taxon_name"),
        "latitude": lat,
        "longitude": lon,
        "quality_grade": obs.get("quality_grade"),
    }


def fetch_island(token: str, island: str, spec: dict) -> list[dict]:
    """Fetch all observations for one island, sliced across date windows."""
    records: list[dict] = []
    for d1, d2 in date_slices():
        first = fetch_page(token, spec, d1, d2, page=1)
        total = first.get("total_results", 0)
        n_pages = min(math.ceil(total / PER_PAGE) if total else 0,
                      MAX_PAGES_PER_QUERY)

        log.info("    %s → %s: %d results → %d pages",
                 d1, d2, total, n_pages)

        if total > MAX_PAGES_PER_QUERY * PER_PAGE:
            log.warning("    Slice %s → %s exceeds 10k cap (%d results). "
                        "Some records in this slice may be dropped.",
                        d1, d2, total)

        records.extend(first.get("results", []))
        for p in range(2, n_pages + 1):
            time.sleep(RATE_LIMIT_SLEEP)
            records.extend(fetch_page(token, spec, d1, d2, page=p).get("results", []))

    return records


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def run_test(token: str) -> None:
    log.info("TEST MODE — one page for Pulau Ubin (radius %.1f km)",
             SEARCH_RADIUS_KM)
    d1, d2 = date_slices()[0]
    log.info("Using first date slice: %s → %s", d1, d2)

    data = fetch_page(token, ISLANDS["Pulau Ubin"], d1, d2, page=1, per_page=10)
    total = data.get("total_results", "?")
    log.info("total_results for Pulau Ubin in that slice: %s", total)

    rows = [r for r in (parse_record(o) for o in data.get("results", [])) if r]
    if not rows:
        log.error("Zero usable rows. Check token or query params.")
        return

    df = pd.DataFrame(rows)
    log.info("Got %d usable rows:\n%s", len(df), df.to_string(index=False))

    assigns = df.apply(lambda r: nearest_island(r["latitude"], r["longitude"]),
                       axis=1, result_type="expand")
    df["assigned_island"] = assigns[0]
    df["distance_km"] = assigns[1].round(3)
    log.info("\nAfter nearest-island assignment:\n%s",
             df[["assigned_island", "distance_km"]].to_string(index=False))

    log.info("\nDate slices that will be queried per island: %s", date_slices())


def run_full(token: str) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info("Date slices per island: %s", date_slices())
    log.info("")

    all_records: list[dict] = []
    seen_ids: set[int] = set()

    for island, spec in ISLANDS.items():
        log.info("%s — querying (radius %.1f km, %d date slices)",
                 island, SEARCH_RADIUS_KM, len(date_slices()))

        island_records = fetch_island(token, island, spec)

        new_count = 0
        for rec in island_records:
            rid = rec.get("id")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                all_records.append(rec)
                new_count += 1
        log.info("  %s: %d unique records (running total: %d)",
                 island, new_count, len(all_records))

    log.info("")
    log.info("Parsing %d unique records…", len(all_records))
    parsed = [r for r in (parse_record(o) for o in all_records) if r]
    log.info("Usable after filtering: %d", len(parsed))

    if not parsed:
        log.error("No usable observations. Aborting.")
        return

    df = pd.DataFrame(parsed)

    assigns = df.apply(lambda r: nearest_island(r["latitude"], r["longitude"]),
                       axis=1, result_type="expand")
    df["island"] = assigns[0]
    df["distance_km"] = assigns[1]

    before = len(df)
    df = df[df["distance_km"] <= MAX_ASSIGN_KM].reset_index(drop=True)
    log.info("Kept %d / %d observations within %.1f km of a centroid",
             len(df), before, MAX_ASSIGN_KM)

    df = df.sort_values(["island", "observed_on"]).reset_index(drop=True)
    df.to_parquet(OUT_PATH, index=False)

    log.info("")
    log.info("Done. Wrote %d rows to %s", len(df), OUT_PATH)
    log.info("")
    log.info("Per-island summary:")
    summary = df.groupby("island").agg(
        observations=("observed_on", "size"),
        unique_species=("taxon_name", "nunique"),
        first_seen=("observed_on", "min"),
        last_seen=("observed_on", "max"),
    )
    log.info("\n%s", summary.to_string())

    log.info("")
    log.info("Yearly observation counts per island:")
    df["year"] = pd.to_datetime(df["observed_on"]).dt.year
    pivot = df.pivot_table(index="year", columns="island",
                           values="obs_id", aggfunc="count").fillna(0).astype(int)
    log.info("\n%s", pivot.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true",
                        help="Fetch 10 rows only, verify auth + query")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    token = get_token()
    if args.test:
        run_test(token)
    else:
        run_full(token)


if __name__ == "__main__":
    main()