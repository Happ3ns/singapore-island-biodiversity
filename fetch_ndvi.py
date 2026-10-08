"""
fetch_ndvi.py — Monthly NDVI statistics for Singapore's micro-islands.

Source: Sentinel-2 L2A via Copernicus Data Space Ecosystem (CDSE)
        Statistical API (server-side evalscript; no raster downloads).

Output: data/islands_ndvi.parquet
Columns: island, year, month, mean_ndvi, std_ndvi, valid_pixels
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path


os.environ.setdefault("SH_BASE_URL", "https://sh.dataspace.copernicus.eu")
os.environ.setdefault(
    "SH_TOKEN_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token",
)

import pandas as pd
from dotenv import load_dotenv

from sentinelhub import (
    CRS,
    DataCollection,
    Geometry,
    SHConfig,
    SentinelHubStatistical,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

OUT_PATH = Path("data/islands_ndvi.parquet")
START_YEAR = 2018
END_YEAR = 2026         # inclusive; last interval is 2025-12

ISLANDS = {
    "Pulau Ubin":  {"lat": 1.4097, "lon": 103.9586, "radius_km": 2.0},
    "St. John's":  {"lat": 1.2168, "lon": 103.8478, "radius_km": 0.5},
    "Kusu":        {"lat": 1.2245, "lon": 103.8605, "radius_km": 0.3},
    "Lazarus":     {"lat": 1.2260, "lon": 103.8550, "radius_km": 0.4},
    "Sisters":     {"lat": 1.2118, "lon": 103.8352, "radius_km": 0.3},
    "Pulau Hantu": {"lat": 1.2258, "lon": 103.7488, "radius_km": 0.3},
}

# Per-pixel SCL cloud mask. Excluded classes:
#   0  no data          1  saturated/defective
#   3  cloud shadow     8  cloud medium prob
#   9  cloud high prob  10 thin cirrus
#   11 snow/ice
# dataMask == 0 also excludes the pixel from statistics.
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL", "dataMask"] }],
    output: [
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

const BAD_SCL = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11];

function evaluatePixel(s) {
  if (BAD_SCL.includes(s.SCL) || s.dataMask === 0) {
    return { ndvi: [0.0], dataMask: [0] };
  }
  const ndvi = (s.B08 - s.B04) / (s.B08 + s.B04);
  return { ndvi: [ndvi], dataMask: [1] };
}
"""

log = logging.getLogger("fetch_ndvi")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def build_config() -> SHConfig:
    load_dotenv()
    client_id = os.getenv("COPERNICUS_CLIENT_ID")
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        sys.exit(
            "Missing COPERNICUS_CLIENT_ID / COPERNICUS_CLIENT_SECRET.\n"
            "Create a .env file in the repo root (see Phase 0)."
        )

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
        "protocol/openid-connect/token"
    )

    # Persist as a named profile on disk so internal SHConfig() calls pick
    # up CDSE endpoints too.
    try:
        config.save("cdse")
    except Exception as exc:
        log.warning("Could not save config profile: %s", exc)

    log.info("Using base_url: %s", config.sh_base_url)
    log.info("Using token_url: %s", config.sh_token_url)
    log.info("Using client_id: %s…", client_id[:8])

    # Sanity check: resolve the collection URL the same way fetch_year will.
    test_url = DataCollection.SENTINEL2_L2A.define_from(
        "s2l2a", service_url=config.sh_base_url
    ).service_url
    log.info("Resolved S2 L2A service_url: %s", test_url)

    return config


def circle_geometry(lat: float, lon: float, radius_km: float, n: int = 64) -> Geometry:
    """Approximate a circle as an n-gon in WGS84. Good enough at this scale."""
    lat_deg_per_km = 1.0 / 111.0
    lon_deg_per_km = 1.0 / (111.0 * math.cos(math.radians(lat)))
    ring = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        ring.append([
            lon + radius_km * lon_deg_per_km * math.cos(theta),
            lat + radius_km * lat_deg_per_km * math.sin(theta),
        ])
    ring.append(ring[0])  # GeoJSON rings must be closed
    return Geometry({"type": "Polygon", "coordinates": [ring]}, crs=CRS.WGS84)


def fetch_year(config: SHConfig, island: str, year: int, max_retries: int = 5):
    spec = ISLANDS[island]
    geom = circle_geometry(spec["lat"], spec["lon"], spec["radius_km"])

    request = SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=EVALSCRIPT,
            time_interval=(f"{year}-01-01", f"{year + 1}-01-01"),
            aggregation_interval="P1M",
        ),
        input_data=[
            SentinelHubStatistical.input_data(
                DataCollection.SENTINEL2_L2A.define_from(
                    "s2l2a",
                    service_url=config.sh_base_url,
                ),
                maxcc=1.0,
            )
        ],
        geometry=geom,
        config=config,
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            return request.get_data()
        except Exception as exc:  # noqa: BLE001 — CDSE raises a zoo of types
            last_err = exc
            wait = 2 ** attempt
            log.warning(
                "  %s %d failed (attempt %d/%d): %s — retrying in %ds",
                island, year, attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"{island} {year}: giving up after {max_retries} tries") from last_err


def parse_response(response, island: str) -> list[dict]:
    rows = []
    for interval in response[0]["data"]:
        date_from = interval["interval"]["from"][:10]
        y, m = int(date_from[:4]), int(date_from[5:7])

        stats = interval["outputs"]["ndvi"]["bands"]["B0"]["stats"]
        sample_count = stats.get("sampleCount", 0) or 0
        no_data = stats.get("noDataCount", 0) or 0
        valid = sample_count - no_data

        MIN_VALID_PIXELS = 100  # below this, the monthly mean is noise

        rows.append({
            "island": island,
            "year": y,
            "month": m,
            "mean_ndvi": stats.get("mean") if valid >= MIN_VALID_PIXELS else None,
            "std_ndvi": stats.get("stDev") if valid >= MIN_VALID_PIXELS else None,
            "valid_pixels": valid,
        })
    return rows


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

def run_test(config: SHConfig) -> None:
    """One island, one year. Costs almost nothing. Run this first."""
    log.info("TEST MODE — Pulau Ubin, 2024 only")
    response = fetch_year(config, "Pulau Ubin", 2024)
    rows = parse_response(response, "Pulau Ubin")
    df = pd.DataFrame(rows)
    log.info("Got %d monthly rows:\n%s", len(df), df.to_string(index=False))

    usable = df[df["valid_pixels"] > 0]
    if usable.empty:
        log.error(
            "Zero valid pixels across all of 2024. Something is wrong — "
            "check credentials, geometry, and the evalscript."
        )
        return
    log.info(
        "OK. Mean NDVI across usable months: %.3f (range %.3f – %.3f)",
        usable["mean_ndvi"].mean(),
        usable["mean_ndvi"].min(),
        usable["mean_ndvi"].max(),
    )


def run_full(config: SHConfig) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: if we already have rows, skip (island, year) pairs we've done.
    done: set[tuple[str, int]] = set()
    existing = pd.DataFrame()
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        done = set(zip(existing["island"], existing["year"]))
        log.info("Resuming — %d rows already on disk", len(existing))

    all_rows: list[dict] = existing.to_dict("records") if not existing.empty else []

    for island in ISLANDS:
        for year in range(START_YEAR, END_YEAR + 1):
            if (island, year) in done:
                log.info("%s %d — already done, skipping", island, year)
                continue

            log.info("%s %d — fetching", island, year)
            try:
                response = fetch_year(config, island, year)
                rows = parse_response(response, island)
            except Exception:
                log.exception("%s %d — FAILED, continuing", island, year)
                continue

            all_rows.extend(rows)

            # Checkpoint after every island-year so a crash never loses everything.
            pd.DataFrame(all_rows).to_parquet(OUT_PATH, index=False)

            valid = sum(1 for r in rows if r["valid_pixels"] > 0)
            log.info("  %s %d — %d/12 months usable", island, year, valid)

    df = pd.DataFrame(all_rows).sort_values(["island", "year", "month"]).reset_index(drop=True)
    df.to_parquet(OUT_PATH, index=False)

    log.info("")
    log.info("Done. Wrote %d rows to %s", len(df), OUT_PATH)
    log.info("")
    log.info("Coverage by island:")
    summary = df.groupby("island").agg(
        months=("month", "size"),
        usable=("valid_pixels", lambda s: (s > 0).sum()),
        mean_ndvi=("mean_ndvi", "mean"),
    )
    log.info("\n%s", summary.to_string())

    gaps = df[df["valid_pixels"] == 0]
    if not gaps.empty:
        log.warning(
            "%d island-months have zero valid pixels (heavy cloud or too small "
            "for 10m resolution). These are your 'insufficient data' rows — "
            "mention them in the writeup's limitations section.",
            len(gaps),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Smoke test: one island, one year")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = build_config()
    if args.test:
        run_test(config)
    else:
        run_full(config)


if __name__ == "__main__":
    main()