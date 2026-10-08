"""
biodiversity_forecast.py — JARVIS tool for querying predicted NDVI change.

Thin wrapper around results/forecast_2026.csv produced by make_map.py.
Fast: no model loading, no re-prediction.

Usage:
    from biodiversity_forecast import biodiversity_forecast
    biodiversity_forecast("Pulau Ubin")

Or run directly:
    python biodiversity_forecast.py "Pulau Ubin"
    python biodiversity_forecast.py --list
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

FORECAST_CSV = Path(__file__).parent / "results" / "forecast_2026.csv"

# Aliases for common variations the LLM or user might say
ISLAND_ALIASES = {
    "pulau ubin": "Pulau Ubin",
    "ubin": "Pulau Ubin",
    "st johns": "St. John's",
    "st. john's": "St. John's",
    "st john's": "St. John's",
    "saint johns": "St. John's",
    "saint john's": "St. John's",
    "st john": "St. John's",
    "kusu": "Kusu",
    "pulau kusu": "Kusu",
    "lazarus": "Lazarus",
    "pulau lazarus": "Lazarus",
    "sisters": "Sisters",
    "sisters'": "Sisters",
    "sisters islands": "Sisters",
    "pulau hantu": "Pulau Hantu",
    "hantu": "Pulau Hantu",
}


def _normalise(name: str) -> Optional[str]:
    """Map a user-provided island name to the canonical name used in the CSV."""
    key = name.strip().lower()
    if key in ISLAND_ALIASES:
        return ISLAND_ALIASES[key]
    # Try a looser match: canonical names lowercased
    for alias, canonical in ISLAND_ALIASES.items():
        if alias in key or key in alias:
            return canonical
    return None


def _load_forecast() -> pd.DataFrame:
    if not FORECAST_CSV.exists():
        raise FileNotFoundError(
            f"Missing {FORECAST_CSV}. Run make_map.py first."
        )
    return pd.read_csv(FORECAST_CSV)


def biodiversity_forecast(island_name: str) -> str:
    """
    Return a one-line forecast summary for the named island.

    Example:
        >>> biodiversity_forecast("Pulau Ubin")
        'Pulau Ubin: predicted NDVI change +14.3% by 2026 (stable).'
    """
    canonical = _normalise(island_name)
    if canonical is None:
        return (
            f"Unknown island '{island_name}'. "
            "Available: Pulau Ubin, St. John's, Kusu, Lazarus, Sisters', "
            "Pulau Hantu."
        )

    df = _load_forecast()
    row = df[df["island"] == canonical]
    if row.empty:
        return f"No forecast data on file for {canonical}."

    r = row.iloc[0]
    change = float(r["change_pct"])
    category = str(r["category"])
    rmse = float(r["rmse"]) if not pd.isna(r["rmse"]) else None

    if category == "green":
        label = "stable"
    elif category == "yellow":
        label = "moderate risk"
    else:
        label = "high risk"

    sign = "+" if change >= 0 else ""
    base = (
        f"{canonical}: predicted NDVI change {sign}{change:.1f}% by 2026 "
        f"({label})."
    )
    if rmse is not None:
        base += f" Model RMSE ±{rmse:.3f} — treat changes smaller than this as noise."

    return base


def list_islands() -> str:
    """Return a formatted list of all islands with their forecasts."""
    df = _load_forecast().sort_values("change_pct")
    lines = ["Predicted 2026 NDVI change:"]
    for _, r in df.iterrows():
        sign = "+" if r["change_pct"] >= 0 else ""
        lines.append(
            f"  {r['island']:<14} {sign}{r['change_pct']:>6.1f}%  ({r['category']})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI for standalone testing
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--list":
        print(list_islands())
        sys.exit(0)

    island = " ".join(sys.argv[1:])
    print(biodiversity_forecast(island))