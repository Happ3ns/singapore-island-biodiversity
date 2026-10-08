"""
make_map.py — Interactive + static map of predicted 2026 NDVI change per island.

Compares predicted 2026 to *predicted* 2025 (not actual 2025) to cancel out
XGBoost's regression-to-the-mean bias. Both years come from the same model, so
the bias affects both equally and the change score becomes meaningful.

Outputs:
    docs/index.html          — interactive Folium map
    docs/map_preview.png     — static preview for README
    results/forecast_2026.csv
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NDVI_PATH = Path("data/islands_ndvi.parquet")
OBS_PATH = Path("data/islands_observations.parquet")
MODEL_PATH = Path("models/ndvi_model.pkl")
PRED_PATH = Path("results/predictions.csv")
FORECAST_CSV = Path("results/forecast_2026.csv")
OUT_HTML = Path("docs/index.html")
OUT_PNG = Path("docs/map_preview.png")

FORECAST_YEAR = 2026
COMPARISON_YEAR = 2025

ISLANDS = {
    "Pulau Ubin":  {"lat": 1.4097, "lon": 103.9586, "area_km2": 10.2, "dist_km": 5.0},
    "St. John's":  {"lat": 1.2168, "lon": 103.8478, "area_km2": 0.41, "dist_km": 6.5},
    "Kusu":        {"lat": 1.2245, "lon": 103.8605, "area_km2": 0.085, "dist_km": 5.5},
    "Lazarus":     {"lat": 1.2260, "lon": 103.8550, "area_km2": 0.18, "dist_km": 6.0},
    "Sisters":     {"lat": 1.2118, "lon": 103.8352, "area_km2": 0.055, "dist_km": 7.0},
    "Pulau Hantu": {"lat": 1.2258, "lon": 103.7488, "area_km2": 0.13, "dist_km": 6.0},
}

FEATURES = [
    "ndvi_last_observed",
    "ndvi_last_gap",
    "ndvi_same_month_last_year",
    "ndvi_trend_12m",
    "ndvi_volatility_12m",
    "ndvi_mean_12m",
    "observation_count_12m",
    "species_richness_12m",
    "island_area_km2",
    "distance_to_mainland_km",
    "month_sin",
    "month_cos",
]

GREEN_THRESHOLD = -3.0
YELLOW_THRESHOLD = -8.0

# Satellite base imagery — Esri World Imagery (free, no API key required).
ESRI_SAT_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_ATTR = "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics"

log = logging.getLogger("make_map")


def build_forecast_features(ndvi: pd.DataFrame, obs: pd.DataFrame,
                            forecast_year: int) -> pd.DataFrame:
    """One row per (island, month) in forecast_year, features from past data only."""
    ndvi = ndvi.copy()
    ndvi["t"] = ndvi["year"] * 12 + ndvi["month"]

    obs = obs.copy()
    obs["observed_on"] = pd.to_datetime(obs["observed_on"])
    obs["t"] = obs["observed_on"].dt.year * 12 + obs["observed_on"].dt.month

    rows = []
    for island, spec in ISLANDS.items():
        sub = ndvi[ndvi["island"] == island].sort_values("t")
        obs_sub = obs[obs["island"] == island]

        for month in range(1, 13):
            t = forecast_year * 12 + month
            past_obs = sub[(sub["t"] < t) & (sub["mean_ndvi"].notna())]

            feat = {
                "island": island,
                "year": forecast_year,
                "month": month,
                "island_area_km2": spec["area_km2"],
                "distance_to_mainland_km": spec["dist_km"],
                "month_sin": float(np.sin(2 * np.pi * month / 12)),
                "month_cos": float(np.cos(2 * np.pi * month / 12)),
            }

            if not past_obs.empty:
                last = past_obs.iloc[-1]
                feat["ndvi_last_observed"] = float(last["mean_ndvi"])
                feat["ndvi_last_gap"] = int(t - last["t"])
            else:
                feat["ndvi_last_observed"] = np.nan
                feat["ndvi_last_gap"] = np.nan

            ly = past_obs[past_obs["t"] == t - 12]
            feat["ndvi_same_month_last_year"] = (
                float(ly.iloc[0]["mean_ndvi"]) if not ly.empty else np.nan
            )

            trail = past_obs[past_obs["t"] >= t - 12]
            if len(trail) >= 2:
                x = trail["t"].to_numpy(dtype=float)
                y = trail["mean_ndvi"].to_numpy(dtype=float)
                feat["ndvi_trend_12m"] = float(np.polyfit(x, y, 1)[0])
                feat["ndvi_volatility_12m"] = float(np.std(y))
                feat["ndvi_mean_12m"] = float(np.mean(y))
            else:
                feat["ndvi_trend_12m"] = np.nan
                feat["ndvi_volatility_12m"] = np.nan
                feat["ndvi_mean_12m"] = np.nan

            obs_trail = obs_sub[(obs_sub["t"] > t - 12) & (obs_sub["t"] <= t)]
            feat["observation_count_12m"] = int(len(obs_trail))
            feat["species_richness_12m"] = int(obs_trail["taxon_name"].nunique())

            rows.append(feat)

    return pd.DataFrame(rows)


def actual_annual_mean(ndvi: pd.DataFrame, year: int) -> dict[str, float]:
    sub = ndvi[(ndvi["year"] == year) & ndvi["mean_ndvi"].notna()]
    return sub.groupby("island")["mean_ndvi"].mean().to_dict()


def per_island_rmse(pred_csv: Path) -> dict[str, float]:
    if not pred_csv.exists():
        return {}
    df = pd.read_csv(pred_csv)
    out = {}
    for island, g in df.groupby("island"):
        if len(g) < 2:
            continue
        out[island] = float(np.sqrt(np.mean((g["actual_ndvi"] - g["predicted_ndvi"]) ** 2)))
    return out


def classify(change_pct: float) -> str:
    if change_pct >= GREEN_THRESHOLD:
        return "green"
    if change_pct >= YELLOW_THRESHOLD:
        return "yellow"
    return "red"


def render_folium(rows: list[dict]) -> None:
    import folium

    # Google Satellite tiles — no API key required.
    m = folium.Map(
        location=[1.29, 103.85],
        zoom_start=12,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
    )

    # Optional hybrid labels overlay (Google Hybrid: satellite + street names)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid",
        name="Labels",
        overlay=True,
        control=False,
        opacity=0.6,
    ).add_to(m)

    colors = {"green": "#2c7a4b", "yellow": "#d4a017", "red": "#c0392b"}

    for r in rows:
        spec = ISLANDS[r["island"]]
        radius = max(6, min(30, spec["area_km2"] ** 0.5 * 8))
        popup_html = (
            f"<b>{r['island']}</b><br>"
            f"2025 predicted: {r['predicted_2025']:.3f}<br>"
            f"2026 predicted: {r['predicted_2026']:.3f}<br>"
            f"Change: <b>{r['change_pct']:+.1f}%</b><br>"
            f"Model RMSE (hold-out): ±{r['rmse']:.3f}<br>"
            f"<i>{r['category'].capitalize()} risk</i>"
        )
        folium.CircleMarker(
            location=[spec["lat"], spec["lon"]],
            radius=radius,
            color="white",
            weight=2,
            fill=True,
            fill_color=colors[r["category"]],
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{r['island']}: {r['change_pct']:+.1f}%",
        ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: rgba(255,255,255,0.94); padding: 12px 16px;
                border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                font-family: sans-serif; font-size: 13px; line-height: 1.6;">
      <b>Predicted 2026 NDVI change</b><br>
      <span style="color:#2c7a4b;">●</span> stable (≥ -3%)<br>
      <span style="color:#d4a017;">●</span> moderate (-3% to -8%)<br>
      <span style="color:#c0392b;">●</span> high risk (&lt; -8%)<br>
      <i style="font-size:11px;">Model RMSE ≈ 0.15 — treat small changes
      as noise.</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))


def render_png(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"green": "#2c7a4b", "yellow": "#d4a017", "red": "#c0392b"}

    for r in rows:
        spec = ISLANDS[r["island"]]
        size = max(80, spec["area_km2"] ** 0.5 * 400)
        ax.scatter(spec["lon"], spec["lat"], s=size, c=colors[r["category"]],
                   edgecolors="black", linewidth=1.2, alpha=0.85, zorder=3)
        ax.annotate(
            f"{r['island']}\n{r['change_pct']:+.1f}%",
            (spec["lon"], spec["lat"]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=10, zorder=4,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Predicted 2026 NDVI change — Singapore micro-islands")
    ax.grid(alpha=0.25)
    ax.set_xlim(103.72, 103.99)
    ax.set_ylim(1.19, 1.44)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=colors["green"], markersize=10,
                   label="stable (≥ -3%)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=colors["yellow"], markersize=10,
                   label="moderate (-3% to -8%)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=colors["red"], markersize=10,
                   label="high risk (< -8%)"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing {MODEL_PATH}. Run train_model.py first.")

    log.info("Loading model and data…")
    with MODEL_PATH.open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    ndvi = pd.read_parquet(NDVI_PATH)
    obs = pd.read_parquet(OBS_PATH)

    # Predict both years with the same model (cancels regression-to-mean)
    log.info("Building forecast features for %d and %d…", COMPARISON_YEAR, FORECAST_YEAR)
    feats_2025 = build_forecast_features(ndvi, obs, COMPARISON_YEAR)
    feats_2026 = build_forecast_features(ndvi, obs, FORECAST_YEAR)
    feats_2025["predicted_ndvi"] = model.predict(feats_2025[FEATURES])
    feats_2026["predicted_ndvi"] = model.predict(feats_2026[FEATURES])

    pred_2025 = feats_2025.groupby("island")["predicted_ndvi"].mean().to_dict()
    pred_2026 = feats_2026.groupby("island")["predicted_ndvi"].mean().to_dict()

    actual_2025 = actual_annual_mean(ndvi, COMPARISON_YEAR)
    rmse_by_island = per_island_rmse(PRED_PATH)

    rows = []
    for island in ISLANDS:
        if island not in pred_2025 or island not in pred_2026:
            log.warning("Skipping %s (missing forecast)", island)
            continue
        p25 = pred_2025[island]
        p26 = pred_2026[island]
        change = (p26 - p25) / p25 * 100
        rows.append({
            "island": island,
            "actual_2025": actual_2025.get(island, float("nan")),
            "predicted_2025": p25,
            "predicted_2026": p26,
            "change_pct": change,
            "category": classify(change),
            "rmse": rmse_by_island.get(island, float("nan")),
        })

    df = pd.DataFrame(rows).sort_values("change_pct").reset_index(drop=True)
    FORECAST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FORECAST_CSV, index=False)

    log.info("")
    log.info("Forecast (change = predicted 2026 vs predicted 2025, same model):")
    log.info("\n%s", df.to_string(index=False))
    log.info("")
    log.info("Wrote %s", FORECAST_CSV)

    log.info("Rendering Folium map…")
    render_folium(rows)
    log.info("  → %s", OUT_HTML)

    log.info("Rendering static PNG…")
    render_png(rows)
    log.info("  → %s", OUT_PNG)

    log.info("")
    log.info("Open docs/index.html in a browser to view the interactive map.")


if __name__ == "__main__":
    main()