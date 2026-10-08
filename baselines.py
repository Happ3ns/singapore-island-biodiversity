"""
baselines.py — Two naive baselines for next-month NDVI prediction.

1. Persistence:  predict next month = last observed month
2. Linear trend: fit linear regression on last 6 observed months, extrapolate

Train period: 2018–2023.   Test period: 2024–2026.

Output: results/baselines.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

NDVI_PATH = Path("data/islands_ndvi.parquet")
OUT_PATH = Path("results/baselines.csv")

TRAIN_END_YEAR = 2023
TEST_START_YEAR = 2024
TREND_WINDOW = 6        # months used for linear-trend baseline

log = logging.getLogger("baselines")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def run_baselines_for_island(df: pd.DataFrame, island: str) -> dict:
    """Return {baseline_name: (y_true, y_pred)} for one island."""
    sub = df[df["island"] == island].sort_values(["year", "month"]).reset_index(drop=True)
    sub["t"] = sub["year"] * 12 + sub["month"]      # absolute month index

    persistence_true, persistence_pred = [], []
    trend_true, trend_pred = [], []

    for i, row in sub.iterrows():
        if row["year"] < TEST_START_YEAR:
            continue
        if pd.isna(row["mean_ndvi"]):
            continue  # no ground truth to score against

        # ---- Persistence: last observed NDVI before this month ----
        past = sub.iloc[:i]
        past_obs = past[past["mean_ndvi"].notna()]
        if len(past_obs) >= 1:
            last = past_obs.iloc[-1]
            # require the observation be within 3 months, else skip
            if row["t"] - last["t"] <= 3:
                persistence_true.append(row["mean_ndvi"])
                persistence_pred.append(last["mean_ndvi"])

        # ---- Linear trend: fit on last N observed months ----
        if len(past_obs) >= TREND_WINDOW:
            window = past_obs.iloc[-TREND_WINDOW:]
            x = window["t"].to_numpy(dtype=float)
            y = window["mean_ndvi"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * row["t"] + intercept
            trend_true.append(row["mean_ndvi"])
            trend_pred.append(pred)

    return {
        "persistence": (np.array(persistence_true), np.array(persistence_pred)),
        "linear_trend": (np.array(trend_true), np.array(trend_pred)),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    df = pd.read_parquet(NDVI_PATH)
    log.info("Loaded %d rows, %d islands", len(df), df["island"].nunique())

    rows = []
    all_true = {"persistence": [], "linear_trend": []}
    all_pred = {"persistence": [], "linear_trend": []}

    for island in sorted(df["island"].unique()):
        result = run_baselines_for_island(df, island)
        for name in ("persistence", "linear_trend"):
            yt, yp = result[name]
            if len(yt) == 0:
                log.warning("%s / %s: no test rows", island, name)
                continue
            rows.append({
                "island": island,
                "baseline": name,
                "n_test": len(yt),
                "rmse": rmse(yt, yp),
                "r2": r2(yt, yp),
            })
            all_true[name].append(yt)
            all_pred[name].append(yp)

    # Combined row
    for name in ("persistence", "linear_trend"):
        if not all_true[name]:
            continue
        yt = np.concatenate(all_true[name])
        yp = np.concatenate(all_pred[name])
        rows.append({
            "island": "ALL ISLANDS",
            "baseline": name,
            "n_test": len(yt),
            "rmse": rmse(yt, yp),
            "r2": r2(yt, yp),
        })

    out = pd.DataFrame(rows).sort_values(["baseline", "island"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH.with_suffix(".parquet"), index=False)
    out.to_csv(OUT_PATH, index=False)

    log.info("")
    log.info("Results:")
    log.info("\n%s", out.to_string(index=False))
    log.info("")
    log.info("Saved %s and %s", OUT_PATH, OUT_PATH.with_suffix(".parquet"))

    # Sanity check per the roadmap
    for name in ("persistence", "linear_trend"):
        sub = out[(out["island"] == "ALL ISLANDS") & (out["baseline"] == name)]
        if not sub.empty:
            r = sub.iloc[0]["rmse"]
            if r < 0.01:
                log.warning("%s RMSE %.4f < 0.01 — NDVI data suspiciously smooth", name, r)
            elif r > 0.5:
                log.warning("%s RMSE %.4f > 0.5 — something is wrong", name, r)
            else:
                log.info("%s RMSE %.4f — in healthy 0.05–0.2 range", name, r)


if __name__ == "__main__":
    main()