"""
train_model.py — XGBoost for next-month NDVI prediction on Singapore micro-islands.

Train period: 2018–2021.   Test period: 2022–2026.

Features per (island, year, month):
    ndvi_last_observed, ndvi_last_gap,
    ndvi_same_month_last_year,
    ndvi_trend_12m, ndvi_volatility_12m, ndvi_mean_12m,
    observation_count_12m, species_richness_12m,
    island_area_km2, distance_to_mainland_km,
    month_sin, month_cos

Target: mean_ndvi of the next observed month, if that month is within
        MAX_TARGET_GAP_MONTHS months.

Outputs:
    models/ndvi_model.pkl
    results/predictions.csv
    results/feature_importance.png
    results/model_vs_baselines.csv
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
from xgboost import XGBRegressor

NDVI_PATH = Path("data/islands_ndvi.parquet")
OBS_PATH = Path("data/islands_observations.parquet")
MODEL_PATH = Path("models/ndvi_model.pkl")
PRED_PATH = Path("results/predictions.csv")
IMP_PATH = Path("results/feature_importance.png")
COMPARE_PATH = Path("results/model_vs_baselines.csv")

TRAIN_END_YEAR = 2021
TEST_START_YEAR = 2022
MAX_TARGET_GAP_MONTHS = 2

ISLAND_AREA_KM2 = {
    "Pulau Ubin":  10.2,
    "St. John's":  0.41,
    "Kusu":        0.085,
    "Lazarus":     0.18,
    "Sisters":     0.055,
    "Pulau Hantu": 0.13,
}
DIST_TO_MAINLAND_KM = {
    "Pulau Ubin":  5.0,
    "St. John's":  6.5,
    "Kusu":        5.5,
    "Lazarus":     6.0,
    "Sisters":     7.0,
    "Pulau Hantu": 6.0,
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

log = logging.getLogger("train_model")


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

def build_features(ndvi: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    ndvi = ndvi.copy()
    ndvi["t"] = ndvi["year"] * 12 + ndvi["month"]

    obs = obs.copy()
    obs["observed_on"] = pd.to_datetime(obs["observed_on"])
    obs["t"] = obs["observed_on"].dt.year * 12 + obs["observed_on"].dt.month

    rows = []
    for island, sub in ndvi.groupby("island"):
        sub = sub.sort_values("t").reset_index(drop=True)
        obs_sub = obs[obs["island"] == island]

        for i, row in sub.iterrows():
            t = int(row["t"])
            past = sub[sub["t"] < t]
            past_obs = past[past["mean_ndvi"].notna()]

            # --- target: next observed month within MAX_TARGET_GAP_MONTHS ---
            future = sub[(sub["t"] > t) & sub["mean_ndvi"].notna()]
            if future.empty:
                continue
            nxt = future.iloc[0]
            gap = int(nxt["t"] - t)
            if gap > MAX_TARGET_GAP_MONTHS:
                continue

            feat = {
                "island": island,
                "year": int(row["year"]),
                "month": int(row["month"]),
                "t": t,
                "target": float(nxt["mean_ndvi"]),
                "target_gap": gap,
                "island_area_km2": ISLAND_AREA_KM2[island],
                "distance_to_mainland_km": DIST_TO_MAINLAND_KM[island],
                "month_sin": float(np.sin(2 * np.pi * row["month"] / 12)),
                "month_cos": float(np.cos(2 * np.pi * row["month"] / 12)),
            }

            # last observed NDVI and how stale it is
            if not past_obs.empty:
                last = past_obs.iloc[-1]
                feat["ndvi_last_observed"] = float(last["mean_ndvi"])
                feat["ndvi_last_gap"] = int(t - last["t"])
            else:
                feat["ndvi_last_observed"] = np.nan
                feat["ndvi_last_gap"] = np.nan

            # same calendar month, previous year
            ly = past_obs[past_obs["t"] == t - 12]
            feat["ndvi_same_month_last_year"] = (
                float(ly.iloc[0]["mean_ndvi"]) if not ly.empty else np.nan
            )

            # trailing-12-month stats
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

            # iNaturalist features over trailing 12 months
            obs_trail = obs_sub[(obs_sub["t"] > t - 12) & (obs_sub["t"] <= t)]
            feat["observation_count_12m"] = int(len(obs_trail))
            feat["species_richness_12m"] = int(obs_trail["taxon_name"].nunique())

            rows.append(feat)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Baselines (re-computed on the same train/test split)
# --------------------------------------------------------------------------

def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, mae, r2


def baseline_persistence(test_df):
    mask = test_df["ndvi_last_observed"].notna() & (test_df["ndvi_last_gap"] <= 3)
    sub = test_df[mask]
    return sub["target"].to_numpy(), sub["ndvi_last_observed"].to_numpy(), len(sub)


def baseline_trend(test_df):
    mask = (
        test_df["ndvi_trend_12m"].notna()
        & test_df["ndvi_last_observed"].notna()
        & test_df["ndvi_last_gap"].notna()
    )
    sub = test_df[mask].copy()
    # extrapolate last observation forward using the trailing slope
    steps = sub["ndvi_last_gap"] + sub["target_gap"] * 0.5
    pred = sub["ndvi_last_observed"] + sub["ndvi_trend_12m"] * steps
    return sub["target"].to_numpy(), pred.to_numpy(), len(sub)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Loading NDVI and observations…")
    ndvi = pd.read_parquet(NDVI_PATH)
    obs = pd.read_parquet(OBS_PATH)
    log.info("  NDVI rows: %d   Observation rows: %d", len(ndvi), len(obs))

    log.info("Building features…")
    df = build_features(ndvi, obs)
    log.info("  Built %d feature rows (one per island-month with a valid target)",
             len(df))

    train = df[df["year"] <= TRAIN_END_YEAR].reset_index(drop=True)
    test = df[df["year"] >= TEST_START_YEAR].reset_index(drop=True)
    log.info("  Train rows (2018–%d): %d", TRAIN_END_YEAR, len(train))
    log.info("  Test  rows (%d–2026): %d", TEST_START_YEAR, len(test))

    if len(train) < 50 or len(test) < 20:
        log.warning("Very small dataset — results will be noisy. "
                    "That's expected for this project.")

    # ----- baselines on the same split -----
    log.info("")
    log.info("Evaluating baselines on the test split…")
    comparison = []
    for name, fn in (("persistence", baseline_persistence),
                     ("linear_trend", baseline_trend)):
        yt, yp, n = fn(test)
        if n == 0:
            log.warning("  %s: no test rows", name)
            continue
        rmse, mae, r2 = eval_metrics(yt, yp)
        log.info("  %-14s n=%-4d RMSE=%.4f  MAE=%.4f  R²=%.3f",
                 name, n, rmse, mae, r2)
        comparison.append({"model": name, "n_test": n,
                           "rmse": rmse, "mae": mae, "r2": r2})

    # ----- XGBoost -----
    log.info("")
    log.info("Training XGBoost…")
    X_train = train[FEATURES]
    y_train = train["target"]
    X_test = test[FEATURES]
    y_test = test["target"]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    rmse, mae, r2 = eval_metrics(y_test.to_numpy(), pred)
    log.info("  XGBoost        n=%-4d RMSE=%.4f  MAE=%.4f  R²=%.3f",
             len(X_test), rmse, mae, r2)
    comparison.append({"model": "xgboost", "n_test": len(X_test),
                       "rmse": rmse, "mae": mae, "r2": r2})

    # ----- save outputs -----
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump({"model": model, "features": FEATURES}, f)

    preds = test[["island", "year", "month"]].copy()
    preds["actual_ndvi"] = y_test.values
    preds["predicted_ndvi"] = pred
    preds["error"] = preds["predicted_ndvi"] - preds["actual_ndvi"]
    preds = preds.sort_values(["island", "year", "month"]).reset_index(drop=True)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PRED_PATH, index=False)
    log.info("")
    log.info("Wrote %s", PRED_PATH)

    # ----- feature importance chart -----
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot.barh(ax=ax, color="#2c7a4b")
    ax.set_xlabel("Importance (gain)")
    ax.set_title("XGBoost feature importance — next-month NDVI")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMP_PATH, dpi=140)
    plt.close(fig)
    log.info("Wrote %s", IMP_PATH)

    # ----- comparison table -----
    comp_df = pd.DataFrame(comparison).sort_values("rmse").reset_index(drop=True)
    COMPARE_PATH.parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(COMPARE_PATH, index=False)
    log.info("Wrote %s", COMPARE_PATH)

    log.info("")
    log.info("=" * 60)
    log.info("Model vs baselines (same test split):")
    log.info("\n%s", comp_df.to_string(index=False))

    # ----- pass/fail against the roadmap's threshold -----
    base_best = comp_df[comp_df["model"] != "xgboost"]["rmse"].min()
    if base_best and not np.isnan(base_best):
        improvement = (base_best - rmse) / base_best * 100
        log.info("")
        log.info("Best baseline RMSE: %.4f", base_best)
        log.info("XGBoost RMSE:       %.4f", rmse)
        log.info("Improvement:        %.1f%%", improvement)
        if improvement >= 20:
            log.info("PASS — XGBoost beats the best baseline by ≥ 20%%.")
        else:
            log.warning("Below the 20%% target. Check the feature importance "
                        "chart — the model is probably leaning on "
                        "ndvi_last_observed only.")

    # ----- per-island RMSE for the writeup -----
    log.info("")
    log.info("Per-island XGBoost RMSE:")
    for island, g in preds.groupby("island"):
        if len(g) < 2:
            continue
        r = float(np.sqrt(np.mean((g["actual_ndvi"] - g["predicted_ndvi"]) ** 2)))
        log.info("  %-14s n=%-3d RMSE=%.4f", island, len(g), r)


if __name__ == "__main__":
    main()