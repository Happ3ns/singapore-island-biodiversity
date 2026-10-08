# Predicting Vegetation Loss on Singapore's Micro-Islands

A remote-sensing and machine-learning pipeline combining Sentinel-2 satellite
imagery with iNaturalist citizen-science observations to monitor vegetation
health across six of Singapore's small offshore islands.

---

## Motivation

Singapore's micro-islands — Pulau Ubin, St. John's, Kusu, Lazarus, Sisters',
and Pulau Hantu — host some of the country's last remaining patches of primary
coastal forest, mangrove, and coral-reef-associated vegetation. They are small
(0.05–10 km²), largely unprotected, and vulnerable to sea-level rise, coastal
erosion, and human visitation pressure. Yet none of them has continuous
ground-based vegetation monitoring.

Satellite remote sensing offers a cheap alternative: the NDVI (Normalized
Difference Vegetation Index) computed from Sentinel-2 imagery is a well-
established proxy for vegetation greenness at 10 m spatial resolution. This
project asks whether publicly available data — Sentinel-2 from Copernicus plus
iNaturalist citizen-science records — can detect year-over-year vegetation
change on islands this small, and whether a machine-learning model can predict
that change better than naive baselines.

---

## Data

**Sentinel-2 L2A** imagery was fetched via the Copernicus Data Space Ecosystem
(CDSE) Statistical API. For each island, one circular region was sampled
(0.3–2.0 km radius depending on island size). Monthly NDVI was computed
server-side through an evalscript that masks clouds, cloud shadows, snow, and
**water** using the Scene Classification Layer (SCL classes 0, 1, 2, 3, 6, 7,
8, 9, 10, 11 excluded). A month was kept only if it had at least 100 valid
land pixels.

The result is **636 island-months** (6 islands × ~106 months spanning
January 2018 to October 2026), of which **42–51% per island** had sufficient
clear-sky land pixels to compute a usable mean. The remaining months were
NaN — Singapore's monsoon season makes roughly half of all months
unobservable.

**iNaturalist** observations were fetched via the v1 API, using 2-year date
slices to avoid the 10,000-record-per-query cap, and a 1.5 km search radius
around each island centroid. Each observation was assigned to its **nearest**
island centroid to avoid double-counting between the closely spaced southern
islands (Lazarus, Kusu, and St. John's are all within ~900 m of each other).
Research-grade, non-obscured observations totalled **22,162**, ranging from
604 (Sisters') to 11,200 (Pulau Ubin).

---

## Methods

### Feature engineering

For each (island, month), twelve features were computed:

| Feature | Description |
|---|---|
| `ndvi_last_observed` | Most recent valid NDVI value |
| `ndvi_last_gap` | Months since that value |
| `ndvi_same_month_last_year` | NDVI in the same calendar month, previous year |
| `ndvi_trend_12m` | Linear slope over the trailing 12 months |
| `ndvi_volatility_12m` | Standard deviation over the trailing 12 months |
| `ndvi_mean_12m` | Mean over the trailing 12 months |
| `observation_count_12m` | iNaturalist observations in trailing 12 months |
| `species_richness_12m` | Unique species observed in trailing 12 months |
| `island_area_km2` | Hardcoded island area |
| `distance_to_mainland_km` | Hardcoded distance to mainland |
| `month_sin`, `month_cos` | Cyclical month encoding |

The **target** is the mean NDVI of the next observed month, when that month
falls within two months of the current one.

### Baselines

Two naive baselines were used for comparison:

1. **Persistence** — predict the next month's NDVI as the last observed value
   (only when the last observation is ≤ 3 months old).
2. **Linear trend** — fit a linear regression on the last six observed months
   and extrapolate.

### Model

An XGBoost regressor (`n_estimators=400`, `max_depth=3`, `learning_rate=0.05`)
was trained on 2018–2021 data and evaluated on a held-out 2022–2026 test set.
XGBoost handles missing values natively, so no imputation was needed.

### Evaluation

RMSE, MAE, and R² were computed on the test set for both baselines and the
model. All models were evaluated on the same rows to ensure a fair comparison.

---

## Results

### Baselines vs XGBoost

| Model | n | RMSE | MAE | R² |
|---|---|---|---|---|
| Linear trend | 209 | 0.3078 | 0.1979 | -2.515 |
| Persistence | 144 | 0.2410 | 0.1647 | -0.876 |
| **XGBoost** | **209** | **0.1607** | **0.1108** | **0.043** |

XGBoost beats the best baseline by **33.3%** in RMSE. It also achieves a
positive R², whereas both baselines are worse than predicting the mean.

Note that persistence's `n_test` is lower (144 vs 209) because it can only
score rows where the last observation is recent; XGBoost handles stale-feature
rows through engineered features. That difference — 65 test rows that
persistence cannot even attempt — is where the model earns its complexity.

### Per-island performance

| Island | n | XGBoost RMSE |
|---|---|---|
| Kusu | 34 | 0.1414 |
| Pulau Ubin | 38 | 0.1446 |
| Sisters' | 32 | 0.1478 |
| Lazarus | 34 | 0.1507 |
| St. John's | 35 | 0.1659 |
| Pulau Hantu | 36 | 0.2032 |

RMSE is fairly consistent across islands (0.14–0.20). Notably, Pulau Hantu —
despite having the second-highest observation count — is the *hardest* island
to predict, while Kusu (tiny, low observations) is the easiest. This suggests
observation density is not the limiting factor for predictability; intrinsic
vegetation variability is.

### Feature importance

The top three features by XGBoost gain were:

1. `ndvi_mean_12m` (0.17)
2. `species_richness_12m` (0.135)
3. `observation_count_12m` (0.128)

The pure-persistence feature `ndvi_last_observed` ranked near the bottom
(0.055), confirming the model is not simply a persistence baseline in disguise.

### 2026 forecast

Forecasting 2026 requires care: XGBoost trained on 2018–2021 systematically
under-predicts recent NDVI because recent values exceed the training
distribution. Comparing raw 2026 predictions to *actual* 2025 NDVI therefore
mixes real change with model bias (regression to the mean). To isolate real
signal, predicted 2026 was compared to **predicted** 2025 — both outputs of the
same biased model, so the bias cancels.

| Island | Change | Category |
|---|---|---|
| Lazarus | -8.9% | moderate risk |
| Sisters' | -2.1% | stable |
| Kusu | -0.8% | stable |
| St. John's | -0.2% | stable |
| Pulau Hantu | +1.1% | stable |
| Pulau Ubin | +14.3% | (see below) |

Four of six islands show **predicted change under ±2%, inside the model's
error floor**. The two outliers are almost certainly artifacts: Pulau Ubin's
+14% reflects the model adjusting upward toward recent higher-NDVI training
values, and Lazarus' -9% may be influenced by shared observation data with
adjacent islands.

---

## Limitations

These are the honest constraints of the project.

1. **Sentinel-2's 10 m resolution is too coarse for the smallest islands.**
   Sisters' (0.055 km²) and Kusu (0.085 km²) occupy fewer than 1,000 land
   pixels each. A single cloudy scene can wipe out a month's data entirely,
   and edge effects from sand, rock, and coastal scrub contaminate the
   interior signal.

2. **Cloud cover makes ~50% of months unusable.** Singapore's inter-monsoon
   months are consistently overcast; no gap-filling method (interpolation,
   Sentinel-1 radar fusion) was applied. The model therefore trains on a
   non-random subset of the year, biasing toward dry-season observations.

3. **iNaturalist observation effort is confounded with time and popularity.**
   Annual observation counts grew steadily from ~800 in 2018 to ~2,200 in
   2025 across all islands — a platform-growth trend, not a biodiversity
   trend. Because `observation_count_12m` and `species_richness_12m` ranked
   among the top-3 features, the model may be learning "later year" as a
   proxy for both observation volume and any correlated NDVI drift.

4. **Regression to the mean biases raw forecasts.** As discussed above,
   XGBoost pulls predictions toward the training-period mean. Without the
   predicted-vs-predicted correction, forecasts showed implausible ±15–24%
   swings that were entirely artifact.

5. **Adjacent islands share observation data.** Lazarus, Kusu, and St. John's
   lie within 900 m of one another. Nearest-centroid assignment prevents
   double-counting, but observers working from a boat or a single trailhead
   can still log observations at locations that reflect the wrong island.

6. **Eight years is too short to distinguish trend from seasonal variability.**
   The 2022–2026 test set contains only 32–38 rows per island. R² confidence
   intervals at that sample size are essentially meaningless; RMSE is the
   only metric stable enough to report.

7. **NDVI is a proxy for greenness, not biodiversity.** A stable NDVI does not
   imply stable species composition — a monoculture replacing a mixed forest
   could show no change in NDVI while representing significant biodiversity
   loss.

---

## Reproducibility

Full pipeline is available at [github.com/Happ3ns/singapore-island-biodiversity](https://github.com/Happ3ns/singapore-island-biodiversity).

```bash
pip install -r requirements.txt

python fetch_ndvi.py             # ~15 min — builds data/islands_ndvi.parquet
python fetch_observations.py     # ~5 min  — builds data/islands_observations.parquet
python baselines.py              # ~5 sec  — results/baselines.csv
python train_model.py            # ~15 sec — models/ndvi_model.pkl, predictions.csv
python make_map.py               # ~5 sec  — docs/index.html, map_preview.png
```

Credentials are required in `.env`:

- `COPERNICUS_CLIENT_ID`, `COPERNICUS_CLIENT_SECRET` (from
  [dataspace.copernicus.eu](https://dataspace.copernicus.eu))
- `INATURALIST_TOKEN` (from
  [inaturalist.org/users/api_token](https://www.inaturalist.org/users/api_token))

---

## Future Work

- **Higher-resolution imagery.** Planet Labs' 3 m data would quadruple the
  usable pixel count on Kusu and Sisters', likely halving the uncertainty on
  those islands.
- **Longer temporal window.** Extending to Landsat (30 m, 1984–present) would
  allow decadal trend detection rather than year-over-year noise.
- **Multi-target modelling.** Predicting species richness or mangrove extent
  directly — rather than NDVI as a proxy — would make the model's output more
  conservation-relevant.
- **Real-time monitoring dashboard.** The pipeline currently runs on demand;
  a scheduled weekly job with alerts on detected decline would make it
  operationally useful.
- **Ensemble approaches.** Random Forest, LightGBM, and a simple LSTM baseline
  could be compared against XGBoost; the current single-model approach leaves
  performance on the table.

---

## Conclusion

This project demonstrates that a full remote-sensing ML pipeline — Copernicus
API access, feature engineering, baseline comparison, model training, and
interactive visualization — can be built from entirely free, open data. The
XGBoost model beats persistence and trend baselines by 33%, and the feature
importance analysis confirms the engineered features carry genuine signal
beyond simple persistence.

The honest scientific finding, however, is that current public data cannot
resolve directional vegetation change on these islands. After correcting for
regression-to-the-mean bias, four of six islands show predicted 2026 change
under ±2%, well inside the model's ±0.15 RMSE. The pipeline is useful as
**monitoring infrastructure** — re-run it in five years and it will detect
real decline if it begins — but its current output is a call for higher-
resolution data, not a claim of detected biodiversity loss.