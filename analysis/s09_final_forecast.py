"""
The forecast script. Produces submission/pest_forecasts_2026.csv and
submission/pest_model_performance.csv in the contest's required format.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "analysis")
from s08_rolling_origin import (  
    CAT, SEED, fit_origin, smoother_predictions, transforms,
)
from s07_features_and_model import (
    BASE_TARGETS, LEAVES, build_feature_table, clip_to_range,
    mechanistic_feature_cols, per_target_feature_cols, rmse,
)

FORECAST_YEAR = 2026
DUMMY = -9999
FORECAST_CSV = "submission/pest_forecasts_2026.csv"
PERF_CSV = "submission/pest_model_performance.csv"
PERF_ORIGINS = range(2015, 2025)   # forecast years 2016-2025, for the performance table

### seed averaging
SEEDS = [0, 1, 2, 3, 4]

ROUTING = {
    "Zymoseptoria_tritici_Disease_Severity": ("model", "all", None, None),
    "Zymoseptoria_tritici_Crop_Incidence":   ("model", "all", None, None),
    "Yellow_rust_Disease_Severity":          ("smoother", None, "ewma_a30", None),
    "Yellow_rust_Crop_Incidence":            ("blend", "mech", "roll_k5", 0.5),
}

#### prepare the rows for ML algorithms, with NaN for missing targets in the training years
def prepare_rows():
    table = build_feature_table(spatial=True)
    rows = table[table.Region != "Scotland"].copy()
    for t in BASE_TARGETS:
        rows.loc[rows[t] <= DUMMY, t] = np.nan
    return rows.sort_values(["Region", "Leaf", "Year"])

### compute the model forecasts for a given origin year, for each target and feature set
def model_forecasts(rows, origin, feature_sets):
    out = {}
    for name, cols in feature_sets.items():
        runs = [fit_origin(rows, cols, origin, seed=s) for s in SEEDS]
        if any(r is None for r in runs):
            out[name] = None
            continue
        out[name] = {t: pd.concat([r[t] for r in runs], axis=1).mean(axis=1)
                     for t in BASE_TARGETS}
    return out

### compute the routed forecast for a given origin year, for each target and feature set
def route(rows, origin, feature_sets, smooth):
    future = rows[rows.Year == origin + 1]
    if future.empty:
        return None
    models = model_forecasts(rows, origin, feature_sets)
    if any(m is None for m in models.values()):
        return None

    out = []
    for target, (kind, mset, sname, w) in ROUTING.items():
        if kind == "model":
            pred = models[mset][target].reindex(future.index)
        elif kind == "smoother":
            pred = smooth[(target, sname)].reindex(future.index)
        elif kind == "blend":
            pred = w * models[mset][target].reindex(future.index) \
                 + (1 - w) * smooth[(target, sname)].reindex(future.index)
        else:
            raise ValueError(kind)

        ### Every target is a percentage. Severity is clipped at 0 already by its expm1
        ### inverse, but incidence is modelled untransformed, so both the raw model and
        ### the model/smoother blend can land outside [0, 100] -- 2026 produced two
        ### negative yellow rust incidences before this. Clipping to the physically
        ### possible range can only reduce RMSE, since every actual value is inside it.
        pred = pd.Series(clip_to_range(pred.to_numpy()), index=pred.index)

        for idx, p in pred.items():
            r = future.loc[idx]
            out.append({"Region": r.Region, "Leaf": r.Leaf, "year": int(r.Year),
                        "base_target": target, "forecast_value": float(p),
                        "actual": r[target]})
    return pd.DataFrame(out)

### convert the forecast DataFrame into the submission format
def to_submission(df):
    d = df.copy()
    d["target"] = d.Leaf.astype(str) + "_" + d.base_target
    return d[["Region", "target", "year", "forecast_value"]].rename(
        columns={"Region": "region"}).sort_values(["region", "target"]).reset_index(drop=True)

if __name__ == "__main__":
    quick = "--quick" in sys.argv

    rows = prepare_rows()
    feature_sets = {"all": per_target_feature_cols(rows), "mech": mechanistic_feature_cols(rows)}
    smooth = smoother_predictions(rows)

    n_future = len(rows[rows.Year == FORECAST_YEAR])
    print(f"rows {len(rows)} (years {rows.Year.min()}-{rows.Year.max()}), "
          f"{FORECAST_YEAR} rows to forecast: {n_future} "
          f"({rows[rows.Year == FORECAST_YEAR].Region.nunique()} regions x {len(LEAVES)} leaves)")
    assert n_future == rows[rows.Year == FORECAST_YEAR].Region.nunique() * len(LEAVES), \
        "missing Region x Leaf combinations for the forecast year"

    ### forecast
    fc = route(rows, FORECAST_YEAR - 1, feature_sets, smooth)
    assert fc is not None, f"no {FORECAST_YEAR} forecast produced"
    assert fc.forecast_value.notna().all(), \
        f"NaN forecasts: {fc[fc.forecast_value.isna()][['Region','Leaf','base_target']].to_dict('records')}"

    fc2 = route(rows, FORECAST_YEAR - 1, feature_sets, smooth)
    same = np.allclose(fc.forecast_value, fc2.forecast_value, rtol=0, atol=0)
    print(f"determinism check (same seed, re-run): {'IDENTICAL' if same else '*** DIFFERS ***'}")
    assert same, "forecast is not reproducible -- rule 7 requires a fixed seed"

    sub = to_submission(fc)
    sub.to_csv(FORECAST_CSV, index=False)
    print(f"\nWrote {FORECAST_CSV}: {sub.shape} ({sub.target.nunique()} targets x {sub.region.nunique()} regions)")

    print(f"\n=== {FORECAST_YEAR} forecast, mean over regions ===")
    summ = fc.groupby(["base_target", "Leaf"], observed=True).forecast_value.agg(["mean", "min", "max"])
    print(summ.round(2).to_string())

    ### performance table
    if quick:
        print("\n--quick: skipped the performance backtest")
        sys.exit(0)

    print(f"\nBacktesting the routed forecast over origins {PERF_ORIGINS.start}-{PERF_ORIGINS.stop - 1} "
          f"(forecast years {PERF_ORIGINS.start + 1}-{PERF_ORIGINS.stop})")
    back = []
    for origin in PERF_ORIGINS:
        got = route(rows, origin, feature_sets, smooth)
        if got is None:
            print(f"  origin {origin}: skipped (no {origin+1} data)")
            continue
        back.append(got)
        print(f"  origin {origin} -> {origin+1} done", flush=True)

    hist = pd.concat(back, ignore_index=True).dropna(subset=["actual"])
    hist["target"] = hist.Leaf.astype(str) + "_" + hist.base_target
    perf = (hist.groupby(["Region", "target"])
                .apply(lambda g: rmse(g.forecast_value, g.actual), include_groups=False)
                .reset_index(name="rmse").rename(columns={"Region": "region"}))
    perf.to_csv(PERF_CSV, index=False)
    print(f"\nWrote {PERF_CSV}: {perf.shape}")

    print("\n=== backtest RMSE by target (pooled over regions and years) ===")
    pooled = (hist.groupby("target")
                  .apply(lambda g: rmse(g.forecast_value, g.actual), include_groups=False)
                  .rename("rmse"))
    naive = hist.groupby("target").actual.transform("mean")
    print(pooled.round(3).to_string())
    print(f"\n(pooled over {hist.year.nunique()} forecast years, {len(hist)} row-forecasts)")
