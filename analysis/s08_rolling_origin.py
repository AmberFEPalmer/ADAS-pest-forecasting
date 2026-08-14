"""
Rolling-origin (walk-forward) evaluation -- the harness every earlier comparison needed.

Here each origin year O is trained on everything <= O and forecasts O+1, walking forward
across ~30 origins. That turns one 9-year verdict into ~30 one-step-ahead forecast years
"""

import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, "analysis")
from s07_features_and_model import (  # noqa: E402
    BASE_TARGETS, build_feature_table, mechanistic_feature_cols, model_rows_from,
    per_target_feature_cols, rmse,
)

FIRST_ORIGIN = 1995        # >= 20 years of history before the first forecast
LAST_ORIGIN = 2024         # forecasts 2025, the last observed year
EARLY_STOP_YEARS = 4
SEED = 0
CAT = ["Region", "Leaf"]


def transforms(target):
    is_sev = "Severity" in target
    tf = (lambda v: np.log1p(v)) if is_sev else (lambda v: v)
    inv = (lambda v: np.clip(np.expm1(v), 0, None)) if is_sev else (lambda v: v)
    return tf, inv


def smoother_predictions(rows):
    """All feature-free forecasts, computed once. Every one is strictly backward-looking
    (shift(1) before any window), so slicing them by origin needs no extra care."""
    out = {}
    d = rows.sort_values(["Region", "Leaf", "Year"])
    for t in BASE_TARGETS:
        tf, inv = transforms(t)
        s = d.assign(tv=tf(d[t])).groupby(["Region", "Leaf"], observed=True)["tv"]
        for k in (5, 8):
            out[(t, f"roll_k{k}")] = inv(
                s.transform(lambda x: x.shift(1).rolling(k, min_periods=2).mean())
            ).reindex(rows.index)
        for a in (0.30, 0.50):
            out[(t, f"ewma_a{int(a*100)}")] = inv(
                s.transform(lambda x: x.shift(1).ewm(alpha=a, min_periods=2).mean())
            ).reindex(rows.index)
        out[(t, "persistence")] = d[t].groupby(
            [d.Region, d.Leaf], observed=True).shift(1).reindex(rows.index)
    return out


def fit_origin(rows, cols_by_target, origin, seed=SEED):
    """Train on Year <= origin, forecast Year == origin + 1."""
    train = rows[rows.Year <= origin]
    future = rows[rows.Year == origin + 1]
    if future.empty or train.Year.nunique() < 15:
        return None

    preds = {}
    for t in BASE_TARGETS:
        tf, inv = transforms(t)
        cols = CAT + cols_by_target[t]
        cut = sorted(train.Year.unique())[-EARLY_STOP_YEARS]
        fit_part, val_part = train[train.Year < cut], train[train.Year >= cut]
        if val_part.empty or fit_part.empty:
            return None

        common = dict(max_depth=3, learning_rate=0.03, subsample=0.8,
                      colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0.5,
                      min_child_weight=5, enable_categorical=True,
                      tree_method="hist", random_state=seed)
        es = xgb.XGBRegressor(n_estimators=600, early_stopping_rounds=40,
                              eval_metric="rmse", **common)
        es.fit(fit_part[cols], tf(fit_part[t]),
               eval_set=[(val_part[cols], tf(val_part[t]))], verbose=False)

        final = xgb.XGBRegressor(n_estimators=int(es.best_iteration) + 1, **common)
        final.fit(train[cols], tf(train[t]))
        preds[t] = pd.Series(inv(final.predict(future[cols])), index=future.index)
    return preds


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    table = build_feature_table(spatial=True)
    rows = model_rows_from(table)

    origins = range(2018 if quick else FIRST_ORIGIN, LAST_ORIGIN + 1)
    feature_sets = {"model_all": per_target_feature_cols(rows),
                    "model_mech": mechanistic_feature_cols(rows)}
    smooth = smoother_predictions(rows)

    print(f"origins {min(origins)}-{max(origins)} "
          f"(forecasting {min(origins)+1}-{max(origins)+1}), seed={SEED}")

    recs = []
    for origin in origins:
        fut = rows[rows.Year == origin + 1]
        if fut.empty:
            continue
        model_preds = {n: fit_origin(rows, fs, origin) for n, fs in feature_sets.items()}
        for t in BASE_TARGETS:
            actual = fut[t].to_numpy(dtype=float)
            got = {}
            for name in ("persistence", "roll_k5", "roll_k8", "ewma_a30", "ewma_a50"):
                got[name] = smooth[(t, name)].reindex(fut.index).to_numpy(dtype=float)
            for name, mp in model_preds.items():
                if mp is not None:
                    got[name] = mp[t].reindex(fut.index).to_numpy(dtype=float)
            if "model_mech" in got:
                got["blend_mech_roll8"] = 0.5 * got["model_mech"] + 0.5 * got["roll_k8"]
            if "model_all" in got:
                got["blend_all_roll8"] = 0.5 * got["model_all"] + 0.5 * got["roll_k8"]
            for name, p in got.items():
                ok = ~np.isnan(p) & ~np.isnan(actual)
                if ok.sum():
                    recs.append({"origin": origin, "forecast_year": origin + 1,
                                 "target": t, "method": name, "n": int(ok.sum()),
                                 "rmse": rmse(p[ok], actual[ok])})
        print(f"  origin {origin} -> {origin+1} done", flush=True)

    d = pd.DataFrame(recs)
    d.to_csv("analysis/s08_rolling_origin_results.csv", index=False)

    # Pooled RMSE across all forecast years (weighted by rows, not a mean of RMSEs).
    d["sse"] = d.rmse ** 2 * d.n
    pooled = (d.groupby(["target", "method"])
                .apply(lambda g: np.sqrt(g.sse.sum() / g.n.sum()), include_groups=False)
                .unstack())
    print(f"\n=== pooled RMSE over {d.forecast_year.nunique()} forecast years ===")
    print(pooled.round(3).to_string())

    print("\n=== win rate: fraction of forecast YEARS each method is best ===")
    best = d.loc[d.groupby(["target", "forecast_year"]).rmse.idxmin()]
    print((best.groupby(["target", "method"]).size()
             .unstack(fill_value=0)
             .pipe(lambda x: (x.T / x.sum(axis=1)).T * 100).round(0).to_string()))

    print("\n=== paired vs roll_k8, per forecast year (negative = method is better) ===")
    piv = d.pivot_table(index=["target", "forecast_year"], columns="method", values="rmse")
    for m in ("model_all", "model_mech", "ewma_a30", "blend_mech_roll8", "blend_all_roll8"):
        if m not in piv:
            continue
        diff = (piv[m] - piv["roll_k8"]).groupby("target")
        s = diff.agg(["mean", "std", lambda x: (x < 0).mean() * 100])
        s.columns = ["mean_diff", "sd", "win_%"]
        print(f"\n{m} vs roll_k8:")
        print(s.round(3).to_string())
