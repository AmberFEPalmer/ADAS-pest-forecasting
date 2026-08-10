"""
Tuning and backtesting for XGBoost models.
- seeds
- feature sets
- smoothing methods
- blending weights
- pooled RMSE on the scored window
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "analysis")
from rolling_origin import (  # noqa: E402
    FIRST_ORIGIN, LAST_ORIGIN, fit_origin, smoother_predictions,
)
from xgboost_model import (  # noqa: E402
    BASE_TARGETS, build_feature_table, mechanistic_feature_cols, model_rows_from,
    per_target_feature_cols, rmse,
)

SEEDS = [0, 1, 2, 3, 4]
SMOOTHERS = ["roll_k5", "roll_k8", "ewma_a30", "ewma_a50"]
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 2)
PRED_CSV = "analysis/tuning_backtest_predictions.csv"
OUT_CSV = "analysis/tuning_backtest_results.csv"


def pooled(pred, actual):
    """Pooled RMSE = sqrt(total SSE / total rows). NOT a mean of per-year RMSEs, which
    would weight a quiet year the same as an epidemic one."""
    ok = np.isfinite(pred) & np.isfinite(actual)
    return rmse(np.asarray(pred)[ok], np.asarray(actual)[ok])


def collect(rows, feature_sets, smooth):
    """One pass over the origins, saving every prediction we might want to recombine."""
    frames = []
    for origin in range(FIRST_ORIGIN, LAST_ORIGIN + 1):
        fut = rows[rows.Year == origin + 1]
        if fut.empty:
            continue
        preds = {}
        for name, cols in feature_sets.items():
            for seed in SEEDS:
                p = fit_origin(rows, cols, origin, seed=seed)
                if p is None:
                    preds = None
                    break
                preds[(name, seed)] = p
            if preds is None:
                break
        if not preds:
            continue

        for t in BASE_TARGETS:
            d = pd.DataFrame({
                "forecast_year": origin + 1,
                "Region": fut.Region.astype(str).to_numpy(),
                "Leaf": fut.Leaf.astype(str).to_numpy(),
                "target": t,
                "actual": fut[t].to_numpy(dtype=float),
            })
            for (name, seed), p in preds.items():
                d[f"model_{name}_s{seed}"] = p[t].reindex(fut.index).to_numpy(dtype=float)
            for sm in SMOOTHERS:
                d[sm] = smooth[(t, sm)].reindex(fut.index).to_numpy(dtype=float)
            frames.append(d)
        print(f"  origin {origin} -> {origin+1} done", flush=True)
    return pd.concat(frames, ignore_index=True)


def add_seed_averages(d):
    for name in ("all", "mech"):
        cols = [f"model_{name}_s{s}" for s in SEEDS]
        d[f"model_{name}_avg"] = d[cols].mean(axis=1)
    return d


def blend_grid(d, target, subset=None):
    """Pooled RMSE for every (model set, smoother, weight) on one target."""
    g = d[d.target == target]
    if subset is not None:
        g = g[subset.reindex(g.index).fillna(False)]
    out = []
    for mset in ("mech", "all"):
        for sm in SMOOTHERS:
            for w in WEIGHTS:
                pred = (1 - w) * g[sm] + w * g[f"model_{mset}_avg"]
                out.append({"model": mset, "smoother": sm, "w": float(w),
                            "rmse": pooled(pred, g.actual)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    rows = model_rows_from(build_feature_table(spatial=True))
    feature_sets = {"all": per_target_feature_cols(rows), "mech": mechanistic_feature_cols(rows)}
    smooth = smoother_predictions(rows)

    print(f"origins {FIRST_ORIGIN}-{LAST_ORIGIN}, {len(SEEDS)} seeds x {len(feature_sets)} feature sets")
    d = add_seed_averages(collect(rows, feature_sets, smooth))
    d.to_csv(PRED_CSV, index=False)
    print(f"\nWrote {PRED_CSV}: {d.shape}, {d.forecast_year.nunique()} forecast years")

    print("\n=== 1. SEED AVERAGING: pooled RMSE, single seed 0 vs mean of 5 seeds ===")
    rec = []
    for t in BASE_TARGETS:
        g = d[d.target == t]
        for name in ("all", "mech"):
            s0 = pooled(g[f"model_{name}_s0"], g.actual)
            spread = [pooled(g[f"model_{name}_s{s}"], g.actual) for s in SEEDS]
            avg = pooled(g[f"model_{name}_avg"], g.actual)
            rec.append({"target": t, "model": name, "seed0": s0,
                        "seed_mean_rmse": float(np.mean(spread)), "seed_sd": float(np.std(spread, ddof=1)),
                        "averaged": avg, "gain_%": 100 * (avg / s0 - 1)})
    seed_tbl = pd.DataFrame(rec)
    print(seed_tbl.round(3).to_string(index=False))
    print("\ngain_% : negative = averaging beats the single seed 0 that is currently shipped")
    print("Compare gain against seed_sd -- if |gain| is small relative to the spread across")
    print("individual seeds, averaging is removing exactly that noise, which is the point.")

    T = "Yellow_rust_Crop_Incidence"
    print(f"\n=== 2. BLEND GRID for {T} (seed-averaged models) ===")
    grid = blend_grid(d, T)
    grid.to_csv(OUT_CSV, index=False)
    best = grid.loc[grid.rmse.idxmin()]
    shipped = grid[(grid.model == "mech") & (grid.smoother == "roll_k8") & (grid.w == 0.5)].iloc[0]
    print(grid.nsmallest(8, "rmse").round(4).to_string(index=False))
    print(f"\ncurrently shipped (mech / roll_k8 / w=0.5): {shipped.rmse:.4f}")
    print(f"grid best ({best.model} / {best.smoother} / w={best.w}): {best.rmse:.4f} "
          f"({100 * (best.rmse / shipped.rmse - 1):+.1f}%)")

    years = sorted(d.forecast_year.unique())
    mid = years[len(years) // 2]
    early, late = d.forecast_year < mid, d.forecast_year >= mid
    print(f"\n=== stability check: is the winner the same on both halves? "
          f"(early <{mid}, late >={mid}) ===")
    for label, mask in (("early", early), ("late", late)):
        gh = blend_grid(d, T, subset=mask)
        b = gh.loc[gh.rmse.idxmin()]
        sh = gh[(gh.model == "mech") & (gh.smoother == "roll_k8") & (gh.w == 0.5)].iloc[0]
        print(f"  {label:5s} best: {b.model:4s} / {b.smoother:8s} / w={b.w:.1f}  {b.rmse:8.4f}"
              f"   | shipped there {sh.rmse:8.4f}"
              f"   | full-grid winner there "
              f"{gh[(gh.model==best.model)&(gh.smoother==best.smoother)&(gh.w==best.w)].iloc[0].rmse:8.4f}")
    print("\nIf the full-grid winner is also strong on BOTH halves it is worth adopting;")
    print("if it only wins on one, it is a window artefact and the shipped blend stands.")
