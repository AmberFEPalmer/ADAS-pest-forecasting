"""
Shrinkage calibration for XGBoost predictions
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "analysis")
from s07_features_and_model import (  # noqa: E402
    BASE_TARGETS, TEST_START, TRAIN_END, build_feature_table, model_rows_from,
    per_target_feature_cols, rmse, run_targets,
)

SEEDS = [0, 1, 2, 3, 4]
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 2)
VAL_TRAIN_END = 2005
VAL_START = 2006
OUT_CSV = "analysis/evidence/shrinkage_calibration_results.csv"


def transforms(target):
    is_sev = "Severity" in target
    tf = (lambda v: np.log1p(v)) if is_sev else (lambda v: v)
    inv = (lambda v: np.clip(np.expm1(v), 0, None)) if is_sev else (lambda v: v)
    return tf, inv


def smoothers(rows):
    out = {}
    d = rows.sort_values(["Region", "Leaf", "Year"])
    for t in BASE_TARGETS:
        tf, inv = transforms(t)
        s = d.assign(tv=tf(d[t])).groupby(["Region", "Leaf"], observed=True)["tv"]
        out[(t, "roll_k8")] = inv(
            s.transform(lambda x: x.shift(1).rolling(8, min_periods=2).mean())).reindex(rows.index)
        out[(t, "ewma_a30")] = inv(
            s.transform(lambda x: x.shift(1).ewm(alpha=0.30, min_periods=2).mean())).reindex(rows.index)
    return out


def model_predictions(rows, feature_cols, train_end, test_start):
    """Seed-averaged model predictions on the scored window"""
    acc = {}
    for seed in SEEDS:
        _res, _imp, preds, _folds, test = run_targets(
            rows, feature_cols, seed=seed, train_end=train_end, test_start=test_start)
        for t in BASE_TARGETS:
            acc.setdefault(t, []).append(
                pd.Series(preds[f"{t}_pred"].to_numpy(), index=test.index))
    return {t: pd.concat(v, axis=1).mean(axis=1) for t, v in acc.items()}


def score_weights(rows, feature_cols, train_end, test_start, smooth):
    """RMSE for every (target, smoother, w) on the scored window."""
    model = model_predictions(rows, feature_cols, train_end, test_start)
    scored = rows[rows.Year >= test_start]
    out = {}
    for t in BASE_TARGETS:
        m = model[t].reindex(scored.index)
        for sname in ("roll_k8", "ewma_a30"):
            b = smooth[(t, sname)].reindex(scored.index)
            # Rows with no smoother history yet fall back to the model, so a
            # baseline can never look good by quietly skipping rows.
            b = b.fillna(m)
            ok = m.notna() & b.notna()
            for w in WEIGHTS:
                blend = b[ok] + w * (m[ok] - b[ok])
                out[(t, sname, float(w))] = rmse(blend, scored[t][ok])
    return out


def select_w_rolling_origin(rows, feature_cols, smooth, last_forecast_year=TRAIN_END):
    from s08_rolling_origin import FIRST_ORIGIN, fit_origin  # noqa: E402 (heavy import, only needed here)

    sse, n_obs = {}, {}
    for origin in range(FIRST_ORIGIN, last_forecast_year):
        fut = rows[rows.Year == origin + 1]
        if fut.empty:
            continue
        preds = fit_origin(rows, feature_cols, origin)
        if preds is None:
            continue
        for t in BASE_TARGETS:
            actual = fut[t].to_numpy(dtype=float)
            m = preds[t].reindex(fut.index).to_numpy(dtype=float)
            for sname in ("roll_k8", "ewma_a30"):
                b = smooth[(t, sname)].reindex(fut.index).to_numpy(dtype=float)
                b = np.where(np.isnan(b), m, b)
                ok = ~np.isnan(m) & ~np.isnan(b) & ~np.isnan(actual)
                if not ok.sum():
                    continue
                for w in WEIGHTS:
                    blend = b[ok] + w * (m[ok] - b[ok])
                    key = (t, sname, float(w))
                    sse[key] = sse.get(key, 0.0) + float(((blend - actual[ok]) ** 2).sum())
                    n_obs[key] = n_obs.get(key, 0) + int(ok.sum())
        print(f"  origin {origin} -> {origin+1} done", flush=True)

    return {k: np.sqrt(v / n_obs[k]) for k, v in sse.items()}


if __name__ == "__main__":
    table = build_feature_table(spatial=True)
    model_rows = model_rows_from(table)
    feature_cols = per_target_feature_cols(model_rows)

    val_rows = model_rows[model_rows.Year <= TRAIN_END].copy()
    print(f"SELECT: fit <= {VAL_TRAIN_END}, choose w on {VAL_START}-{TRAIN_END}; 2016+ removed")
    val = score_weights(val_rows, feature_cols, VAL_TRAIN_END, VAL_START, smoothers(val_rows))

    chosen = {}
    for t in BASE_TARGETS:
        cand = {(s, w): r for (tt, s, w), r in val.items() if tt == t}
        chosen[t] = min(cand, key=cand.get)

    print("\n=== w chosen on validation (0 = pure smoother, 1 = pure model) ===")
    for t, (s, w) in chosen.items():
        print(f"  {t:40s} {s:9s} w={w:.1f}   val RMSE {val[(t, s, w)]:.3f}")

    print(f"\nSELECT-RO: pooling forecast years up to {TRAIN_END} across rolling origins")
    ro = select_w_rolling_origin(model_rows, feature_cols, smoothers(model_rows))
    chosen_ro = {}
    for t in BASE_TARGETS:
        cand = {(s, w): r for (tt, s, w), r in ro.items() if tt == t}
        chosen_ro[t] = min(cand, key=cand.get)
    print("\n=== w chosen on rolling origins ===")
    for t, (s, w) in chosen_ro.items():
        print(f"  {t:40s} {s:9s} w={w:.1f}   pooled RMSE {ro[(t, s, w)]:.3f}")

    print(f"\nCONFIRM: refit <= {TRAIN_END}, score {TEST_START}+ (once)")
    test = score_weights(model_rows, feature_cols, TRAIN_END, TEST_START, smoothers(model_rows))

    rows_out = []
    for t in BASE_TARGETS:
        s, w = chosen[t]
        cand = {(ss, ww): r for (tt, ss, ww), r in test.items() if tt == t}
        o_s, o_w = min(cand, key=cand.get)
        s_ro, w_ro = chosen_ro[t]
        rows_out.append({
            "target": t, "smoother": s, "w_selected": w,
            "test_rmse": test[(t, s, w)],
            "smoother_ro": s_ro, "w_selected_ro": w_ro,
            "test_rmse_ro": test[(t, s_ro, w_ro)],
            "model_only_rmse": test[(t, s, 1.0)],
            "smoother_only_rmse": test[(t, s, 0.0)],
            "fixed_blend_0.5_rmse": test[(t, s, 0.5)],
            "oracle_w": o_w, "oracle_smoother": o_s, "oracle_rmse": test[(o_s, o_w)]
            if (o_s, o_w) in test else cand[(o_s, o_w)],
        })

    out = pd.DataFrame(rows_out).set_index("target")
    out["vs_model_%"] = 100 * (out.test_rmse / out.model_only_rmse - 1)
    out["ro_vs_model_%"] = 100 * (out.test_rmse_ro / out.model_only_rmse - 1)
    out["vs_fixed_blend_%"] = 100 * (out.test_rmse / out["fixed_blend_0.5_rmse"] - 1)
    out["cost_of_honesty_%"] = 100 * (out.test_rmse / out.oracle_rmse - 1)
    out.to_csv(OUT_CSV)

    print("\n=== test RMSE (2016+) ===")
    print(out.round(3).to_string())
    print("\nvs_model_%       : negative = tuned shrinkage beats the current model")
    print("vs_fixed_blend_% : negative = tuning w beats the 0.5 blend s08_rolling_origin.py already had")
    print("cost_of_honesty_%: how much worse than choosing w on the test split")

    print("\n=== full test-RMSE curve over w (roll_k8), to see how flat the optimum is ===")
    curve = pd.DataFrame(
        {t: [test[(t, "roll_k8", float(w))] for w in WEIGHTS] for t in BASE_TARGETS},
        index=[f"w={w:.1f}" for w in WEIGHTS])
    print(curve.round(3).to_string())
