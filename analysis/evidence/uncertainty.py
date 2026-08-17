"""
Uncertainty quantification for the routed forecast, by split conformal prediction.

Why conformal rather than a quantile objective: the routing sends two of the four targets
to a smoother or a model/smoother blend, and a quantile loss only exists for the XGBoost
ones. Conformal wraps whatever the routed forecast happens to be, so all four targets get
intervals built the same way, and its coverage guarantee needs no assumption about the
residual distribution.

A Gaussian +/- z*sigma band on the same calibration set is reported alongside it. On this
data the two are close -- see uncertainty_coverage.csv -- so the choice is defended by the
weaker assumption, not by a visible accuracy gap.

Calibration is strictly backward-looking: the interval for forecast year t uses only
residuals from years < t, the same discipline as the point forecast. That makes the
reported coverage an honest out-of-sample number rather than a fit.

Intervals are built on the modelling scale (log1p for severity) and then back-transformed,
so severity intervals come out asymmetric and cannot go negative.

Inputs:  analysis/s08_rolling_origin_predictions.csv  (from s08_rolling_origin.py)
Outputs: analysis/evidence/uncertainty_coverage.csv
         submission/pest_forecasts_2026_intervals.csv
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "analysis")
from s07_features_and_model import BASE_TARGETS, clip_to_range  # noqa: E402
from s09_final_forecast import (  # noqa: E402
    FORECAST_CSV, FORECAST_YEAR, PERF_CSV, ROUTING,
)

PRED_CSV = "analysis/s08_rolling_origin_predictions.csv"
COVERAGE_CSV = "analysis/evidence/uncertainty_coverage.csv"
INTERVAL_CSV = "submission/pest_forecasts_2026_intervals.csv"
ALPHAS = [0.20, 0.10]          # 80% and 90% nominal intervals
MIN_CAL = 40                   # residuals needed before an interval is issued
Z = {0.20: 1.2815515655, 0.10: 1.6448536270}   # two-sided normal quantiles

SHORT = {
    "Zymoseptoria_tritici_Crop_Incidence": "Septoria inc",
    "Zymoseptoria_tritici_Disease_Severity": "Septoria sev",
    "Yellow_rust_Crop_Incidence": "YR inc",
    "Yellow_rust_Disease_Severity": "YR sev",
}


def is_sev(target):
    return "Severity" in target


def to_model_scale(v, target):
    return np.log1p(v) if is_sev(target) else v


def from_model_scale(v, target):
    return clip_to_range(np.expm1(v) if is_sev(target) else v)


def routed_backtest(preds):
    """Collapse the per-method dump to the single routed forecast, as s09 does."""
    wide = preds.pivot_table(
        index=["forecast_year", "Region", "Leaf", "target", "observed"],
        columns="method", values="pred").reset_index()
    out = []
    for target, (kind, mset, sname, w) in ROUTING.items():
        d = wide[wide.target == target].copy()
        if kind == "model":
            d["pred"] = d[f"model_{mset}"]
        elif kind == "smoother":
            d["pred"] = d[sname]
        elif kind == "blend":
            d["pred"] = w * d[f"model_{mset}"] + (1 - w) * d[sname]
        else:
            raise ValueError(kind)
        out.append(d[["forecast_year", "Region", "Leaf", "target", "observed", "pred"]])
    d = pd.concat(out, ignore_index=True).dropna(subset=["pred", "observed"])
    d["pred"] = clip_to_range(d.pred.to_numpy())
    ### the residual the interval is calibrated on lives on the modelling scale
    d["resid"] = [to_model_scale(o, t) - to_model_scale(p, t)
                  for o, p, t in zip(d.observed, d.pred, d.target)]
    return d


def conformal_q(abs_res, alpha):
    """The (1-alpha) conformal radius: the ceil((n+1)(1-alpha))-th smallest |residual|.

    The +1 is the finite-sample correction that makes marginal coverage >= 1-alpha. When
    the required order statistic exceeds n the calibration set is too small to certify
    that level at all, and the honest answer is an unbounded interval.
    """
    n = len(abs_res)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(abs_res)[k - 1])


def intervals(pred, target, radius):
    """Symmetric on the modelling scale -> asymmetric, non-negative on the raw scale."""
    ms = to_model_scale(pred, target)
    return from_model_scale(ms - radius, target), from_model_scale(ms + radius, target)


if __name__ == "__main__":
    back = routed_backtest(pd.read_csv(PRED_CSV))
    years = sorted(back.forecast_year.unique())
    print(f"routed backtest: {len(back)} row-forecasts, "
          f"{years[0]}-{years[-1]} ({len(years)} years)")

    ### how conservative is calibrating on single-seed residuals? The production forecast
    ### averages 5 seeds, which is slightly tighter, so the intervals inherit a small
    ### amount of extra width. Quantified rather than assumed.
    perf = pd.read_csv(PERF_CSV)
    perf["base_target"] = perf.target.str.slice(3)
    ovl = back[back.forecast_year >= 2016]
    print("\n=== single-seed (calibration) vs 5-seed (production) RMSE, 2016-2025 ===")
    for t in BASE_TARGETS:
        one = float(np.sqrt((ovl[ovl.target == t].eval("(observed - pred) ** 2")).mean()))
        five = float(np.sqrt((perf[perf.base_target == t].rmse ** 2).mean()))
        print(f"  {SHORT[t]:14s} single-seed {one:8.3f}   5-seed {five:8.3f}   "
              f"ratio {one / five:.3f}")

    ### backward-looking coverage: year t is scored with residuals from years < t only
    rows, scored = [], []
    for (target, leaf), g in back.groupby(["target", "Leaf"], observed=True):
        for year in years:
            cal = g[g.forecast_year < year]
            test = g[g.forecast_year == year]
            if test.empty or len(cal) < MIN_CAL:
                continue
            a = cal.resid.abs().to_numpy()
            sd = float(cal.resid.std(ddof=1))
            for alpha in ALPHAS:
                r_conf = conformal_q(a, alpha)
                ### the Gaussian comparator, same calibration set
                for kind, radius in (("conformal", r_conf), ("gaussian", Z[alpha] * sd)):
                    lo, hi = intervals(test.pred.to_numpy(), target, radius)
                    scored.append(pd.DataFrame({
                        "target": target, "Leaf": leaf, "forecast_year": year,
                        "alpha": alpha, "kind": kind, "n_cal": len(cal),
                        "observed": test.observed.to_numpy(),
                        "lo": lo, "hi": hi,
                    }))
    scored = pd.concat(scored, ignore_index=True)
    scored["covered"] = (scored.observed >= scored.lo) & (scored.observed <= scored.hi)
    scored["width"] = scored.hi - scored.lo

    cov = (scored.groupby(["target", "Leaf", "alpha", "kind"], observed=True)
                 .agg(n=("covered", "size"), coverage=("covered", "mean"),
                      mean_width=("width", "mean"),
                      first_year=("forecast_year", "min"))
                 .reset_index())
    cov["nominal"] = 1 - cov.alpha
    cov["coverage"] *= 100
    cov["nominal"] *= 100
    cov.to_csv(COVERAGE_CSV, index=False)

    print(f"\n=== empirical coverage, calibrated on prior years only "
          f"(scored {cov.first_year.min()}-{years[-1]}) ===")
    for alpha in ALPHAS:
        print(f"\n  nominal {100 * (1 - alpha):.0f}%")
        tab = (cov[cov.alpha == alpha]
               .assign(label=lambda d: d.target.map(SHORT) + " " + d.Leaf)
               .pivot(index="label", columns="kind", values=["coverage", "mean_width"]))
        print(tab.round(1).to_string())

    print("\n=== pooled coverage by method ===")
    pooled = (scored.groupby(["alpha", "kind"])
                    .agg(coverage=("covered", "mean"), n=("covered", "size")))
    pooled["coverage"] *= 100
    print(pooled.round(1).to_string())

    ### the 2026 intervals: calibrate on every residual now available, apply to the
    ### production (5-seed) point forecast rather than re-deriving it here
    fc = pd.read_csv(FORECAST_CSV)
    fc["Leaf"] = fc.target.str.slice(0, 2)
    fc["base_target"] = fc.target.str.slice(3)
    out = []
    for (target, leaf), g in back.groupby(["target", "Leaf"], observed=True):
        a = g.resid.abs().to_numpy()
        sel = fc[(fc.base_target == target) & (fc.Leaf == leaf)].copy()
        for alpha in ALPHAS:
            r = conformal_q(a, alpha)
            lo, hi = intervals(sel.forecast_value.to_numpy(), target, r)
            sel[f"lo{100 - int(alpha * 100)}"] = lo
            sel[f"hi{100 - int(alpha * 100)}"] = hi
        sel["n_cal"] = len(a)
        out.append(sel)

    iv = pd.concat(out, ignore_index=True).sort_values(["region", "target"])
    keep = ["region", "target", "year", "forecast_value",
            "lo80", "hi80", "lo90", "hi90", "n_cal"]
    iv[keep].to_csv(INTERVAL_CSV, index=False)
    print(f"\nwrote {INTERVAL_CSV}: {iv.shape[0]} rows")
    print(f"wrote {COVERAGE_CSV}")

    print(f"\n=== {FORECAST_YEAR} forecast with 90% conformal intervals, "
          f"mean over regions ===")
    summ = (iv.assign(label=lambda d: d.base_target.map(SHORT) + " " + d.Leaf)
              .groupby("label")[["lo90", "forecast_value", "hi90"]].mean())
    summ["width"] = summ.hi90 - summ.lo90
    print(summ.round(2).to_string())
