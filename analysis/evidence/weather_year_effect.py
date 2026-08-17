"""
Does weather actually predict the year effect?

The variance decomposition says the shared year effect is 47-58% of total variance for
three of the four targets, and the region-by-year term another 18-58%. The whole
mechanistic-feature argument -- thermal time, wetness hours, splash risk, growth-stage
windows -- is a bet that weather explains those components. Nothing in the pipeline tests
that directly: SHAP says the model *uses* weather, which is not the same claim, because a
feature can absorb attribution while carrying no out-of-sample signal.

Two tests, both leave-one-year-out so nothing is scored on a year that helped fit it:

  A. national year effect (n=29) ~ national weather
  B. region-by-year anomaly (n=232) ~ regional weather anomaly, both double-demeaned so
     the region and year main effects are removed from each side

Feature selection happens INSIDE each fold (forward selection, at most K_MAX terms), so
the reported R2 pays for the search. With ~77 candidates and 29 years, an unguarded
in-sample R2 of 0.2 is what noise alone produces -- so the honest LOO R2 is reported next
to the in-sample best, and a permutation null gives a p-value for the LOO number.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "analysis")
from s07_features_and_model import BASE_TARGETS, melt_pest_long  # noqa: E402

DUMMY = -9999
WINDOW = (1996, 2025)
K_MAX = 3            # at most three weather terms; n=29 cannot support more
N_PERM = 200
SEED = 0

SHORT = {
    "Zymoseptoria_tritici_Crop_Incidence": "Septoria inc",
    "Zymoseptoria_tritici_Disease_Severity": "Septoria sev",
    "Yellow_rust_Crop_Incidence": "YR inc",
    "Yellow_rust_Disease_Severity": "YR sev",
}


def load_weather():
    w = pd.read_csv("analysis/growing_season_weather.csv")
    t = pd.read_csv("analysis/thermal_wetness_splash.csv")
    g = pd.read_csv("analysis/growth_stage.csv")
    d = w.merge(t, on=["Region", "Year"]).merge(g, on=["Region", "Year"])
    feats = [c for c in d.columns if c not in ("Region", "Year")]
    return d, [c for c in feats if pd.api.types.is_numeric_dtype(d[c])]


def load_disease():
    d = melt_pest_long(pd.read_csv("data/pest_data.csv"))
    d = d[(d.Region != "Scotland") & d.Year.between(*WINDOW)].copy()
    for t in BASE_TARGETS:
        d.loc[d[t] <= DUMMY, t] = np.nan
    d = d.dropna(subset=BASE_TARGETS)
    ### keep only regions surveyed in every year, so the panel is balanced and the
    ### demeaning below is not distorted by which regions happen to be present
    years = sorted(d.Year.unique())
    per_region = d.groupby("Region").Year.nunique()
    keep = sorted(per_region[per_region == len(years)].index)
    return d[d.Region.isin(keep)], keep, years


def ols_predict(X_tr, y_tr, X_te):
    """Least squares with an intercept; returns predictions for X_te."""
    A = np.column_stack([np.ones(len(X_tr)), X_tr])
    beta, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    return np.column_stack([np.ones(len(X_te)), X_te]) @ beta


def forward_select(X, y, k_max):
    """Greedy forward selection on training data only, scored by in-fold SSE."""
    chosen = []
    remaining = list(range(X.shape[1]))
    best_sse = float(((y - y.mean()) ** 2).sum())
    while len(chosen) < k_max and remaining:
        scores = []
        for j in remaining:
            cols = chosen + [j]
            resid = y - ols_predict(X[:, cols], y, X[:, cols])
            scores.append((float((resid ** 2).sum()), j))
        sse, j = min(scores)
        if sse >= best_sse:      # no in-fold improvement, stop early
            break
        best_sse, chosen = sse, chosen + [j]
        remaining.remove(j)
    return chosen


def loo_r2(X, y, groups, k_max=K_MAX):
    """Leave-one-group-out R2 against the training-fold mean.

    groups is the year of each row, so a whole year leaves at once -- otherwise rows from
    the same year would train on each other and the shared year effect would leak.
    """
    sse_model = sse_base = 0.0
    for gr in np.unique(groups):
        te = groups == gr
        tr = ~te
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        ### standardise on the training fold only
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
        cols = forward_select(Xtr_s, ytr, k_max)
        pred = (ols_predict(Xtr_s[:, cols], ytr, Xte_s[:, cols]) if cols
                else np.full(te.sum(), ytr.mean()))
        sse_model += float(((yte - pred) ** 2).sum())
        sse_base += float(((yte - ytr.mean()) ** 2).sum())
    return 1 - sse_model / sse_base


def best_insample_r2(X, y):
    """Best single-feature R2 with no held-out data -- the number to NOT believe."""
    best = (-np.inf, None)
    tot = float(((y - y.mean()) ** 2).sum())
    for j in range(X.shape[1]):
        resid = y - ols_predict(X[:, [j]], y, X[:, [j]])
        r2 = 1 - float((resid ** 2).sum()) / tot
        if r2 > best[0]:
            best = (r2, j)
    return best


def permutation_p(X, y, groups, observed, names, rng):
    """How often does the same pipeline beat `observed` on a shuffled response?

    For test A the response is one value per year, so years are shuffled directly. For
    test B the response is a year-by-region matrix and whole years are permuted, which
    preserves the within-year spatial pattern and destroys only the year-weather link.
    """
    uy = np.unique(groups)
    beat = 0
    for _ in range(N_PERM):
        perm = rng.permutation(uy)
        mapping = dict(zip(uy, perm))
        ### move each row's response to the year it was permuted onto
        order = np.concatenate([np.where(groups == mapping[g])[0] for g in uy])
        src = np.concatenate([np.where(groups == g)[0] for g in uy])
        y_perm = y.copy()
        y_perm[src] = y[order] if len(order) == len(src) else y[src]
        if loo_r2(X, y_perm, groups, k_max=1) >= observed:
            beat += 1
    return (beat + 1) / (N_PERM + 1)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    wx, feats = load_weather()
    dis, regions, years = load_disease()
    print(f"window {years[0]}-{years[-1]} ({len(years)} surveyed years), "
          f"{len(regions)} regions, {len(feats)} weather features")

    wx = wx[wx.Region.isin(regions) & wx.Year.isin(years)].copy()
    assert len(wx) == len(regions) * len(years), "weather panel is not balanced"

    ### the gs61_survey_* window is undefined in years where GS61 falls after the survey
    ### date, so those features are incomplete; drop rather than impute them
    incomplete = [f for f in feats if wx[f].isna().any()]
    if incomplete:
        feats = [f for f in feats if f not in incomplete]
        print(f"dropped {len(incomplete)} features with missing region-years: "
              f"{', '.join(incomplete)}")
        print(f"{len(feats)} weather features remain")

    rows = []
    for target in BASE_TARGETS:
        sev = "Severity" in target
        d = dis.copy()
        d["v"] = np.log1p(d[target]) if sev else d[target]
        ### leaf-average first: the year effect is a property of the season, not the leaf
        rt = d.groupby(["Region", "Year"], observed=True).v.mean().unstack("Year")
        rt = rt.loc[regions, years]

        ### ---- test A: national year effect vs national weather
        y_a = rt.mean(axis=0).to_numpy() - rt.to_numpy().mean()
        Xw_nat = (wx.groupby("Year")[feats].mean().loc[years].to_numpy())
        g_a = np.array(years)

        r2_a = loo_r2(Xw_nat, y_a, g_a)
        ins_a, j_a = best_insample_r2(
            (Xw_nat - Xw_nat.mean(0)) / np.where(Xw_nat.std(0) > 0, Xw_nat.std(0), 1), y_a)
        p_a = permutation_p(Xw_nat, y_a, g_a, r2_a, feats, rng)

        ### ---- test B: region-by-year anomaly vs regional weather anomaly
        M = rt.to_numpy()
        inter = M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()
        y_b = inter.flatten(order="F")            # year-major, matching X below
        W = wx.pivot(index="Region", columns="Year", values=feats)
        Xb = []
        for f in feats:
            A = W[f].loc[regions, years].to_numpy()
            A = A - A.mean(axis=1, keepdims=True) - A.mean(axis=0, keepdims=True) + A.mean()
            Xb.append(A.flatten(order="F"))
        Xb = np.column_stack(Xb)
        g_b = np.repeat(years, len(regions))

        r2_b = loo_r2(Xb, y_b, g_b)
        ins_b, j_b = best_insample_r2(
            (Xb - Xb.mean(0)) / np.where(Xb.std(0) > 0, Xb.std(0), 1), y_b)
        p_b = permutation_p(Xb, y_b, g_b, r2_b, feats, rng)

        print(f"\n{SHORT[target]}")
        print(f"  A national year effect   (n={len(y_a)}):  LOO R2 = {r2_a:+.3f}  "
              f"(perm p = {p_a:.3f});  best in-sample single feature R2 = {ins_a:.3f} "
              f"[{feats[j_a]}]")
        print(f"  B region x year anomaly  (n={len(y_b)}):  LOO R2 = {r2_b:+.3f}  "
              f"(perm p = {p_b:.3f});  best in-sample single feature R2 = {ins_b:.3f} "
              f"[{feats[j_b]}]")

        rows += [
            {"target": SHORT[target], "test": "A national year effect", "n": len(y_a),
             "loo_r2": r2_a, "perm_p": p_a, "best_insample_r2": ins_a,
             "best_insample_feature": feats[j_a]},
            {"target": SHORT[target], "test": "B region x year anomaly", "n": len(y_b),
             "loo_r2": r2_b, "perm_p": p_b, "best_insample_r2": ins_b,
             "best_insample_feature": feats[j_b]},
        ]

    out = pd.DataFrame(rows)
    out.to_csv("analysis/evidence/weather_year_effect.csv", index=False)
    print("\nwrote analysis/evidence/weather_year_effect.csv")
    print("\nLOO R2 > 0 means weather beats the training-year mean out of sample; "
          "<= 0 means it does not.")
