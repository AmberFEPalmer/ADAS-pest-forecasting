"""
XGBoost model for the 4 disease-severity/incidence outcomes
"""

import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, "analysis")
from s05_fungicide_decay import compute_fungicide_features  
from s01_season_reference import compute_sow_offsets  
from s06_spatial_spread import compute_spatial_features  

CURATED_AGRO_COLS = [
    "Unknown_ag_12",   # % fields sown with non-certified/unknown-provenance seed
    "Cereals_ag_8_3",  # % fields where the crop 3 years prior was a cereal
    "Percentage of fields with wheat thrice in previous 4 years (%)_ag_11",
]

### predictor target names
BASE_TARGETS = [
    "Zymoseptoria_tritici_Disease_Severity",
    "Yellow_rust_Disease_Severity",
    "Zymoseptoria_tritici_Crop_Incidence",
    "Yellow_rust_Crop_Incidence",
]
LEAVES = ["L1", "L2"]
OTHER_LEAF = {"L1": "L2", "L2": "L1"}

TRAIN_END = 2015
TEST_START = 2016
VAL_YEARS = 4    
N_FOLDS = 4 

def melt_pest_long(pest):
    """Wide (L1_*, L2_* columns) -> long (Region, Year, Leaf, BASE_TARGETS...)."""
    frames = []
    for leaf in LEAVES:
        cols = {f"{leaf}_{t}": t for t in BASE_TARGETS}
        sub = pest[["Region", "Year"] + list(cols.keys())].rename(columns=cols).copy()
        sub["Leaf"] = leaf
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)

### growth stage names
GROWTH_STAGE_PREFIXES = ("gs31_39_", "gs39_survey_", "gs61_survey_")

def growth_stage_cols(cols):
    return [c for c in cols if c.startswith(GROWTH_STAGE_PREFIXES) and not c.endswith("_mean_rh")]


### UKCPVS (UK Cereal Pathogen Virulence Survey) yellow rust race surveillance. Numeric
### columns lifted from data/ukcpvs_yellow_rust.csv; the free-text ones are provenance
### only. OFF by default -- see build_feature_table(ukcpvs=...) for why.
UKCPVS_CSV = "data/ukcpvs_yellow_rust.csv"
UKCPVS_COLS = ["samples_received", "n_counties_sampled", "new_race_detected",
               "rl_ratings_revised", "epidemic_followed"]


def load_ukcpvs():
    u = pd.read_csv(UKCPVS_CSV, comment="#")[["year"] + UKCPVS_COLS]
    u = u.rename(columns={"year": "Year", **{c: f"ukcpvs_{c}_lag1" for c in UKCPVS_COLS}})
    u["Year"] = u["Year"] + 1
    return u


def build_feature_table(lag1_all_targets=True, spatial=False, growth_stage=True,
                        ukcpvs=False):
    pest = pd.read_csv("data/pest_data.csv")
    long_pest = melt_pest_long(pest)

    fung_feats = compute_fungicide_features()[
        ["Region", "Year", "fungicide_smoothed", "fungicide_decay_dose", "fungicide_residual_at_survey"]
    ]
    luc = pd.read_csv("data/prop_LUC.csv")
    weather = pd.read_csv("analysis/growing_season_weather.csv")
    thermal = pd.read_csv("analysis/thermal_wetness_splash.csv")
    sow_map, sow_default = compute_sow_offsets()
    agro = pd.read_csv("data/agronomic_data.csv")[["Region", "Year"] + CURATED_AGRO_COLS]

    base = long_pest.copy()

    lag1_cols = BASE_TARGETS if lag1_all_targets else []

    own_lag1 = long_pest.copy()
    own_lag1["Year"] = own_lag1["Year"] + 1
    own_lag1 = own_lag1.rename(columns={t: f"{t}_lag1" for t in lag1_cols})
    base = base.merge(own_lag1[["Region", "Year", "Leaf"] + [f"{t}_lag1" for t in lag1_cols]],
                       on=["Region", "Year", "Leaf"], how="left")

    own_lag2 = long_pest.copy()
    own_lag2["Year"] = own_lag2["Year"] + 2
    own_lag2 = own_lag2.rename(columns={t: f"{t}_lag2" for t in BASE_TARGETS})
    base = base.merge(own_lag2[["Region", "Year", "Leaf"] + [f"{t}_lag2" for t in BASE_TARGETS]],
                       on=["Region", "Year", "Leaf"], how="left")

    other_lag1 = long_pest.copy()
    other_lag1["Year"] = other_lag1["Year"] + 1
    other_lag1["Leaf"] = other_lag1["Leaf"].map(OTHER_LEAF)  # relabel so the merge lines up on THIS row's leaf
    other_lag1 = other_lag1.rename(columns={t: f"{t}_other_leaf_lag1" for t in lag1_cols})
    base = base.merge(other_lag1[["Region", "Year", "Leaf"] + [f"{t}_other_leaf_lag1" for t in lag1_cols]],
                       on=["Region", "Year", "Leaf"], how="left")

    ### Between-region spread columns 
    if spatial:
        base = base.merge(compute_spatial_features(long_pest, BASE_TARGETS),
                          on=["Region", "Year", "Leaf"], how="left")

    base = base.merge(weather, on=["Region", "Year"], how="left")
    base = base.merge(thermal, on=["Region", "Year"], how="left")

    if growth_stage:
        gs = pd.read_csv("analysis/growth_stage.csv")
        base = base.merge(gs[["Region", "Year"] + growth_stage_cols(gs.columns)],
                          on=["Region", "Year"], how="left")
    base = base.merge(fung_feats, on=["Region", "Year"], how="left")
    base = base.merge(agro, on=["Region", "Year"], how="left")

    ### national, so it varies by Year only -- no Region key
    if ukcpvs:
        base = base.merge(load_ukcpvs(), on="Year", how="left")

    base["sow_offset"] = base.apply(lambda r: sow_map.get((r["Region"], r["Year"]), sow_default), axis=1)

    luc_wide = luc.set_index(["Region", "Year"])[["prop_anth", "prop_nature"]]
    base = base.merge(luc_wide, on=["Region", "Year"], how="left")
    base = base.sort_values(["Region", "Leaf", "Year"])
    base[["prop_anth", "prop_nature"]] = base.groupby(["Region", "Leaf"])[["prop_anth", "prop_nature"]].ffill()

    base["Region"] = base["Region"].astype("category")
    base["Leaf"] = base["Leaf"].astype("category")
    return base


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


### Every target is a percentage: incidence is % of fields, severity is % leaf area
TARGET_LO, TARGET_HI = 0.0, 100.0


def clip_to_range(values):
    """Force predictions into the physically possible range for a percentage."""
    return np.clip(values, TARGET_LO, TARGET_HI)

### expanding-window cross-validation folds
def time_series_folds(years, n_folds=N_FOLDS, val_years=VAL_YEARS):
    years = sorted(years)
    n = len(years)
    burn_in = max(20, n // 3)  # every fold needs enough history to train on
    candidate_starts = sorted(set(np.linspace(burn_in, n - val_years, n_folds).astype(int)))
    folds = []
    for start_idx in candidate_starts:
        train_years = years[:start_idx]
        val_years_block = years[start_idx:start_idx + val_years]
        if len(val_years_block) == val_years and len(train_years) >= 20:
            folds.append((train_years, val_years_block))
    return folds


def model_rows_from(table):
    return table[(table.Year < 2026) & (table.Region != "Scotland")].copy()


def default_feature_cols(model_rows):
    drop_cols = set(BASE_TARGETS) | {"Region", "Year", "Leaf"}
    return [c for c in model_rows.columns if c not in drop_cols]

MECH_WEATHER = {
    "Zymoseptoria_tritici": [
        "spring_to_june_mean_rh",         # humidity during the main infection window
        "spring_to_june_wetness_hours",   # leaf wetness duration -- the infection requirement
        "spring_to_june_splash_risk",     # rain-splash dispersal up the canopy
        "spring_to_june_total_precip",
        "spring_mean_temp",               # latent period / development rate
        "winter_mean_rh",                 # early inoculum build-up on lower leaves
    ],
    "Yellow_rust": [
        "autumn_mean_temp",               # green bridge / overwinter inoculum survival
        "winter_mean_temp",               # survival through winter
        "winter_total_precip",            # mild wet winter (see note above re splash)
        "spring_to_june_mean_temp",       # cool spring favours rapid cycling
        "spring_to_june_mean_rh",         # moisture for infection
    ],
}


def mechanistic_feature_cols(model_rows, lean_history=False):
    all_cols = default_feature_cols(model_rows)
    natl = [c for c in all_cols if c.endswith("_natl_lag1")]
    fung = [c for c in all_cols if c.startswith("fungicide")]

    out = {}
    for t in BASE_TARGETS:
        pathogen = "Yellow_rust" if "Yellow_rust" in t else "Zymoseptoria_tritici"
        hist = [c for c in all_cols
                if ("_lag1" in c or "_lag2" in c) and not c.endswith(("_natl_lag1", "_nbr_lag1"))]
        if lean_history:
            hist = [c for c in hist if c.startswith(t)]
        cols = MECH_WEATHER[pathogen] + hist + fung
        if pathogen == "Yellow_rust":
            cols += natl
        missing = [c for c in cols if c not in all_cols]
        if missing:
            raise KeyError(f"{t}: feature(s) not in the table: {missing}")
        out[t] = cols
    return out


def per_target_feature_cols(model_rows):
    all_cols = default_feature_cols(model_rows)
    natl_cols = [c for c in all_cols if c.endswith("_natl_lag1")]
    gs = growth_stage_cols(all_cols)
    ### UKCPVS surveys a yellow rust pathogen, so it is routed to those targets only --
    ### same treatment as the national spread columns. Absent unless ukcpvs=True.
    ukcpvs_cols = [c for c in all_cols if c.startswith("ukcpvs_")]
    routed = (set(natl_cols) | set(gs) | set(ukcpvs_cols)
              | {c for c in all_cols if c.endswith("_nbr_lag1")})
    base_cols = [c for c in all_cols if c not in routed]
    return {t: base_cols
               + (natl_cols + ukcpvs_cols if "Yellow_rust" in t else [])
               + ([] if "Yellow_rust" in t else gs)
            for t in BASE_TARGETS}


def rolling_baseline(model_rows, target, k, tf):
    d = model_rows[["Region", "Leaf", "Year", target]].copy()
    d["tv"] = tf(d[target])
    d = d.sort_values(["Region", "Leaf", "Year"])
    base = (d.groupby(["Region", "Leaf"], observed=True)["tv"]
              .transform(lambda s: s.shift(1).rolling(k, min_periods=2).mean()))
    return base.reindex(model_rows.index)


def run_targets(model_rows, feature_cols, cat_cols=None, seed=0, baseline_k=None,
                train_end=None, test_start=None):
    if cat_cols is None:
        cat_cols = ["Region", "Leaf"]

    def cols_for(target):
        cols = feature_cols[target] if isinstance(feature_cols, dict) else feature_cols
        return cat_cols + cols

    train_end = TRAIN_END if train_end is None else train_end
    test_start = TEST_START if test_start is None else test_start

    train_full = model_rows[model_rows.Year <= train_end]
    test = model_rows[model_rows.Year >= test_start]
    folds = time_series_folds(train_full.Year.unique())

    results = []
    importances = {}
    preds = test[["Region", "Year", "Leaf"]].copy()

    for target in BASE_TARGETS:
        use_cols = cols_for(target)
        X_test = test[use_cols]  
        is_severity = "Severity" in target
        tf = (lambda v: np.log1p(v)) if is_severity else (lambda v: v)
        inv = (lambda v: np.clip(np.expm1(v), 0, None)) if is_severity else (lambda v: v)

        if baseline_k:
            base_all = rolling_baseline(model_rows, target, baseline_k, tf)
            b_train = base_all.reindex(train_full.index)
            b_test = base_all.reindex(test.index)
            fit_mask = b_train.notna()
            b_test = b_test.fillna(tf(train_full[target]).mean())
        else:
            b_train = pd.Series(0.0, index=train_full.index)
            b_test = pd.Series(0.0, index=test.index)
            fit_mask = pd.Series(True, index=train_full.index)

        fit_rows = train_full[fit_mask]
        X_train_full = fit_rows[use_cols]
        y_train_full = tf(fit_rows[target]) - b_train[fit_mask]

        fold_best_iters = []
        for tr_years, val_years_block in folds:
            fold_fit = train_full[train_full.Year.isin(tr_years) & fit_mask]
            fold_val = train_full[train_full.Year.isin(val_years_block) & fit_mask]
            if fold_fit.empty or fold_val.empty:
                continue
            X_fold_fit = fold_fit[use_cols]
            X_fold_val = fold_val[use_cols]
            y_fold_fit = tf(fold_fit[target]) - b_train.reindex(fold_fit.index)
            y_fold_val = tf(fold_val[target]) - b_train.reindex(fold_val.index)

            fold_model = xgb.XGBRegressor(
                n_estimators=600, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0.5,
                min_child_weight=5, enable_categorical=True, tree_method="hist",
                early_stopping_rounds=40, eval_metric="rmse", random_state=seed,
            )
            fold_model.fit(X_fold_fit, y_fold_fit, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
            fold_best_iters.append(fold_model.best_iteration)

        if not fold_best_iters:
            raise ValueError(
                f"no usable early-stopping folds for {target!r}: time_series_folds() needs "
                f">= 21 training years (got {train_full.Year.nunique()}). Lower N_FOLDS/"
                f"VAL_YEARS or widen the training window."
            )
        best_iter = int(np.median(fold_best_iters))

        final_model = xgb.XGBRegressor(
            n_estimators=best_iter + 1, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0.5,
            min_child_weight=5, enable_categorical=True, tree_method="hist",
            random_state=seed,
        )
        final_model.fit(X_train_full, y_train_full)

        pred_train = inv(final_model.predict(X_train_full) + b_train[fit_mask].to_numpy())
        pred_test = inv(final_model.predict(X_test) + b_test.to_numpy())
        y_train_full_raw = fit_rows[target]
        y_test = test[target]
        preds[f"{target}_pred"] = pred_test

        naive = np.full_like(y_test, y_train_full_raw.mean(), dtype=float)
        train_rmse = rmse(pred_train, y_train_full_raw)
        test_rmse = rmse(pred_test, y_test)
        naive_rmse = rmse(naive, y_test)
        results.append({
            "target": target, "best_iter": best_iter, "fold_best_iters": fold_best_iters,
            "train_rmse": train_rmse, "test_rmse": test_rmse,
            "naive_rmse": naive_rmse, "obs_std": y_test.std(),
            "improvement_pct": 100 * (1 - test_rmse / naive_rmse),
        })

        imp = pd.Series(final_model.feature_importances_, index=X_train_full.columns)
        importances[target] = imp.sort_values(ascending=False).head(10)

    return pd.DataFrame(results), importances, preds, folds, test


if __name__ == "__main__":
    table = build_feature_table(spatial=True)
    model_rows = model_rows_from(table)
    print(f"Modelling rows: {len(model_rows)}  (years {model_rows.Year.min()}-{model_rows.Year.max()}, "
          f"{len(LEAVES)} leaves pooled per row-group)")

    feature_cols = per_target_feature_cols(model_rows)
    for t, cols in feature_cols.items():
        print(f"n features [{t}]: {len(cols)} (+ Region, Leaf as categorical inputs)")

    res, importances, preds, folds, test = run_targets(model_rows, feature_cols)

    print(f"test n={len(test)} (>= {TEST_START})")
    print(f"early-stopping folds (expanding-window, spread across the full training span):")
    for tr_years, val_years_block in folds:
        print(f"  train <= {tr_years[-1]} (n_years={len(tr_years)})  val {val_years_block[0]}-{val_years_block[-1]}")

    res.to_csv("analysis/s07_test_results.csv", index=False)
    # Pivot predictions back to the submission's wide L1_/L2_ column layout.
    preds_wide = preds.pivot(index=["Region", "Year"], columns="Leaf",
                              values=[f"{t}_pred" for t in BASE_TARGETS])
    preds_wide.columns = [f"{leaf}_{t}" for t, leaf in preds_wide.columns]
    preds_wide = preds_wide.reset_index()
    preds_wide.to_csv("analysis/s07_test_predictions.csv", index=False)

    print("\n=== RMSE by target (test = 2016+, L1+L2 pooled) ===")
    print(res.to_string(index=False))

    print("\n=== Top-10 feature importances per target ===")
    for target, imp in importances.items():
        print(f"\n{target}:")
        print(imp.to_string())
