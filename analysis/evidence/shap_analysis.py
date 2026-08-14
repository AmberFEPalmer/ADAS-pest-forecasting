"""
SHAP feature analysis for the production XGBoost model.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402

sys.path.insert(0, "analysis")
from s07_features_and_model import (  # noqa: E402
    BASE_TARGETS, CURATED_AGRO_COLS, GROWTH_STAGE_PREFIXES, TEST_START, TRAIN_END,
    build_feature_table, model_rows_from, per_target_feature_cols, rmse,
    time_series_folds,
)

SEEDS = [0, 1, 2, 3, 4]
CAT_COLS = ["Region", "Leaf"]
TOP_N = 15
FIG_DIR = Path("report/figures")

XGB_PARAMS = dict(
    max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
    reg_lambda=2.0, reg_alpha=0.5, min_child_weight=5,
    enable_categorical=True, tree_method="hist",
)


def transforms(target):
    is_sev = "Severity" in target
    tf = (lambda v: np.log1p(v)) if is_sev else (lambda v: v)
    inv = (lambda v: np.clip(np.expm1(v), 0, None)) if is_sev else (lambda v: v)
    return tf, inv


def feature_family(col, weather_cols, thermal_cols):
    """Roll the ~76 features up into the families the report talks about."""
    if col in CAT_COLS:
        return "region / leaf"
    if col.endswith(("_natl_lag1", "_nbr_lag1")):
        return "spatial spread"
    if "_lag1" in col or "_lag2" in col:
        return "disease history"
    if col.startswith(GROWTH_STAGE_PREFIXES):
        return "growth-stage weather"
    if col in thermal_cols:
        return "thermal / wetness / splash"
    if col in weather_cols:
        return "seasonal weather"
    if col.startswith("fungicide"):
        return "fungicide"
    if col in CURATED_AGRO_COLS:
        return "agronomic"
    if col in ("prop_anth", "prop_nature"):
        return "land use"
    if col == "sow_offset":
        return "sowing date"
    return "other"


def fit_production_model(train_full, use_cols, target, tf, seed):
    """Refit exactly as xgboost_model.run_targets does: median best-iteration
    across the expanding-window folds, then one final fit on the full window."""
    folds = time_series_folds(train_full.Year.unique())
    best_iters = []
    for tr_years, val_years in folds:
        f = train_full[train_full.Year.isin(tr_years)]
        v = train_full[train_full.Year.isin(val_years)]
        if f.empty or v.empty:
            continue
        m = xgb.XGBRegressor(n_estimators=600, early_stopping_rounds=40,
                             eval_metric="rmse", random_state=seed, **XGB_PARAMS)
        m.fit(f[use_cols], tf(f[target]), eval_set=[(v[use_cols], tf(v[target]))],
              verbose=False)
        best_iters.append(m.best_iteration)

    best_iter = int(np.median(best_iters))
    final = xgb.XGBRegressor(n_estimators=best_iter + 1, random_state=seed, **XGB_PARAMS)
    final.fit(train_full[use_cols], tf(train_full[target]))
    return final


def tree_shap(model, X):
    """Exact TreeSHAP via XGBoost. Returns (contribs, bias) with contribs shaped
    (n_rows, n_features) -- the trailing bias column is split off."""
    dm = xgb.DMatrix(X, enable_categorical=True)
    raw = model.get_booster().predict(dm, pred_contribs=True)
    return raw[:, :-1], raw[:, -1]


def numeric_for_plot(X):
    """Beeswarm colouring needs numeric feature values; encode the categoricals."""
    out = X.copy()
    for c in CAT_COLS:
        out[c] = out[c].cat.codes
    return out


if __name__ == "__main__":
    table = build_feature_table(spatial=True)
    model_rows = model_rows_from(table)
    feature_cols = per_target_feature_cols(model_rows)
    train_full = model_rows[model_rows.Year <= TRAIN_END]
    test = model_rows[model_rows.Year >= TEST_START]

    weather_cols = set(pd.read_csv("analysis/growing_season_weather.csv", nrows=0).columns)
    thermal_cols = set(pd.read_csv("analysis/thermal_wetness_splash.csv", nrows=0).columns)

    print(f"rows {len(model_rows)} | train {len(train_full)} (<= {TRAIN_END}) | "
          f"test {len(test)} (>= {TEST_START})")
    print(f"TreeSHAP over {len(SEEDS)} seeds; severity in log1p units, "
          f"incidence in percentage points\n")

    imp_rows, dir_rows, check_rows = [], [], []
    shap_for_plot = {}

    for target in BASE_TARGETS:
        tf, inv = transforms(target)
        use_cols = CAT_COLS + feature_cols[target]
        X_train, X_test = train_full[use_cols], test[use_cols]

        acc_train, acc_test, acc_signed, acc_gain = [], [], [], []
        for seed in SEEDS:
            model = fit_production_model(train_full, use_cols, target, tf, seed)

            s_train, _ = tree_shap(model, X_train)
            s_test, bias = tree_shap(model, X_test)

            ### additivity check -- TreeSHAP must reconstruct the margin exactly
            recon = s_test.sum(axis=1) + bias
            margin = model.predict(X_test, output_margin=True)
            assert np.allclose(recon, margin, atol=1e-4), \
                f"{target}: SHAP does not reconstruct the prediction"

            acc_train.append(np.abs(s_train).mean(axis=0))
            acc_test.append(np.abs(s_test).mean(axis=0))
            acc_signed.append(s_test)

            acc_gain.append(model.feature_importances_)
            if seed == SEEDS[0]:
                check_rows.append({"target": target, "seed": seed,
                                   "test_rmse": rmse(inv(model.predict(X_test)), test[target])})

        mean_train = np.mean(acc_train, axis=0)
        mean_test = np.mean(acc_test, axis=0)
        signed = np.mean(acc_signed, axis=0)
        shap_for_plot[target] = signed

        mean_gain = np.mean(acc_gain, axis=0)
        for j, col in enumerate(use_cols):
            imp_rows.append({
                "target": target, "feature": col,
                "family": feature_family(col, weather_cols, thermal_cols),
                "mean_abs_shap_train": mean_train[j],
                "mean_abs_shap_test": mean_test[j],
                "gain_importance": mean_gain[j],
            })
            ### direction: does a HIGH value of this feature push the prediction up?
            v = X_test[col]
            v = v.cat.codes if col in CAT_COLS else v
            ok = v.notna().to_numpy()
            corr = (np.corrcoef(v[ok].astype(float), signed[ok, j])[0, 1]
                    if ok.sum() > 2 and np.std(signed[ok, j]) > 0 else np.nan)
            dir_rows.append({"target": target, "feature": col,
                             "mean_abs_shap_test": mean_test[j], "value_shap_corr": corr})

    imp = pd.DataFrame(imp_rows)
    imp.to_csv("analysis/evidence/shap_importance.csv", index=False)
    direction = pd.DataFrame(dir_rows)
    direction.to_csv("analysis/evidence/shap_direction.csv", index=False)

    ### top features per target
    print("=== top features by mean |SHAP| on the test window (2016+) ===")
    print("    corr = correlation between feature value and its SHAP;")
    print("    positive means a high value pushes the prediction UP.\n")
    for target in BASE_TARGETS:
        unit = "log1p" if "Severity" in target else "pct pts"
        d = (imp[imp.target == target]
             .merge(direction[["target", "feature", "value_shap_corr"]],
                    on=["target", "feature"])
             .sort_values("mean_abs_shap_test", ascending=False).head(TOP_N))
        print(f"{target}  [{unit}]")
        for _, r in d.iterrows():
            c = r.value_shap_corr
            arrow = "  " if pd.isna(c) else ("up" if c > 0.15 else "dn" if c < -0.15 else "~ ")
            print(f"   {r.mean_abs_shap_test:9.4f}  {arrow}  {r.feature[:58]:58s} {r.family}")
        print()

    ### gain based importance
    print("=== gain-based importance vs SHAP: Spearman rank correlation ===")
    print("    plus the features whose rank moves most between the two.\n")
    for target in BASE_TARGETS:
        d = imp[imp.target == target].copy()
        d["rank_shap"] = d.mean_abs_shap_test.rank(ascending=False)
        d["rank_gain"] = d.gain_importance.rank(ascending=False)
        rho = d.rank_shap.corr(d.rank_gain, method="spearman")
        d["move"] = d.rank_gain - d.rank_shap  # + = SHAP ranks it higher than gain did
        big = d.reindex(d.move.abs().sort_values(ascending=False).index).head(3)
        moves = ", ".join(f"{r.feature[:34]} ({int(r.rank_gain)}->{int(r.rank_shap)})"
                          for _, r in big.iterrows())
        print(f"  {target:40s} rho={rho:.2f}")
        print(f"      biggest rank moves (gain->SHAP): {moves}")
    print()

    ### family roll up
    grp = (imp.groupby(["target", "family"])[["mean_abs_shap_train", "mean_abs_shap_test"]]
              .sum().reset_index())
    grp["share_test_%"] = 100 * grp.groupby("target").mean_abs_shap_test.transform(
        lambda s: s / s.sum())
    grp["share_train_%"] = 100 * grp.groupby("target").mean_abs_shap_train.transform(
        lambda s: s / s.sum())
    grp["shift_pp"] = grp["share_test_%"] - grp["share_train_%"]
    grp.to_csv("analysis/evidence/shap_groups.csv", index=False)

    print("=== share of total attribution by feature family (test window) ===")
    ### float32 comes back from XGBoost; cast so round() prints cleanly
    share = grp.pivot(index="family", columns="target",
                      values="share_test_%").fillna(0).astype(float)
    share.columns = [c.replace("_Disease_Severity", " sev").replace("_Crop_Incidence", " inc")
                     .replace("Zymoseptoria_tritici", "Septoria").replace("Yellow_rust", "YR")
                     for c in share.columns]
    print(share.round(1).sort_values(share.columns[0], ascending=False).to_string())

    print("\n=== train -> test shift in attribution share (pp; + = leaned on MORE in test) ===")
    shift = grp.pivot(index="family", columns="target",
                      values="shift_pp").fillna(0).astype(float)
    shift.columns = share.columns
    print(shift.round(1).to_string())

    ### plotting
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for target in BASE_TARGETS:
        use_cols = CAT_COLS + feature_cols[target]
        plt.figure()
        shap.summary_plot(shap_for_plot[target], numeric_for_plot(test[use_cols]),
                          max_display=TOP_N, show=False, plot_size=(9, 6))
        unit = "log1p severity" if "Severity" in target else "incidence (pct pts)"
        plt.title(f"{target}\nSHAP on test window 2016+ ({unit})", fontsize=10)
        plt.tight_layout()
        out = FIG_DIR / f"shap_beeswarm_{target}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  wrote {out}")

    fig, ax = plt.subplots(figsize=(10, 5))
    share.plot.barh(ax=ax)
    ax.set_xlabel("share of total |SHAP| attribution (%)")
    ax.set_ylabel("")
    ax.set_title("What the model actually uses, by feature family (test window 2016+)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_families.png", dpi=150)
    plt.close()
    print(f"  wrote {FIG_DIR / 'shap_families.png'}")

    stored = pd.read_csv("analysis/s07_test_results.csv").set_index("target").test_rmse
    print("\n=== self-check: refit here vs analysis/s07_test_results.csv (seed 0) ===")
    worst = 0.0
    for r in check_rows:
        s = stored.get(r["target"], np.nan)
        delta = abs(r["test_rmse"] - s)
        worst = max(worst, delta / max(s, 1e-9))
        flag = "exact" if delta < 1e-9 else f"differs by {delta:.2e}"
        print(f"  {r['target']:40s} {r['test_rmse']:8.4f} vs {s:8.4f}  {flag}")
    if worst > 1e-3:
        raise SystemExit("SHAP models differ materially from the stored results -- "
                         "XGB_PARAMS is out of sync with "
                         "s07_features_and_model.run_targets()")
    if worst > 1e-6:
        print(f"\n  note: max relative difference {worst:.1e} -- too large for float32 "
              f"round-tripping.\n  s07_features_and_model.py is deterministic, so the "
              f"stored CSV is stale;\n  re-run it to refresh.")
    else:
        print(f"\n  all four match within float32 precision (max rel. diff {worst:.1e}) "
              f"-- these ARE the production models.")
