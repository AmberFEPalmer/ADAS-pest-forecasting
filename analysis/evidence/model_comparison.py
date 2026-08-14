"""
XGBoost vs LightGBM vs CatBoost vs penalised GLM, on IDENTICAL data.

RESULT. Test RMSE on 2016+, mean over 5 seeds, everything else identical:

  target                  naive   XGBoost  LightGBM  CatBoost      GLM
  Septoria severity       2.875     2.547     2.545    2.531*    3.951
  Septoria incidence     30.333   21.120*    21.661    22.294   23.768
  Yellow rust severity    0.198     0.109*    0.112     0.115    0.521
  Yellow rust incidence  10.678    11.478*   11.877    12.126   12.865
  (* = best; across-seed sd 0.002-0.256 for the trees, GLM deterministic)

"""

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "analysis")
from s07_features_and_model import (  # noqa: E402
    BASE_TARGETS, TEST_START, TRAIN_END, build_feature_table, model_rows_from,
    per_target_feature_cols, rmse, time_series_folds,
)

SEEDS = [0, 1, 2, 3, 4]
CAT_COLS = ["Region", "Leaf"]
OUT_CSV = "analysis/evidence/model_comparison_results.csv"

COMMON = dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample=0.7,
              l2=2.0, l1=0.5, min_child_weight=5, n_rounds=600, patience=40)


def transforms(target):
    is_sev = "Severity" in target
    tf = (lambda v: np.log1p(v)) if is_sev else (lambda v: v)
    inv = (lambda v: np.clip(np.expm1(v), 0, None)) if is_sev else (lambda v: v)
    return tf, inv


### XGBoost
def _xgb(X_fit, y_fit, X_val, y_val, X_full, y_full, X_test, seed):
    import xgboost as xgb
    p = dict(max_depth=COMMON["max_depth"], learning_rate=COMMON["learning_rate"],
             subsample=COMMON["subsample"], colsample_bytree=COMMON["colsample"],
             reg_lambda=COMMON["l2"], reg_alpha=COMMON["l1"],
             min_child_weight=COMMON["min_child_weight"], enable_categorical=True,
             tree_method="hist", random_state=seed)
    es = xgb.XGBRegressor(n_estimators=COMMON["n_rounds"], eval_metric="rmse",
                          early_stopping_rounds=COMMON["patience"], **p)
    es.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    best = int(es.best_iteration)
    final = xgb.XGBRegressor(n_estimators=best + 1, **p)
    final.fit(X_full, y_full)
    return best, final.predict(X_test)

### LightGBM
def _lgb(X_fit, y_fit, X_val, y_val, X_full, y_full, X_test, seed):
    import lightgbm as lgb
    p = dict(objective="regression", max_depth=COMMON["max_depth"], num_leaves=7,
             learning_rate=COMMON["learning_rate"], bagging_fraction=COMMON["subsample"],
             bagging_freq=1, feature_fraction=COMMON["colsample"], lambda_l2=COMMON["l2"],
             lambda_l1=COMMON["l1"], min_child_weight=COMMON["min_child_weight"],
             verbosity=-1, seed=seed)
    tr = lgb.Dataset(X_fit, label=y_fit, categorical_feature=CAT_COLS, free_raw_data=False)
    va = lgb.Dataset(X_val, label=y_val, reference=tr, categorical_feature=CAT_COLS, free_raw_data=False)
    b = lgb.train(p, tr, num_boost_round=COMMON["n_rounds"], valid_sets=[va],
                  callbacks=[lgb.early_stopping(COMMON["patience"], verbose=False)])
    best = int(b.best_iteration)
    full = lgb.Dataset(X_full, label=y_full, categorical_feature=CAT_COLS, free_raw_data=False)
    final = lgb.train(p, full, num_boost_round=max(best, 1))
    return best, final.predict(X_test)

### CatBoost
def _cat(X_fit, y_fit, X_val, y_val, X_full, y_full, X_test, seed):
    from catboost import CatBoostRegressor, Pool
    # CatBoost needs categoricals as strings and cannot take NaN in them.
    def prep(X):
        X = X.copy()
        for c in CAT_COLS:
            X[c] = X[c].astype(str)
        return X
    X_fit, X_val, X_full, X_test = map(prep, (X_fit, X_val, X_full, X_test))
    p = dict(depth=COMMON["max_depth"], learning_rate=COMMON["learning_rate"],
             l2_leaf_reg=COMMON["l2"], subsample=COMMON["subsample"],
             rsm=COMMON["colsample"], min_data_in_leaf=COMMON["min_child_weight"],
             bootstrap_type="Bernoulli", random_seed=seed, verbose=0, allow_writing_files=False)
    m = CatBoostRegressor(iterations=COMMON["n_rounds"], early_stopping_rounds=COMMON["patience"], **p)
    m.fit(Pool(X_fit, y_fit, cat_features=CAT_COLS),
          eval_set=Pool(X_val, y_val, cat_features=CAT_COLS))
    best = int(m.get_best_iteration())
    final = CatBoostRegressor(iterations=max(best + 1, 1), **p)
    final.fit(Pool(X_full, y_full, cat_features=CAT_COLS))
    return best, final.predict(Pool(X_test, cat_features=CAT_COLS))


TREES = {"XGBoost": _xgb, "LightGBM": _lgb, "CatBoost": _cat}


def run_tree(name, model_rows, feature_cols, seed):
    """One tree model, all four targets, on the shared split/folds. Returns
    {target: (test_rmse, best_iter)}."""
    fn = TREES[name]
    train_full = model_rows[model_rows.Year <= TRAIN_END]
    test = model_rows[model_rows.Year >= TEST_START]
    folds = time_series_folds(train_full.Year.unique())

    out = {}
    for t in BASE_TARGETS:
        tf, inv = transforms(t)
        cols = CAT_COLS + feature_cols[t]
        iters = []
        for tr_years, val_years in folds:
            f = train_full[train_full.Year.isin(tr_years)]
            v = train_full[train_full.Year.isin(val_years)]
            if f.empty or v.empty:
                continue
            best, _ = fn(f[cols], tf(f[t]), v[cols], tf(v[t]),
                         f[cols], tf(f[t]), v[cols], seed)
            iters.append(best)
        best_iter = int(np.median(iters))

        saved = COMMON["n_rounds"]
        COMMON["n_rounds"] = best_iter + 1
        try:
            _, pred = fn(train_full[cols], tf(train_full[t]),
                         train_full[cols], tf(train_full[t]),
                         train_full[cols], tf(train_full[t]), test[cols], seed)
        finally:
            COMMON["n_rounds"] = saved
        out[t] = (rmse(inv(pred), test[t]), best_iter)
    return out


### GLM
def run_glm(model_rows, feature_cols):
    from glm_model import ALPHAS, CAT_COLS as GLM_CAT, GLM  

    train_full = model_rows[model_rows.Year <= TRAIN_END]
    test = model_rows[model_rows.Year >= TEST_START]
    folds = time_series_folds(train_full.Year.unique())
    families = ["lognormal", "tweedie", "poisson", "quasibinomial"]

    out = {}
    for t in BASE_TARGETS:
        cols = feature_cols[t]
        best, best_key = np.inf, None
        for fam in families:
            for a in ALPHAS:
                errs = []
                for tr_years, val_years in folds:
                    f = train_full[train_full.Year.isin(tr_years)]
                    v = train_full[train_full.Year.isin(val_years)]
                    if f.empty or v.empty:
                        continue
                    try:
                        m = GLM(fam, a, cols).fit(f[GLM_CAT + cols], f[t])
                        p = m.predict(v[GLM_CAT + cols])
                    except (ValueError, np.linalg.LinAlgError):
                        errs = None
                        break
                    if not np.all(np.isfinite(p)):
                        errs = None
                        break
                    errs.append(rmse(p, v[t]))
                if errs and np.mean(errs) < best:
                    best, best_key = float(np.mean(errs)), (fam, a)
        fam, a = best_key
        m = GLM(fam, a, cols).fit(train_full[GLM_CAT + cols], train_full[t])
        out[t] = (rmse(m.predict(test[GLM_CAT + cols]), test[t]), f"{fam}/a={a:g}")
    return out


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "analysis/evidence") 

    table = build_feature_table(spatial=True)
    model_rows = model_rows_from(table)
    feature_cols = per_target_feature_cols(model_rows)
    test = model_rows[model_rows.Year >= TEST_START]
    train_full = model_rows[model_rows.Year <= TRAIN_END]

    print(f"rows {len(model_rows)} | train {len(train_full)} (<= {TRAIN_END}) | "
          f"test {len(test)} (>= {TEST_START})")
    print(f"features per target: { {t: len(c) for t, c in feature_cols.items()} }")
    print(f"{len(SEEDS)} seeds for the tree models; GLM is deterministic\n")

    recs = []
    for name in TREES:
        for seed in SEEDS:
            for t, (r, bi) in run_tree(name, model_rows, feature_cols, seed).items():
                recs.append({"model": name, "seed": seed, "target": t, "test_rmse": r})
        print(f"  {name} done", flush=True)
    for t, (r, cfg) in run_glm(model_rows, feature_cols).items():
        recs.append({"model": "GLM", "seed": np.nan, "target": t, "test_rmse": r, "config": cfg})
    print("  GLM done", flush=True)

    long = pd.DataFrame(recs)
    long.to_csv(OUT_CSV, index=False)

    order = ["XGBoost", "LightGBM", "CatBoost", "GLM"]
    mean = long.pivot_table(index="target", columns="model", values="test_rmse")[order]
    sd = (long.pivot_table(index="target", columns="model", values="test_rmse", aggfunc="std")
              .reindex(columns=order))

    naive = {t: rmse(np.full(len(test), train_full[t].mean()), test[t]) for t in BASE_TARGETS}
    mean.insert(0, "naive", pd.Series(naive))

    print(f"\n=== test RMSE (2016+), mean over {len(SEEDS)} seeds; identical features/folds/split ===")
    print(mean.round(3).to_string())
    print("\n=== across-seed SD (GLM deterministic, so blank) ===")
    print(sd.round(3).to_string())

    print("\n=== % improvement vs a naive train-mean (higher is better) ===")
    imp = mean[order].rsub(mean["naive"], axis=0).div(mean["naive"], axis=0) * 100
    print(imp.round(1).to_string())

    print("\n=== winner per target ===")
    for t in mean.index:
        row = mean.loc[t, order]
        w = row.idxmin()
        margin = 100 * (row.min() / row.drop(w).min() - 1)
        print(f"  {t:40s} {w:9s} {row.min():8.3f}  ({margin:+.1f}% vs next best, "
              f"seed sd {sd.loc[t, w] if not pd.isna(sd.loc[t, w]) else 0:.3f})")
