"""
Does a generalised linear
model beat the gradient-boosted trees on these four targets?

Same feature table, same per-target feature routing, same 1971-2015 / 2016+ split, and the same
multi-fold expanding-window CV, so what changes between this
script and the XGBoost one is the type of model only.

Why GLM tested?
- small dataset
- severity if non-negative, right-skewed

The GLM loses to XGBoost on all four targets.

  target                 GLM (1se)   XGBoost   naive     GLM selected as
  Septoria severity          3.459     2.816   2.875     quasibinomial/mech_lean a=0.003
  Yellow rust severity       0.521     0.109   0.198     tweedie/full            a=0
  Septoria incidence        24.409    23.060  30.333     quasibinomial/mech      a=0.3
  Yellow rust incidence     12.709    11.607  10.678     lognormal/mech_lean     a=3
"""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "analysis")
from s07_features_and_model import (  # noqa: E402
    BASE_TARGETS, TEST_START, TRAIN_END, build_feature_table, mechanistic_feature_cols,
    model_rows_from, per_target_feature_cols, rmse, time_series_folds,
)

CAT_COLS = ["Region", "Leaf"]

ALPHAS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

# Upper bound of each target's natural scale, used to put the quasi-binomial
# response on [0, 1]. Both families of target are percentages.
SCALE_MAX = 100.0

def _preprocessor(numeric_cols):
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric_cols),
        ],
        remainder="drop",
    )


class GLM:
    def __init__(self, family, alpha, numeric_cols):
        self.family = family
        self.alpha = alpha
        self.numeric_cols = numeric_cols

    def _make_pipe(self, estimator):
        return Pipeline([("prep", _preprocessor(self.numeric_cols)), ("est", estimator)])

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if self.family == "lognormal":
            est = TweedieRegressor(power=0, link="identity", alpha=self.alpha,
                                   max_iter=10000, tol=1e-6)
            self.pipe_ = self._make_pipe(est).fit(X, np.log1p(y))
        elif self.family in ("tweedie", "poisson"):
            power = 1.5 if self.family == "tweedie" else 1.0
            est = TweedieRegressor(power=power, link="log", alpha=self.alpha,
                                   max_iter=10000, tol=1e-6)
            self.pipe_ = self._make_pipe(est).fit(X, y)
        elif self.family == "quasibinomial":
            p = np.clip(y / SCALE_MAX, 0.0, 1.0)
            X2 = pd.concat([X, X], axis=0)
            y2 = np.r_[np.ones(len(X)), np.zeros(len(X))]
            w = np.r_[p, 1.0 - p]
            C = 1e12 if self.alpha == 0 else 1.0 / (self.alpha * len(X))
            # l1_ratio=0 is pure L2; sklearn >= 1.8 deprecates the `penalty` argument.
            est = LogisticRegression(l1_ratio=0, C=C, solver="lbfgs", max_iter=10000, tol=1e-6)
            self.pipe_ = self._make_pipe(est).fit(X2, y2, est__sample_weight=w)
        else:
            raise ValueError(f"unknown family {self.family!r}")
        return self

    def predict(self, X):
        if self.family == "lognormal":
            return np.clip(np.expm1(self.pipe_.predict(X)), 0, SCALE_MAX)
        if self.family == "quasibinomial":
            return self.pipe_.predict_proba(X)[:, 1] * SCALE_MAX
        return np.clip(self.pipe_.predict(X), 0, SCALE_MAX)


def feature_sets(model_rows):
    """The three candidate predictor sets, all already defined and justified in
    analysis/s07_features_and_model.py"""
    return {
        "full": per_target_feature_cols(model_rows),
        "mech": mechanistic_feature_cols(model_rows),
        "mech_lean": mechanistic_feature_cols(model_rows, lean_history=True),
    }


def cv_score(train_full, folds, target, cols, family, alpha):
    errs = []
    for tr_years, val_years in folds:
        fit_rows = train_full[train_full.Year.isin(tr_years)]
        val_rows = train_full[train_full.Year.isin(val_years)]
        if fit_rows.empty or val_rows.empty:
            continue
        try:
            model = GLM(family, alpha, cols).fit(fit_rows[CAT_COLS + cols], fit_rows[target])
            pred = model.predict(val_rows[CAT_COLS + cols])
        except (ValueError, np.linalg.LinAlgError):
            return []
        if not np.all(np.isfinite(pred)):
            return []
        errs.append(rmse(pred, val_rows[target]))
    return errs


def select(grid):
    best = grid.loc[grid.cv_rmse.idxmin()]
    path = grid[(grid.family == best.family) & (grid.feature_set == best.feature_set)]
    within = path[path.cv_rmse <= best.cv_rmse + best.cv_se]
    return {"min_cv": best, "1se": within.loc[within.alpha.idxmax()]}


def fit_and_score(train_full, test, target, cols, family, alpha):
    """Refit on the whole training window and score once on the test split."""
    model = GLM(family, alpha, cols).fit(train_full[CAT_COLS + cols], train_full[target])
    pred_train = model.predict(train_full[CAT_COLS + cols])
    pred_test = model.predict(test[CAT_COLS + cols])
    return pred_train, pred_test


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    table = build_feature_table(spatial=True)
    model_rows = model_rows_from(table)
    fsets = feature_sets(model_rows)

    train_full = model_rows[model_rows.Year <= TRAIN_END]
    test = model_rows[model_rows.Year >= TEST_START]
    folds = time_series_folds(train_full.Year.unique())

    print(f"Modelling rows: {len(model_rows)} (years {model_rows.Year.min()}-{model_rows.Year.max()})")
    print(f"train n={len(train_full)} (<= {TRAIN_END}), test n={len(test)} (>= {TEST_START})")
    print(f"{len(folds)} expanding-window CV folds, all inside the training window:")
    for tr_years, val_years in folds:
        print(f"  train <= {tr_years[-1]} (n_years={len(tr_years)})  val {val_years[0]}-{val_years[-1]}")
    print(f"grid per target: {len(FAMILIES := ['lognormal', 'tweedie', 'poisson', 'quasibinomial'])} families"
          f" x {len(fsets)} feature sets x {len(ALPHAS)} penalties")

    results, all_cv, all_test = [], [], []
    preds = test[["Region", "Year", "Leaf"]].copy()

    for target in BASE_TARGETS:
        grid = []
        for fs_name, fs in fsets.items():
            cols = fs[target]
            for family in FAMILIES:
                for alpha in ALPHAS:
                    errs = cv_score(train_full, folds, target, cols, family, alpha)
                    grid.append({
                        "target": target, "feature_set": fs_name, "family": family, "alpha": alpha,
                        "cv_rmse": float(np.mean(errs)) if errs else np.nan,
                        "cv_se": float(np.std(errs, ddof=1) / np.sqrt(len(errs))) if len(errs) > 1 else np.nan,
                    })
        grid = pd.DataFrame(grid)
        all_cv.append(grid)
        usable = grid.dropna(subset=["cv_rmse", "cv_se"])
        if usable.empty:
            raise RuntimeError(f"no GLM config could be fitted for {target!r}")

        y_train, y_test = train_full[target], test[target]
        naive_rmse = rmse(np.full(len(y_test), y_train.mean()), y_test)

        for rule, pick in select(usable).items():
            cols = fsets[pick.feature_set][target]
            pred_train, pred_test = fit_and_score(train_full, test, target, cols,
                                                  pick.family, pick.alpha)
            test_rmse = rmse(pred_test, y_test)
            if rule == "1se":  # the reported GLM forecast; see the printout below
                preds[f"{target}_pred"] = pred_test
            results.append({
                "target": target, "rule": rule, "family": pick.family,
                "feature_set": pick.feature_set, "alpha": pick.alpha, "n_features": len(cols),
                "cv_rmse": pick.cv_rmse, "train_rmse": rmse(pred_train, y_train),
                "test_rmse": test_rmse, "naive_rmse": naive_rmse, "obs_std": y_test.std(),
                "improvement_pct": 100 * (1 - test_rmse / naive_rmse),
            })

        for _, row in usable.iterrows():
            c = fsets[row.feature_set][target]
            _, pt = fit_and_score(train_full, test, target, c, row.family, row.alpha)
            all_test.append({**row.to_dict(), "test_rmse": rmse(pt, y_test)})

    res = pd.DataFrame(results)
    res.to_csv("analysis/glm_results.csv", index=False)
    pd.concat(all_cv, ignore_index=True).to_csv("analysis/glm_cv_grid.csv", index=False)
    test_grid = pd.DataFrame(all_test)
    test_grid.to_csv("analysis/glm_test_grid.csv", index=False)

    preds_wide = preds.pivot(index=["Region", "Year"], columns="Leaf",
                             values=[f"{t}_pred" for t in BASE_TARGETS])
    preds_wide.columns = [f"{leaf}_{t}" for t, leaf in preds_wide.columns]
    preds_wide.reset_index().to_csv("analysis/glm_predictions.csv", index=False)

    print("\n=== GLM, CV-selected per target (test = 2016+, L1+L2 pooled) ===")
    print(res.round(4).to_string(index=False))

    print("\n=== cost of honest selection (selected vs. best-on-test oracle) ===")
    for target in BASE_TARGETS:
        sub = test_grid[test_grid.target == target]
        orc = sub.loc[sub.test_rmse.idxmin()]
        for _, sel in res[res.target == target].iterrows():
            print(f"  {target:38s} [{sel.rule:6s}] {sel.family:13s}/{sel.feature_set:9s} a={sel.alpha:<6g} "
                  f"{sel.test_rmse:7.3f}   oracle {orc.family:13s}/{orc.feature_set:9s} a={orc.alpha:<6g} "
                  f"{orc.test_rmse:7.3f}   (+{100 * (sel.test_rmse / orc.test_rmse - 1):.1f}%)")

    try:
        xgb_res = pd.read_csv("analysis/s07_test_results.csv")
        cmp = res[["target", "rule", "test_rmse", "naive_rmse"]].merge(
            xgb_res[["target", "test_rmse"]], on="target", suffixes=("_glm", "_xgboost"))
        cmp["winner"] = np.where(cmp.test_rmse_glm < cmp.test_rmse_xgboost, "GLM", "XGBoost")
        cmp["glm_vs_xgb_%"] = 100 * (cmp.test_rmse_glm / cmp.test_rmse_xgboost - 1)
        print("\n=== GLM vs XGBoost (test RMSE, lower is better; negative % = GLM better) ===")
        print(cmp.round(4).to_string(index=False))

        print("\n=== how many of the 156 GLM configs beat XGBoost on test, per target ===")
        xgb_by_t = xgb_res.set_index("target").test_rmse
        for target in BASE_TARGETS:
            sub = test_grid[test_grid.target == target]
            n_beat = int((sub.test_rmse < xgb_by_t[target]).sum())
            print(f"  {target:40s} {n_beat:3d}/{len(sub):3d}   "
                  f"best GLM {sub.test_rmse.min():7.3f} vs XGBoost {xgb_by_t[target]:7.3f}")
    except FileNotFoundError:
        print("\n(run analysis/s07_features_and_model.py first for a side-by-side comparison)")
