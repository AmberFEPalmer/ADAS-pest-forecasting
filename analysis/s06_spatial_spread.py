"""
Adjacency-weighted between-region disease spread features.
"""

import numpy as np
import pandas as pd

### geographic adjacency of the 11 UK regions, for neighbour-mean features
ADJACENCY = {
    "East": ["East Midlands", "South East"],
    "East Midlands": ["East", "Yorkshire and The Humber", "West Midlands",
                      "South East", "North West"],
    "North East": ["Scotland", "North West", "Yorkshire and The Humber"],
    "North West": ["Scotland", "North East", "Yorkshire and The Humber",
                   "East Midlands", "West Midlands", "Wales"],
    "Scotland": ["North East", "North West"],
    "South East": ["East", "East Midlands", "West Midlands", "South West"],
    "South West": ["South East", "West Midlands", "Wales"],
    "Wales": ["North West", "West Midlands", "South West"],
    "West Midlands": ["Wales", "North West", "East Midlands", "South East",
                      "South West"],
    "Yorkshire and The Humber": ["North East", "North West", "East Midlands"],
}


def _assert_symmetric(adj):
    for region, neighbours in adj.items():
        for n in neighbours:
            if n not in adj:
                raise ValueError(f"{region!r} lists unknown neighbour {n!r}")
            if region not in adj[n]:
                raise ValueError(f"adjacency not symmetric: {region!r} lists {n!r}, "
                                 f"but {n!r} does not list {region!r}")


_assert_symmetric(ADJACENCY)


def _safe_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else np.nan


def compute_spatial_features(long_pest, targets):
    prior = long_pest.copy()
    prior[targets] = prior[targets].mask(prior[targets] <= -999.0)
    prior["Year"] = prior["Year"] + 1  # so it joins onto the year it predicts

    rows = []
    for (year, leaf), grp in prior.groupby(["Year", "Leaf"], observed=True):
        by_region = grp.set_index("Region")
        for region in ADJACENCY:
            rec = {"Region": region, "Year": year, "Leaf": leaf}
            neighbours = [n for n in ADJACENCY[region] if n in by_region.index]
            others = [r for r in by_region.index if r != region]
            for t in targets:
                rec[f"{t}_nbr_lag1"] = _safe_mean(by_region.loc[neighbours, t])
                rec[f"{t}_natl_lag1"] = _safe_mean(by_region.loc[others, t])
            rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from s07_features_and_model import BASE_TARGETS, melt_pest_long

    pest = pd.read_csv("data/pest_data.csv")
    long_pest = melt_pest_long(pest)
    feats = compute_spatial_features(long_pest, BASE_TARGETS)

    print(f"rows: {len(feats)}  years {feats.Year.min()}-{feats.Year.max()}")
    print(f"mean neighbours per region: "
          f"{np.mean([len(v) for v in ADJACENCY.values()]):.1f}")
    print("\nnon-null coverage per feature:")
    fcols = [c for c in feats.columns if c.endswith("_lag1")]
    print((feats[fcols].notna().mean() * 100).round(1).to_string())

    print("\ncorrelation between neighbour and national means "
          "(high => adjacency adds little beyond a national year effect):")
    for t in BASE_TARGETS:
        sub = feats[[f"{t}_nbr_lag1", f"{t}_natl_lag1"]].dropna()
        print(f"  {t}: r={sub.corr().iloc[0, 1]:.3f}  (n={len(sub)})")
