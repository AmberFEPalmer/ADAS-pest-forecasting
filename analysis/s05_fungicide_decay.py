"""
Fungicide timing/decay feature: an engineered replacement for feeding raw
biennial dose_rate straight into a model (as analysis/s07_features_and_model.py did
previously).

Two problems with the raw DEFRA figure (data/fungicide_data.csv, see
analysis/fungicide_audit.py):

  1. Coverage is BIENNIAL (even years only, 1990-2024) 
  2. It is a single annual (kg active substance / treated area) aggregate
     with no application date 
This module builds two features that address both:

  fungicide_smoothed  : an exponential-kernel-weighted average of dose_rate
      over nearby EVEN years within the same region 

  fungicide_decay_dose : fungicide_smoothed multiplied by a within-season
      RESIDUAL-ACTIVITY decay factor at the survey date. 
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "analysis")
from s01_season_reference import N_DAYS, compute_sow_offsets  # noqa: E402

TAU_YEARS = 3.0          # kernel half-width (years) for smoothing across even-year readings
MAX_GAP_YEARS = 10.0     # beyond this distance from the nearest reading, leave NaN (no extrapolation)

T2_DAYS_AFTER_SOWING = 215   # approx GS39/flag-leaf timing for UK autumn-sown wheat (literature default)
PERSISTENCE_DAYS = 25        # approx residual protection window of a modern T2 fungicide mix
SURVEY_DAY = N_DAYS          # end of the Oct1(Year-1) -> 20 Jun(Year) window used throughout analysis/


def _smooth_region(years_avail, doses_avail, query_years, tau, max_gap):
    years_avail = np.asarray(years_avail, dtype=float)
    doses_avail = np.asarray(doses_avail, dtype=float)
    out = np.full(len(query_years), np.nan)
    for i, y in enumerate(query_years):
        dist = np.abs(years_avail - y)
        if dist.min() > max_gap:
            continue
        w = np.exp(-dist / tau)
        out[i] = float(np.sum(w * doses_avail) / np.sum(w))
    return out


def compute_fungicide_features(year_range=range(1971, 2027)):
    fung = pd.read_csv("data/fungicide_data.csv")
    fung["dose_rate"] = fung.fungicide_kg / fung.fungicide_area
    sow_map, sow_default = compute_sow_offsets()

    rows = []
    for region, sub in fung.groupby("Region"):
        sub = sub.dropna(subset=["dose_rate"])
        smoothed = _smooth_region(sub.Year.to_numpy(), sub.dose_rate.to_numpy(),
                                   list(year_range), TAU_YEARS, MAX_GAP_YEARS)
        for year, dose in zip(year_range, smoothed):
            rows.append({"Region": region, "Year": year, "fungicide_smoothed": dose})

    out = pd.DataFrame(rows)

    sow_offset = out.apply(lambda r: sow_map.get((r["Region"], r["Year"]), sow_default), axis=1)
    sow_day = np.clip(sow_offset, 0, SURVEY_DAY - 30)
    t2_day = sow_day + T2_DAYS_AFTER_SOWING
    gap_days = np.clip(SURVEY_DAY - t2_day, 0, None)  # if T2 hasn't happened yet by "survey", no residual credit
    residual = np.exp(-gap_days / PERSISTENCE_DAYS)

    out["fungicide_residual_at_survey"] = residual
    out["fungicide_decay_dose"] = out["fungicide_smoothed"] * residual
    return out


if __name__ == "__main__":
    feats = compute_fungicide_features()
    feats.to_csv("analysis/fungicide_decay.csv", index=False)
    print(f"Wrote analysis/fungicide_decay.csv: {feats.shape}")
    print(feats.describe())
