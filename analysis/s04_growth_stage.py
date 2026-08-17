"""
Daily weather -> crop development features
Aligns weather to crop growth stages e.g. GS31, GS39, GS61, GS75
Measures how much disease favourable weather the crop experienced
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "analysis")
from s02_fetch_weather import REGIONS, load_daily ### Crop degree days
from s01_season_reference import compute_sow_offsets, daily_wetness_hours ### wetness hours
from s03_thermal_wetness_splash import daily_splash_risk ### splash risk

OUT_CSV = "analysis/growth_stage.csv"

CROP_BASE = 0.0  
### GS31 = start of stem extension
MEDIAN_GS31_DATE = "04-15"   
### only dates until 2025 used
CALIBRATION_END = 2025
SURVEY_MD = (6, 20)   

### Thermal intervals between stages
STAGE_INTERVALS = {"GS39": 250.0, "GS61": 300.0, "GS75": 300.0}

### How strongly drilling date shifts stem extension. A crop drilled 10 days late reaches GS31 about 3 days late 
SOW_SENSITIVITY = 0.3

### daily weather frame for a given region, with derived features
def daily_frame(region):
    df = load_daily(region).sort_values("time").reset_index(drop=True)
    df["cdd"] = np.clip(df.temperature_2m_mean - CROP_BASE, 0, None)
    df["wet_hours"] = daily_wetness_hours(df.precipitation_sum.to_numpy(dtype=float),
                                          df.relative_humidity_2m_mean.to_numpy(dtype=float))
    df["splash"] = daily_splash_risk(df.precipitation_sum.to_numpy(dtype=float),
                                     df.wet_hours.to_numpy(dtype=float))
    return df

### filter to 1 jan - 31 aug for a year (time when crop grows)
def year_slice(df, year):
    sub = df[(df.time >= f"{year}-01-01") & (df.time <= f"{year}-08-31")].copy()
    if sub.empty:
        return None
    sub["cum_cdd"] = sub.cdd.cumsum()
    return sub

### calibrate the GS31 threshold so that the median across all region-years is the literature date (15 Apr)
def calibrate_gs31_threshold(dailies):
    acc = []
    for region, df in dailies.items():
        for year in range(1971, CALIBRATION_END + 1):
            sub = year_slice(df, year)
            if sub is None:
                continue
            upto = sub[sub.time <= f"{year}-{MEDIAN_GS31_DATE}"]
            if not upto.empty:
                acc.append(upto.cum_cdd.iloc[-1])
    return float(np.median(acc))

### compute the first date in a year when the cumulative degree days exceeds a threshold
def stage_date(sub, threshold):
    hit = sub[sub.cum_cdd >= threshold]
    return None if hit.empty else hit.time.iloc[0]

### given a date window return 6 statistics for that window
def window_stats(sub, start, end, prefix):
    if start is None or end is None or end < start:
        return {f"{prefix}_{k}": np.nan for k in
                ("days", "mean_temp", "total_precip", "mean_rh", "wetness_hours", "splash_risk")}
    w = sub[(sub.time >= start) & (sub.time <= end)]
    if w.empty:
        return {f"{prefix}_days": 0.0,
                **{f"{prefix}_{k}": np.nan for k in
                   ("mean_temp", "total_precip", "mean_rh", "wetness_hours", "splash_risk")}}
    return {
        f"{prefix}_days": float(len(w)),
        f"{prefix}_mean_temp": float(w.temperature_2m_mean.mean()),
        f"{prefix}_total_precip": float(w.precipitation_sum.sum()),
        f"{prefix}_mean_rh": float(w.relative_humidity_2m_mean.mean()),
        f"{prefix}_wetness_hours": float(w.wet_hours.sum()),
        f"{prefix}_splash_risk": float(w.splash.sum()),
    }

### main feature pipeline
def build_features():
    dailies = {}
    for region in REGIONS:
        slug = region.replace(" ", "_")
        if not (Path("analysis/weather_raw") / f"{slug}.json").exists():
            print(f"  {region}: no cached data, skipping")
            continue
        dailies[region] = daily_frame(region)

    gs31_threshold = calibrate_gs31_threshold(dailies)
    print(f"Calibrated GS31 threshold: {gs31_threshold:.0f} degree-days (base {CROP_BASE}C) from 1 Jan, "
          f"set so median GS31 = {MEDIAN_GS31_DATE}")

    sow_map, sow_default = compute_sow_offsets()
    rows = []
    for region, df in dailies.items():
        for year in range(1971, 2027):
            sub = year_slice(df, year)
            if sub is None:
                continue

            ### Drilling-date shift: sow_offset is days from 1 Oct
            sow_offset = sow_map.get((region, year), sow_default)
            shift_cdd = SOW_SENSITIVITY * (sow_offset - sow_default) * sub.cdd.mean()

            thresholds = {"GS31": gs31_threshold + shift_cdd}
            for stage, interval in STAGE_INTERVALS.items():
                thresholds[stage] = list(thresholds.values())[-1] + interval
            dates = {s: stage_date(sub, t) for s, t in thresholds.items()}

            survey = pd.Timestamp(year, *SURVEY_MD)
            row = {"Region": region, "Year": year}
            for stage, d in dates.items():
                row[f"{stage}_doy"] = np.nan if d is None else float(d.dayofyear)

            row["flagleaf_exposure_days"] = (
                np.nan if dates["GS39"] is None else float(max((survey - dates["GS39"]).days, 0))
            )
            ### Where in its development the crop actually is when scored. 
            row["survey_cum_cdd"] = float(sub[sub.time <= survey].cum_cdd.iloc[-1]) if not sub[sub.time <= survey].empty else np.nan
            row["survey_gs_progress"] = (row["survey_cum_cdd"] - thresholds["GS31"]) / (thresholds["GS75"] - thresholds["GS31"])

            # GS-anchored exposure windows.
            row.update(window_stats(sub, dates["GS31"], dates["GS39"], "gs31_39"))   # T1->T2, lower canopy
            row.update(window_stats(sub, dates["GS39"], survey, "gs39_survey"))      # L1 exposure -- the key one
            row.update(window_stats(sub, dates["GS61"], survey, "gs61_survey"))      # post-anthesis

            for k in (14, 30):
                row.update(window_stats(sub, survey - pd.Timedelta(days=k), survey, f"pre_survey_{k}d"))

            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}: {out.shape}")
    return out


if __name__ == "__main__":
    feats = build_features()

    print("\n=== stage date distribution (day-of-year -> calendar date) ===")
    for stage in ["GS31", "GS39", "GS61", "GS75"]:
        s = feats[f"{stage}_doy"].dropna()
        def as_date(doy):
            return (pd.Timestamp("2001-01-01") + pd.Timedelta(days=float(doy) - 1)).strftime("%d %b")
        print(f"  {stage}: median {as_date(s.median())}  "
              f"(10th {as_date(s.quantile(0.1))}, 90th {as_date(s.quantile(0.9))}, "
              f"range {as_date(s.min())}-{as_date(s.max())})")

    print("\n=== sanity check: is the 20 Jun survey between GS61 and GS75? ===")
    gs61 = feats.GS61_doy.dropna()
    gs75 = feats.GS75_doy.dropna()
    survey_doy = pd.Timestamp(2001, *SURVEY_MD).dayofyear
    print(f"  survey doy {survey_doy}: after GS61 in {100 * (gs61 < survey_doy).mean():.0f}% of region-years, "
          f"before GS75 in {100 * (gs75 > survey_doy).mean():.0f}%")
    print("  (expected: high on both -- the ADAS survey is carried out during grain fill)")

    print("\n=== flag-leaf exposure at assessment (days since GS39) ===")
    e = feats.flagleaf_exposure_days.dropna()
    print(f"  median {e.median():.0f} days, range {e.min():.0f}-{e.max():.0f}, sd {e.std():.1f}")
    print("  This spread is the whole point: a fixed calendar window treats the "
          f"{e.min():.0f}-day and {e.max():.0f}-day cases identically.")

    print("\n=== earliest and latest developing years (mean GS39 doy across regions) ===")
    by_year = feats.groupby("Year").GS39_doy.mean().dropna()
    print("  earliest:", ", ".join(f"{y} ({d:.0f})" for y, d in by_year.nsmallest(5).items()))
    print("  latest  :", ", ".join(f"{y} ({d:.0f})" for y, d in by_year.nlargest(5).items()))
