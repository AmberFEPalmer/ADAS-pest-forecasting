"""
Build three additional weather-derived feature families per region-year, on top
of the plain window aggregates in analysis/growing_season_weather.csv:

    - Thermal time (cumulative degree days above a base temperature)
    - Wetness hours (hours of canopy wetness, from rainfall and humidity)
    - Splash risk (rainfall events on already-wet canopy)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "analysis")
from s02_fetch_weather import REGIONS, load_daily  
from s01_season_reference import daily_wetness_hours 

OUT_CSV = "analysis/thermal_wetness_splash.csv"

THERMAL_BASE = -2.4  
SPLASH_RAIN_THRESHOLD = 2.0  

WINDOWS = {
    "autumn": lambda year: (f"{year-1}-09-01", f"{year-1}-11-30"),
    "winter": lambda year: (f"{year-1}-12-01", f"{year}-02-28"),
    "spring": lambda year: (f"{year}-03-01", f"{year}-05-31"),
    ### both end on the survey-date proxy (s01.END_MD), NOT 30 Jun -- weather after the
    ### assessment cannot have caused it
    "spring_to_june": lambda year: (f"{year}-03-01", f"{year}-06-20"),
    "season_total": lambda year: (f"{year-1}-09-01", f"{year}-06-20"),
}


def daily_splash_risk(precip, wet_hours):
    """precip, wet_hours: same-length daily arrays (chronological). Splash risk on day t
    needs a splash-capable rain event on day t AND an already-wet canopy carried over from
    day t-1 -- see module docstring."""
    wet_prev = np.roll(wet_hours, 1)
    wet_prev[0] = wet_hours[0]  # no day-before-record available; assume unchanged rather than dry
    antecedent_wet_frac = np.clip(wet_prev / 24.0, 0.0, 1.0)
    splash_event_mm = np.where(precip >= SPLASH_RAIN_THRESHOLD, precip, 0.0)
    return splash_event_mm * antecedent_wet_frac


def window_agg(df, start, end):
    sub = df[(df.time >= start) & (df.time <= end)]
    if sub.empty:
        return None
    temp = sub.temperature_2m_mean.to_numpy(dtype=float)
    precip = sub.precipitation_sum.to_numpy(dtype=float)
    rh = sub.relative_humidity_2m_mean.to_numpy(dtype=float)

    gdd = np.clip(temp - THERMAL_BASE, 0, None)
    wet = daily_wetness_hours(precip, rh)
    splash = daily_splash_risk(precip, wet)

    return {
        "thermal_time": gdd.sum(),
        "wetness_hours": wet.sum(),
        "wet_day_frac": float((wet >= 12.0).mean()),  # share of days at least half-wet
        "splash_risk": splash.sum(),
    }


def build_features():
    rows = []
    for region in REGIONS:
        slug = region.replace(" ", "_")
        if not (Path("analysis/weather_raw") / f"{slug}.json").exists():
            print(f"  {region}: no cached data, skipping")
            continue
        daily = load_daily(region)
        for year in range(1971, 2027):
            row = {"Region": region, "Year": year}
            ok = True
            for wname, span_fn in WINDOWS.items():
                start, end = span_fn(year)
                agg = window_agg(daily, start, end)
                if agg is None:
                    ok = False
                    break
                for k, v in agg.items():
                    row[f"{wname}_{k}"] = v
            if ok:
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}: {out.shape}")
    return out


if __name__ == "__main__":
    build_features()
