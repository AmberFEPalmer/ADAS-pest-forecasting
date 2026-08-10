"""
Build growing-season weather features (autumn, winter, spring) per region-year
Data from open-Meteo Historical Weather API (ERA5/ERA5-Land reanalysis) https://open-meteo.com/en/docs/historical-weather-api
Converts data into seasonal weather features for each region/year
ERA5 reconstructs historical weather

Crop growing seasons (windows):
  - Autumn: 1 Sep - 30 Nov 
  - Winter: 1 Dec - 28 Feb 
    - Spring: 1 Mar - 31 May 
For each window computes mean temperature, total rainfall, number of rainy days,
average humidity, average wind speed.

"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

### handle paths
RAW_DIR = Path("analysis/weather_raw")
OUT_CSV = "analysis/growing_season_weather.csv"

### dictionary mapping regions to co-ordinates
REGIONS = {
    "East": (52.40, 0.26),
    "East Midlands": (53.08, -0.80),
    "North East": (54.60, -1.60),
    "North West": (53.75, -2.70),
    "South East": (51.20, 0.50),
    "South West": (50.94, -2.63),
    "Wales": (51.45, -3.40),
    "West Midlands": (52.70, -2.30),
    "Yorkshire and The Humber": (53.90, -1.00),
}

### variables requested from the Open-Meteo API, and date range
DAILY_VARS = "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_mean"
START = "1970-08-01"
END = "2026-06-30"
BACKOFF = [60, 180, 420, 900, 1800] ### seconds to wait becuase of rate limiting

### get API request
def _get(url):
    import urllib.error
    import urllib.request

    for wait in BACKOFF:
        try:
            with urllib.request.urlopen(url, timeout=300) as resp: ### open URL
                data = json.load(resp)
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            code = getattr(exc, "code", None)
            if code is not None and code != 429:
                raise
            body = exc.read().decode(errors="replace") if code == 429 else ""
            if "Daily API request limit" in body: ### see whether the quota is exhausted
                raise RuntimeError(
                    "Open-Meteo DAILY request quota is exhausted"
                ) from exc
            print(f"rate-limited/network ({code or getattr(exc, 'reason', '')}), "
                  f"waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        if data.get("error"):
            print(f"    {data.get('reason')}, waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        return data ### return JSON
    raise RuntimeError("gave up after repeated rate-limiting -- re-run to resume")

### build URL for API request
def _url(lat, lon, start, end):
    return (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date={start}&end_date={end}&daily={DAILY_VARS}&timezone=Europe%2FLondon")


def fetch_all():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for region, (lat, lon) in REGIONS.items(): ### loop over each region
        path = RAW_DIR / f"{region.replace(' ', '_')}.json"

        if not path.exists():
            data = _get(_url(lat, lon, START, END))
            path.write_text(json.dumps(data))
            print(f"  {region}: fetched {len(data['daily']['time'])} days", flush=True)
            time.sleep(20)
            continue

        cached = json.loads(path.read_text())
        last = cached["daily"]["time"][-1]
        if last >= END:
            print(f"  {region}: already covers to {last}, skipping")
            continue

        resume = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        new = _get(_url(lat, lon, resume, END))
        for key, vals in new["daily"].items(): ### for every variable append values onto list
            cached["daily"][key].extend(vals)
        assert cached["daily"]["time"] == sorted(cached["daily"]["time"]), f"{region}: dates out of order"
        assert len(set(cached["daily"]["time"])) == len(cached["daily"]["time"]), f"{region}: duplicate dates"
        path.write_text(json.dumps(cached))
        print(f"  {region}: extended {last} -> {cached['daily']['time'][-1]} "
              f"(+{len(new['daily']['time'])} days)", flush=True)
        time.sleep(20)

### load one region's daily data into a DataFrame
def load_daily(region):
    slug = region.replace(" ", "_")
    data = json.loads((RAW_DIR / f"{slug}.json").read_text())
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def window_agg(df, start, end):
    sub = df[(df.time >= start) & (df.time <= end)]
    if sub.empty:
        return None
    return {
        "mean_temp": sub.temperature_2m_mean.mean(),
        "total_precip": sub.precipitation_sum.sum(),
        "rain_days": (sub.precipitation_sum >= 1.0).sum(),
        "mean_rh": sub.relative_humidity_2m_mean.mean(),
        "mean_wind": sub.wind_speed_10m_mean.mean(),
    }

### create final dataset of seasonal weather features for each region-year
def build_features():
    rows = []
    for region in REGIONS:
        slug = region.replace(" ", "_")
        if not (RAW_DIR / f"{slug}.json").exists():
            print(f"  {region}: no cached data yet, skipping (re-run this script later to backfill)")
            continue
        daily = load_daily(region)
        for year in range(1971, 2027):
            windows = {
                "autumn": (f"{year-1}-09-01", f"{year-1}-11-30"),
                "winter": (f"{year-1}-12-01", f"{year}-02-28"),
                "spring": (f"{year}-03-01", f"{year}-05-31"),
                "spring_to_june": (f"{year}-03-01", f"{year}-06-30"),
            }
            row = {"Region": region, "Year": year}
            ok = True
            for wname, (start, end) in windows.items():
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
    if "--parse-only" not in sys.argv:
        fetch_all()
    build_features()
