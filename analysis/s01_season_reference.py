"""
Shared season/wetness reference constants and helpers
"""

import numpy as np
import pandas as pd

START_MD = (10, 1)   # 1 Oct (Year-1) - approximate start of sowing
END_MD = (6, 20)     # 20 Jun (Year) - fixed assessment date proxy
N_DAYS = (pd.Timestamp(2001, *END_MD) - pd.Timestamp(2000, *START_MD)).days  # ~263

### convert each weeks calender label into a number of how many days away from 1Oct is the middle of the week
SOW_WEEK_OFFSETS = {  # days from 1 Oct to the midpoint of each drilling week
    "September 12-18_ag_1": -16,
    "September 19-25_ag_1": -9,
    "September 26 - October 02_ag_1": -2,
    "October 03-09_ag_1": 5,
    "October 10-16_ag_1": 12,
    "October 17-23_ag_1": 19,
    "October 24-30_ag_1": 26,
    "October 31 - November 05_ag_1": 33,
}

### estimate typical planting date per region per year using percentages from survey
def compute_sow_offsets():
    agro = pd.read_csv("data/agronomic_data.csv")
    cols = list(SOW_WEEK_OFFSETS.keys())
    weights = agro[cols].fillna(0.0)
    offsets = np.array([SOW_WEEK_OFFSETS[c] for c in cols])
    wsum = weights.sum(axis=1)
    mean_offset = (weights.to_numpy() * offsets).sum(axis=1) / wsum.replace(0, np.nan)
    default = mean_offset.mean()
    out = pd.DataFrame({"Region": agro.Region, "Year": agro.Year, "sow_offset": mean_offset.fillna(default)})
    return out.set_index(["Region", "Year"])["sow_offset"], default

### estimate how many hrs per day the crop leaves were wet
### rain at least 1mm = assume leaves were wet the whole day
### then look at humidity 
#### above 90% = 24 hrs wet
### 75-90 - scales e.g. 82.5% = 12 hrs wet
def daily_wetness_hours(precip, rh):
    return np.where(precip >= 1.0, 24.0, np.clip((rh - 75.0) / 15.0, 0, 1) * 24.0)
