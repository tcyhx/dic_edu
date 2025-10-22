# -*- coding: utf-8 -*-
"""
CGSS 2021 data cleaning script
- Input : CGSS2021_converted.csv
- Output: CGSS2021_eud_cleaned.csv

Implements variable definitions per variables.docx:
  ln_income  = ln(A8a + 1)
  edu_years  = years of schooling derived from A7a (0–19)
  edu_std    = z-score of edu_years
  int_use    = A28_5 (1=Never ... 5=Daily)
  int_std    = z-score of int_use
  edu_int_interact = edu_std * int_std
  gender     = A2 (1=Male, 0=Female)
  age        = 2021 - A3_1
  urban      = 1 Urban, 0 Rural (auto-detected from isurban if present or existing binary)
  health     = A15 (1=Very poor ... 5=Very good)
  hukou      = from A18, 1 if non-agricultural (>=2), else 0
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------
# 0) Paths
# -------------------------
INFILE  = "CGSS2021_converted.csv"
OUTFILE = "CGSS2021_eud_cleaned.csv"

# -------------------------
# 1) Load data
# -------------------------
df = pd.read_csv(INFILE, low_memory=False)

# -------------------------
# 2) Standard missing/invalid codes → NaN
#    (common CGSS placeholders)
# -------------------------
INVALID = {
    96, 97, 98, 99,
    996, 997, 998, 999,
    9996, 9997, 9998, 9999,
    99996, 99997, 99998, 99999,
    999996, 999997, 999998, 999999
}
def _nanize(series):
    try:
        s = pd.to_numeric(series, errors="coerce")
    except Exception:
        s = series
    return s.mask(s.isin(INVALID))

for col in ["A8a","A7a","A28_5","A2","A3_1","isurban","A15","A18"]:
    if col in df.columns:
        df[col] = _nanize(df[col])

# -------------------------
# 3) Construct variables
# -------------------------

# 3.1 ln_income = ln(A8a + 1)
if "A8a" not in df.columns:
    raise KeyError("A8a (annual personal income) not found in the CSV.")
df["ln_income"] = np.log(pd.to_numeric(df["A8a"], errors="coerce").fillna(0) + 1)

# 3.2 edu_years from A7a (highest education level) → years 0–19
# Mapping below follows common CGSS recodes; adjust if your A7a coding differs.
# If A7a already contains years, we will clip into [0,19].
edu_map = {
    # If A7a is categorical codes, map to years; extend as needed
    # (examples)  0/None: no schooling
    0: 0, 1: 6, 2: 9, 3: 12, 4: 15, 5: 16, 6: 19
}
def derive_edu_years(s):
    s_num = pd.to_numeric(s, errors="coerce")
    # attempt to map category codes; if few unique codes, use map; else treat as years directly
    uniq = s_num.dropna().unique()
    if np.isin(uniq, list(edu_map.keys())).all():
        out = s_num.map(edu_map)
    else:
        # treat as years if it's plausibly numeric; clip to [0,19]
        out = s_num.clip(lower=0, upper=19)
    return out

if "A7a" not in df.columns:
    raise KeyError("A7a (highest education level) not found in the CSV.")
df["edu_years"] = derive_edu_years(df["A7a"])

# 3.3 Standardize education (z-score)
def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    m = x.mean()
    s = x.std(ddof=0)
    return (x - m) / s if s and not np.isclose(s, 0) else pd.Series(np.zeros(len(x)), index=x.index)

df["edu_std"] = zscore(df["edu_years"])

# 3.4 Internet use
if "A28_5" not in df.columns:
    raise KeyError("A28_5 (Internet use frequency) not found in the CSV.")
df["int_use"] = pd.to_numeric(df["A28_5"], errors="coerce")
df["int_std"] = zscore(df["int_use"])

# 3.5 Interaction: edu_std × int_std
df["edu_int_interact"] = df["edu_std"] * df["int_std"]

# 3.6 Gender: A2 (1=Male, 0=Female) per variables.docx
if "A2" not in df.columns:
    raise KeyError("A2 (gender) not found in the CSV.")
# Ensure binary coding conforms to spec
gender = pd.to_numeric(df["A2"], errors="coerce")
# If values are 1/2 (1=Male, 2=Female) convert to 1/0; if already binary 1/0 keep
if set(gender.dropna().unique()).issubset({0,1}):
    df["gender"] = gender  # already 1/0
else:
    # common CGSS: 1=Male, 2=Female → keep Male=1, Female=0
    df["gender"] = np.where(gender == 1, 1,
                     np.where(gender == 2, 0, np.nan))

# 3.7 Age: 2021 - A3_1
if "A3_1" not in df.columns:
    raise KeyError("A3_1 (birth year) not found in the CSV.")
df["age"] = 2021 - pd.to_numeric(df["A3_1"], errors="coerce")

# 3.8 Urban: 1 Urban, 0 Rural
# If 'isurban' exists and uses {1=Urban,2=Rural}, recode; if already 0/1, keep.
if "isurban" in df.columns:
    urb = pd.to_numeric(df["isurban"], errors="coerce")
    if set(urb.dropna().unique()).issubset({0,1}):
        df["residence"] = urb
    else:
        df["residence"] = np.where(urb == 1, 1,
                        np.where(urb == 2, 0, np.nan))
else:
    # If no isurban column provided, create empty and allow NaN
    df["residence"] = np.nan

# 3.9 Health: A15 (1–5)
df["health"] = pd.to_numeric(df["A15"], errors="coerce") if "A15" in df.columns else np.nan

# 3.10 Hukou: from A18 → 1 if non-agricultural (>=2) else 0
if "A18" in df.columns:
    a18 = pd.to_numeric(df["A18"], errors="coerce")
    df["hukou"] = np.where(a18 >= 2, 1,
                    np.where(a18 < 2, 0, np.nan))
else:
    df["hukou"] = np.nan

# -------------------------
# 4) Keep analysis variables & drop rows with missing core fields
#    (per variables.docx core specs)
# -------------------------
core = [
    "ln_income",       # outcome
    "edu_years","edu_std",
    "int_use","int_std",
    "edu_int_interact",
    "gender","age","residence","health","hukou"
]
existing = [c for c in core if c in df.columns]
clean = df[existing].copy()

# Listwise deletion on essential variables for the model
essential = ["ln_income","edu_std","int_std","edu_int_interact","gender","age","residence","health","hukou"]
essential = [c for c in essential if c in clean.columns]
clean = clean.dropna(subset=essential)

# (Optional) Reasonable age bounds (comment out if not desired)
clean = clean[(clean["age"] >= 16) & (clean["age"] <= 89)]

# -------------------------
# 5) Save
# -------------------------
Path(OUTFILE).parent.mkdir(parents=True, exist_ok=True)
clean.to_csv(OUTFILE, index=False)

print(f"✅ Cleaned dataset saved to: {OUTFILE}")
print("Columns:", list(clean.columns))
print("N =", len(clean))
