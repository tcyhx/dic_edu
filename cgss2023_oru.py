# -*- coding: utf-8 -*-
"""
CGSS2023 | ORU（教育错配）建模用清洗脚本
输出：CGSS2023_ORU_clean.csv

功能：
- 读入 CGSS2023.dta
- 清洗收入，生成 ln_income
- a7a -> edu_years（教育年限）
- 自动识别 职业码/行业码（若失败请手工指定）
- 以职业3位（不足则回落到2位 -> 行业 -> 全样本）估计岗位所需教育 R（修剪均值）
- 构造 ORU：R, O, U
- 生成 hukou_ag、rural、gender、age、age2、province 等
"""

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- 基本配置 ----------------
DATA_DTA   = Path("CGSS2023.dta")
OUT_CSV    = Path("CGSS2023_ORU_clean.csv")
SURVEY_YEAR = 2023
DEC = 4

# 顶码/异常值规则（与前述一致）
TOP_CODES = {9_999_996, 9_999_997, 9_999_998, 9_999_999}
MIN_AGE, MAX_AGE = 18, 60

# 估计“所需教育 R”时的层级与最小样本要求
MIN_N_OCC3 = 30     # 职业3位最小样本数
MIN_N_OCC2 = 50     # 职业2位最小样本数
MIN_N_INDU = 60     # 行业大类最小样本数
TRIM_LO, TRIM_HI = 0.10, 0.90   # 修剪均值分位

# ---------------- 工具函数 ----------------
def trimmed_mean(x, lo=TRIM_LO, hi=TRIM_HI):
    """分位修剪均值（无权重）。"""
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) < 5:
        return np.nan
    ql, qh = x.quantile(lo), x.quantile(hi)
    return x[(x >= ql) & (x <= qh)].mean()

def digits_or_nan(val):
    """提取值中的纯数字并返回字符串；若无数字返回 NaN。"""
    if pd.isna(val):
        return np.nan
    s = str(val)
    m = re.findall(r"\d+", s)
    if not m: 
        return np.nan
    return "".join(m)

def occ_k_digits(series, k=3):
    """
    将职业码标准化为 k 位字符串（取前 k 位数字）。不足 k 位或无数字 -> NaN
    """
    s = series.apply(digits_or_nan)
    s = s.astype("object")
    s = s.apply(lambda x: (x[:k] if (isinstance(x, str) and len(x) >= k) else np.nan))
    return s

def find_column(columns, patterns, hint=None):
    """
    在列名中按若干 pattern 正则寻找最匹配的一列；若多匹配取第一；若无匹配返回 None
    patterns: 正则表达式列表（从高到低优先级）
    """
    cols = list(columns)
    low = [c.lower() for c in cols]
    for pat in patterns:
        regex = re.compile(pat)
        for orig, lc in zip(cols, low):
            if regex.search(lc):
                return orig
    if hint:
        warnings.warn(f"未自动识别到 {hint} 列；将返回 None。可在脚本中手工指定。")
    return None

# ---------------- 读取与基础清洗 ----------------
if not DATA_DTA.exists():
    raise FileNotFoundError(f"找不到数据文件：{DATA_DTA.resolve()}")

df = pd.read_stata(DATA_DTA, convert_categoricals=False)
cols = df.columns

# 教育年限：a7a -> edu_years
edu_map = {
    1: 0, 2: 3, 3: 6, 4: 9,
    5:12, 6:12, 7:12, 8:12,
    9:15,10:15,11:16,12:16,
    13:19,14: np.nan
}
df["edu_years"] = df["a7a"].map(edu_map)

# 收入：a8a 清洗 + ln
df = df.loc[df["a8a"] > 0].copy()
a8 = pd.to_numeric(df.get("a8a"), errors="coerce")
mask_inc = a8.notna() & (~a8.isin(TOP_CODES)) & (a8 >= 0) & (a8 <= 9_999_995)
df = df.loc[mask_inc].copy()
df["inc_raw"]   = a8.loc[mask_inc].astype(float)
df = df[df["inc_raw"].notna()].copy()
df["ln_income"] = np.log1p(df["inc_raw"])

# 性别：a2（男=1 女=0）
df["gender"] = np.where(df.get("a2") == 1, 1, np.where(df.get("a2") == 2, 0, np.nan))

# 年龄：a3a（出生年）-> age & age2
byr = pd.to_numeric(df.get("a3a"), errors="coerce")
age = SURVEY_YEAR - byr
age[(age < 10) | (age > 110)] = np.nan
df["age"]  = age
df["age2"] = df["age"] ** 2

# 户籍/居住地/省份
df["hukou"]     = df.get("a18")
df["residence"] = df.get("a25a")
df["province"]  = df.get("s41")

# 户籍农业指示：常见编码 1=农业，2=非农业
df["hukou"] = np.where(df["hukou"] == 1, 1, np.where(df["hukou"] == 2, 0, np.nan))

# 居住地 rural：1/2=城市中心/边缘；3/4=镇/乡
df["residence"] = np.where(df["residence"].isin([3, 4]), 1,
                np.where(df["residence"].isin([1, 2]), 0, np.nan))

# 年龄样本限制（就业年龄段）
df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]

# 权重
df["weight2"] = df.get("weight2")

# ---------------- 自动识别：职业码 / 行业码 ----------------
# 常见候选：职业 occ / occupation / isco / occ_code / a57b / a57c ...
occ_patterns = [
    r"\bocc\b", r"occupation", r"isco", r"occ[_\- ]?code",
    r"\ba57b\b", r"\ba57c\b", r"\ba58c\b", r"\bjob\b"
]
# 常见候选：行业 industry / ind / sector / a57a / a57d ...
ind_patterns = [
    r"industry", r"\bind\b", r"sector", r"\ba57a\b", r"\ba57d\b"
]

occ_col = find_column(cols, occ_patterns, hint="职业代码（occupation）")
ind_col = find_column(cols, ind_patterns, hint="行业代码（industry）")

# 【需要时在这里手工指定列名】（若上面返回 None，请把等号右侧替换成你数据的真实列名）
# 例如：occ_col = "occ_code_3digit"; ind_col = "industry_code"
# occ_col = "你的职业列名"
# ind_col = "你的行业列名"

if occ_col is None:
    warnings.warn("未找到职业列，后续无法计算 R/O/U。请手动指定 occ_col。")
if ind_col is None:
    warnings.warn("未找到行业列，行业回落层级将不可用。可手动指定 ind_col。")

# ---------------- 生成职业层级：occ3 / occ2 ----------------
if occ_col is not None:
    df["occ3"] = occ_k_digits(df[occ_col], k=3)
    df["occ2"] = occ_k_digits(df[occ_col], k=2)
else:
    df["occ3"] = np.nan
    df["occ2"] = np.nan

# 行业层级（可直接用行业大类原码，也可取前2位）
if ind_col is not None:
    ind_digits = df[ind_col].apply(digits_or_nan)
    df["indu2"] = ind_digits.apply(lambda x: (x[:2] if (isinstance(x, str) and len(x) >= 2) else np.nan))
else:
    df["indu2"] = np.nan

# ---------------- 计算 R：岗位所需教育（修剪均值） ----------------
# 按层级估计：occ3 -> occ2 -> indu2 -> overall
R_occ3 = {}
R_occ2 = {}
R_indu = {}
R_all  = trimmed_mean(df["edu_years"])

if occ_col is not None:
    # 3位职业
    tmp3 = (df[["occ3", "edu_years"]]
            .dropna()
            .groupby("occ3")["edu_years"]
            .agg(N="size", R=lambda s: trimmed_mean(s)))
    R_occ3 = tmp3.loc[tmp3["N"] >= MIN_N_OCC3, "R"].to_dict()

    # 2位职业（对3位未覆盖者备用）
    tmp2 = (df[["occ2", "edu_years"]]
            .dropna()
            .groupby("occ2")["edu_years"]
            .agg(N="size", R=lambda s: trimmed_mean(s)))
    R_occ2 = tmp2.loc[tmp2["N"] >= MIN_N_OCC2, "R"].to_dict()

# 行业回落层级
if ind_col is not None:
    tmpi = (df[["indu2", "edu_years"]]
            .dropna()
            .groupby("indu2")["edu_years"]
            .agg(N="size", R=lambda s: trimmed_mean(s)))
    R_indu = tmpi.loc[tmpi["N"] >= MIN_N_INDU, "R"].to_dict()

# 将 R 映射到个体
def map_required_edu(row):
    # 按层级回落：occ3 -> occ2 -> indu2 -> overall
    if isinstance(row.get("occ3"), str) and row["occ3"] in R_occ3:
        return R_occ3[row["occ3"]]
    if isinstance(row.get("occ2"), str) and row["occ2"] in R_occ2:
        return R_occ2[row["occ2"]]
    if isinstance(row.get("indu2"), str) and row["indu2"] in R_indu:
        return R_indu[row["indu2"]]
    return R_all

df["R_required"] = df.apply(map_required_edu, axis=1)

# ---------------- 构造 ORU ----------------
# O = max(0, Edu - R), U = max(0, R - Edu)
df["O_over"] = np.maximum(0, df["edu_years"] - df["R_required"])
df["U_under"] = np.maximum(0, df["R_required"] - df["edu_years"])

# 分类错配（可选）：-1=不足 0=匹配 1=过度
def mismatch_cat(o, u):
    if pd.isna(o) or pd.isna(u):
        return np.nan
    if (o > 0) and (u == 0):
        return 1
    if (u > 0) and (o == 0):
        return -1
    return 0

df["mismatch_cat"] = [mismatch_cat(o, u) for o, u in zip(df["O_over"], df["U_under"])]

# ---------------- 最终样本与导出 ----------------
# ORU建模常在“受雇”/有职业码群体；若你有“就业形态”变量，可在此进一步筛选
# 这里先仅做关键变量非缺失筛选：
need = ["ln_income", "edu_years", "R_required", "O_over", "U_under",
        "gender", "age", "age2", "hukou", "residence", "province", "weight2"]
# 职业/行业用于后续 FE；若存在就保留
maybe = ["occ3", "occ2", ind_col if ind_col else None]
maybe = [c for c in maybe if c is not None]

clean = df.dropna(subset=["ln_income","edu_years","R_required","gender","age","hukou","residence","province","weight2"]).copy()

# 输出列
keep_cols = ([
    "inc_raw", "ln_income",
    "a7a", "edu_years",
    "R_required", "O_over", "U_under", "mismatch_cat",
    "a2", "gender", "a3a", "age", "age2",
    "a18", "hukou", "a25a", "residence",
    "s41", "province", "weight2"
] + maybe)

keep_cols = [c for c in keep_cols if c in clean.columns]
clean = clean[keep_cols].copy()

clean.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# —— 日志输出 —— 
print("=== ORU 清洗完成 ===")
print(f"- 输出文件: {OUT_CSV.resolve()}")
print(f"- 有效样本量: {len(clean):,}")
print(f"- 岗位所需教育（overall 修剪均值）: {np.round(trimmed_mean(df['edu_years']), DEC)} 年")
print("- R 层级可用：",
      f"occ3={len(R_occ3)} | occ2={len(R_occ2)} | indu2={len(R_indu)}")
if occ_col is None:
    print("! 注意：未识别到职业列，R 仅使用行业/总体回落，请手工设置 occ_col 以获得更准确的 R。")
