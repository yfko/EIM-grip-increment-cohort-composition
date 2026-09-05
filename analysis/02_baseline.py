# -*- coding: utf-8 -*-
"""人口學基準 vs EIM：階段二第一個必要對照。

問題：sex+age+BMI 單獨能解釋多少握力變異？EIM 在其之上還增加多少？
2,912 支既有握力腳本中沒有任何一支跑過不含 EIM 的基準模型。

設計：受試者層級重複分層交叉驗證，巢狀 LassoCV 選 alpha。
      所有模型走完全相同的管線（median impute → RobustScaler → LassoCV），
      因此組間差異只來自特徵集。唯讀。
"""
import re, glob, os, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = "/Users/yfko/Library/Mobile Documents/com~apple~CloudDocs/Documents/New idea/EIM"
STATIC = f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/static_output"
SEED, N_REPEATS, N_FOLDS = 20260904, 10, 5
ALPHAS = np.logspace(-4, 2, 100)          # 文獻的固定格點

# ── 資料 ────────────────────────────────────────────────
m = pd.read_excel(f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/完整資料.xlsx")
m["id"] = m["id"].astype(str)
dom = m["慣用手"].astype(str).str.contains("右").map({True: "右", False: "左"})
gcols = lambda side: [f"{side}{i}({f})" for f in (50, 100) for i in (1, 2, 3)]
m["grip_dom"]  = [m.loc[i, gcols(d)].max() for i, d in zip(m.index, dom)]
m["grip_nd"]   = [m.loc[i, gcols("左" if d == "右" else "右")].max() for i, d in zip(m.index, dom)]
m["sex_male"]  = (m["性別"] == "男").astype(int)
demo = m.set_index("id")[["sex_male", "年齡", "BMI", "grip_dom", "grip_nd"]].rename(
    columns={"年齡": "age"})

audit = pd.read_csv(f"{ROOT}/stage1_rebuild/outputs/main_109/cohort_audit.csv",
                    encoding="utf-8-sig", dtype={"id": str}).set_index("id")
demo["included_109"] = audit["分析納入"].astype(bool)   # 此欄已由 pandas 解析為 bool

# ── EIM 特徵 ────────────────────────────────────────────
feat = {}
for path in sorted(glob.glob(f"{STATIC}/*.csv")):
    fq = int(re.search(r"freq(\d+)kHz", os.path.basename(path)).group(1))
    d = pd.read_csv(path, encoding="utf-8-sig", dtype={"subject_id": str}).set_index("subject_id")
    for col in d.columns:
        mm = re.match(r"^(.+?)\((PhA|R|X)\)_Static$", col)
        if mm:
            feat[f"{mm.group(1)}|{mm.group(2)}|{fq}"] = d[col]
E = pd.DataFrame(feat)
# 有效性過濾同階段一：R<=0 或 X<=0 的通道，該通道三個量測一律設為缺失
for mu in {c.split("|")[0] for c in E.columns}:
    for fq in (50, 100):
        cols = [f"{mu}|{p}|{fq}" for p in ("PhA", "R", "X") if f"{mu}|{p}|{fq}" in E]
        if len(cols) == 3:
            bad = (E[f"{mu}|R|{fq}"] <= 0) | (E[f"{mu}|X|{fq}"] <= 0) | (E[f"{mu}|PhA|{fq}"] <= 0)
            E.loc[bad, cols] = np.nan

df = demo.join(E, how="inner")
FOREARM = [c for c in E.columns if "腕肌" in c]
ALL_EIM = list(E.columns)
PHA_ALL = [c for c in E.columns if "|PhA|" in c]
DEMO = ["sex_male", "age", "BMI"]

print(f"受試者 {len(df)}｜EIM 特徵 {len(ALL_EIM)}（前臂 {len(FOREARM)}、僅相位角 {len(PHA_ALL)}）")
print(f"肌肉: {sorted({c.split('|')[0] for c in E.columns})}\n")

MODELS = {
    "① sex 單獨":            ["sex_male"],
    "② sex+age":             ["sex_male", "age"],
    "③ sex+age+BMI ★基準":   DEMO,
    "④ EIM 前臂":             FOREARM,
    "⑤ EIM 全通道":           ALL_EIM,
    "⑥ 基準+EIM前臂":         DEMO + FOREARM,
    "⑦ 基準+EIM全通道":       DEMO + ALL_EIM,
    "⑧ 基準+EIM相位角":       DEMO + PHA_ALL,
}

def make_pipe(seed):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", RobustScaler()),
                     ("clean", FunctionTransformer(lambda A: np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0))),
                     ("las", LassoCV(alphas=ALPHAS, cv=KFold(5, shuffle=True, random_state=seed),
                                     max_iter=100000, random_state=seed, n_jobs=-1))])

def evaluate(sub, cols, y_name):
    """重複分層 CV；回傳每次重複的 out-of-fold R² / MAE。"""
    X, y = sub[cols].to_numpy(float), sub[y_name].to_numpy(float)
    strat = (sub["sex_male"].astype(str) + "_" +
             (sub[y_name] > sub[y_name].median()).astype(int).astype(str))
    r2s, maes = [], []
    for rep in range(N_REPEATS):
        pred = np.empty(len(y))
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
        for tr, te in skf.split(X, strat):
            p = make_pipe(SEED + rep).fit(X[tr], y[tr])
            pred[te] = p.predict(X[te])
        r2s.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        maes.append(np.abs(y - pred).mean())
    return np.array(r2s), np.array(maes)

def report(sub, label, y_name):
    print(f"\n{'='*78}\n{label}｜結果變項：{y_name}｜n={len(sub)}｜"
          f"握力 {sub[y_name].min():.1f}–{sub[y_name].max():.1f} kg (SD {sub[y_name].std():.2f})\n{'='*78}")
    print(f"{'模型':<22}{'特徵數':>6}{'out-of-fold R²':>22}{'MAE (kg)':>16}")
    base = None
    for name, cols in MODELS.items():
        r2, mae = evaluate(sub, cols, y_name)
        if "★" in name: base = r2
        lo, hi = np.percentile(r2, [2.5, 97.5])
        delta = ""
        if base is not None and "★" not in name and ("基準+" in name):
            d = r2 - base
            delta = f"  Δ基準 {d.mean():+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]"
        print(f"{name:<22}{len(cols):>6}{r2.mean():>12.3f} [{lo:.3f},{hi:.3f}]{mae.mean():>12.2f}{delta}")

full = df
h109 = df[df["included_109"]]
import sys
_which = sys.argv[1] if len(sys.argv) > 1 else "ABC"
if "A" in _which: report(full, "A. 全 134 人（完整握力範圍）", "grip_dom")
if "B" in _which: report(h109, "B. 健康 109 人（握力經 AWGS 門檻篩選）", "grip_dom")
if "C" in _which: report(full, "C. 全 134 人・非慣用手", "grip_nd")
