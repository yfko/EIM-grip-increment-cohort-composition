# -*- coding: utf-8 -*-
"""投影檢定：階段一（109 人）的年齡別參考下限，能否分辨被排除的 25 人？

唯讀。只用階段一既有輸出 + 原始 static CSV。不修改任何檔案。
"""
import csv, glob, os, re, math, statistics as st
from collections import defaultdict

ROOT = "/Users/yfko/Library/Mobile Documents/com~apple~CloudDocs/Documents/New idea/EIM"
STATIC = os.path.join(ROOT, "20260401復刻_投影片0903更新/02_data/raw/static_output")
S1 = os.path.join(ROOT, "stage1_rebuild/outputs/main_109")

def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))

# ── 1. 族群 ──────────────────────────────────────────────
cohort = {}
for r in rd(os.path.join(S1, "cohort_audit.csv")):
    cohort[r["id"]] = dict(
        age=float(r["年齡"]), sex=r["性別"], grip=float(r["握力_最大值"]),
        included=(r["分析納入"] == "True"),
        grip_ok=(r["握力正常"] == "True"),
    )

# ── 2. 模型係數（109 人建立） ─────────────────────────────
Z = 1.65
model = {}
for r in rd(os.path.join(S1, "model_coefficients_and_diagnostics.csv")):
    key = (int(r["frequency_khz"]), r["muscle"])
    model[key] = dict(
        c0=float(r["coef_age^0"]), c1=float(r["coef_age^1"]), c2=float(r["coef_age^2"]),
        sd=float(r["sd_used"]), amin=float(r["age_min"]), amax=float(r["age_max"]),
    )

def lln(key, age):
    m = model[key]
    return math.exp(m["c0"] + m["c1"] * age + m["c2"] * age * age - Z * m["sd"])

# ── 3. 讀 static，取 PhA / R / X ──────────────────────────
obs = defaultdict(dict)          # id -> {(freq, muscle): pha}
for path in sorted(glob.glob(os.path.join(STATIC, "*.csv"))):
    fn = os.path.basename(path)
    freq = int(re.search(r"freq(\d+)kHz", fn).group(1))
    for r in rd(path):
        sid = r["subject_id"]
        for col in r:
            m = re.match(r"^(.+?)\(PhA\)_Static$", col)
            if not m:
                continue
            muscle = m.group(1)
            try:
                pha = float(r[col]); R = float(r[f"{muscle}(R)_Static"]); X = float(r[f"{muscle}(X)_Static"])
            except (ValueError, KeyError):
                continue
            if R <= 0 or X <= 0 or pha <= 0:      # 階段一的有效性過濾
                continue
            if (freq, muscle) in model:
                obs[sid][(freq, muscle)] = pha

# ── 4. 每人低於 LLN 的通道比例 ────────────────────────────
FOREARM = [k for k in model if "腕肌" in k[1]]

def rate(sid, keys):
    ch = [k for k in keys if k in obs.get(sid, {})]
    if not ch:
        return None, 0
    age = cohort[sid]["age"]
    below = sum(1 for k in ch if obs[sid][k] < lln(k, age))
    return below / len(ch), len(ch)

def block(title, keys):
    print(f"\n{'='*62}\n{title}（{len(keys)} 條通道）\n{'='*62}")
    groups = {
        "納入 109（建模族群）": [s for s in cohort if cohort[s]["included"]],
        "排除 25（全部）":      [s for s in cohort if not cohort[s]["included"]],
        "  └ 握力未達 21 人":   [s for s in cohort if not cohort[s]["grip_ok"]],
    }
    for label, ids in groups.items():
        rs = [(s, *rate(s, keys)) for s in ids]
        rs = [(s, r, n) for s, r, n in rs if r is not None]
        vals = [r for _, r, _ in rs]
        anyb = sum(1 for v in vals if v > 0)
        print(f"{label:22s} n={len(vals):3d}  低於LLN通道比例 中位={st.median(vals):6.1%} "
              f"平均={st.mean(vals):6.1%}  至少一條低於者={anyb}/{len(vals)} ({anyb/len(vals):.0%})")

block("全部通道", list(model))
block("僅前臂（尺/橈側深腕肌，與握力最相關）", FOREARM)

# ── 5. 檢定 ───────────────────────────────────────────────
print(f"\n{'='*62}\nMann-Whitney U（納入 109 vs 排除 25）\n{'='*62}")
try:
    from scipy.stats import mannwhitneyu, spearmanr
    for title, keys in [("全部通道", list(model)), ("僅前臂", FOREARM)]:
        a = [rate(s, keys)[0] for s in cohort if cohort[s]["included"]]
        b = [rate(s, keys)[0] for s in cohort if not cohort[s]["included"]]
        a = [x for x in a if x is not None]; b = [x for x in b if x is not None]
        u, p = mannwhitneyu(a, b, alternative="less")
        print(f"  {title:8s} U={u:8.1f}  p={p:.4g}")
    # 低於率 vs 握力（全 134）
    print(f"\n{'='*62}\n低於LLN比例 vs 握力（全 134，Spearman）\n{'='*62}")
    for title, keys in [("全部通道", list(model)), ("僅前臂", FOREARM)]:
        pairs = [(rate(s, keys)[0], cohort[s]["grip"], cohort[s]["sex"]) for s in cohort]
        pairs = [p for p in pairs if p[0] is not None]
        r_, p_ = spearmanr([x[0] for x in pairs], [x[1] for x in pairs])
        line = f"  {title:8s} 全體 rho={r_:+.3f} p={p_:.4g}"
        for sx in ("男", "女"):
            sub = [x for x in pairs if x[2] == sx]
            rs, ps = spearmanr([x[0] for x in sub], [x[1] for x in sub])
            line += f" | {sx}內 rho={rs:+.3f} p={ps:.3g} (n={len(sub)})"
        print(line)
except ImportError:
    print("  scipy 未安裝，略過")
