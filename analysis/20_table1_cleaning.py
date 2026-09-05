# -*- coding: utf-8 -*-
"""R10c 回溯用：Table 1 描述統計與資料清理計數（4,824 → 4,791；無效值、4SD、插補格）。唯讀。
載入與清理規則與 11_models.py 完全相同。"""
import re, glob, os, numpy as np, pandas as pd
ROOT="/Users/yfko/Library/Mobile Documents/com~apple~CloudDocs/Documents/New idea/EIM"
STATIC=f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/static_output"
m=pd.read_excel(f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/完整資料.xlsx"); m["id"]=m["id"].astype(str)
dom=m["慣用手"].astype(str).str.contains("右").map({True:"右",False:"左"})
gc=lambda s:[f"{s}{i}({f})" for f in (50,100) for i in (1,2,3)]
m["grip_L"]=m[gc("左")].max(axis=1); m["grip_R"]=m[gc("右")].max(axis=1)
m["grip_dom"]=np.where(dom=="右",m["grip_R"],m["grip_L"])
m["sex_male"]=(m["性別"]=="男").astype(int)
a=pd.read_csv(f"{ROOT}/stage1_rebuild/outputs/main_109/cohort_audit.csv",encoding="utf-8-sig",dtype={"id":str}).set_index("id")
d=m.set_index("id").join(a,rsuffix="_audit")
d["healthy"]=a["分析納入"].astype(bool)
d["weak"]=~a["握力正常"].astype(bool)
print("cohort_audit 欄位：",list(a.columns))
sarc=["SARC-F"]; sppb=["得分"]  # 得分 = SPPB 總分（cohort_audit 的 SPPB正常 由此判定）
print("SARC-F 欄位候選：",sarc,"SPPB 欄位候選：",sppb)
def desc(g,name):
    print(f"\n[{name}] n={len(g)}")
    print(f"  女 n(%)：{int((g.sex_male==0).sum())} ({100*(g.sex_male==0).mean():.0f})")
    print(f"  年齡 mean(SD); range：{g['年齡'].mean():.1f} ({g['年齡'].std(ddof=1):.1f}); {g['年齡'].min():.0f}–{g['年齡'].max():.0f}；median {g['年齡'].median():.0f}")
    print(f"  ≥65 n：{int((g['年齡']>=65).sum())}")
    print(f"  BMI mean(SD)：{g['BMI'].mean():.1f} ({g['BMI'].std(ddof=1):.1f})")
    print(f"  慣用手握力 mean(SD); range：{g.grip_dom.mean():.1f} ({g.grip_dom.std(ddof=1):.1f}); {g.grip_dom.min():.1f}–{g.grip_dom.max():.1f}")
    for c in sppb[:1]:
        s=pd.to_numeric(g[c],errors="coerce"); print(f"  SPPB median(IQR)：{s.median():.0f} ({s.quantile(.25):.0f}–{s.quantile(.75):.0f})；<10 n：{int((s<10).sum())}")
    for c in sarc[:1]:
        s=pd.to_numeric(g[c],errors="coerce"); print(f"  SARC-F ≥4 n：{int((s>=4).sum())}")
    print(f"  握力低於 AWGS n：{int(g.weak.sum())}")
desc(d,"All"); desc(d[d.healthy],"Healthy"); desc(d[~d.healthy],"Impaired")
imp=d[~d.healthy]
if sppb and sarc:
    s=pd.to_numeric(imp[sppb[0]],errors="coerce"); f=pd.to_numeric(imp[sarc[0]],errors="coerce")
    print(f"\n受損者判準重疊：僅握力 {int((imp.weak&(s>=10)&(f<4)).sum())}；握力+SPPB {int((imp.weak&(s<10)).sum())}；三項任一 {len(imp)}")
# ── 清理計數 ──
feat={}
for p in sorted(glob.glob(f"{STATIC}/*.csv")):
    fq=int(re.search(r"freq(\d+)kHz",os.path.basename(p)).group(1))
    t=pd.read_csv(p,encoding="utf-8-sig",dtype={"subject_id":str}).set_index("subject_id")
    for c in t.columns:
        mm=re.match(r"^(.+?)\((PhA|R|X)\)_Static$",c)
        if mm: feat[f"{mm.group(1)}|{mm.group(2)}|{fq}"]=t[c]
E=pd.DataFrame(feat).loc[d.index]; MUS=sorted({c.split("|")[0] for c in E.columns})
total=len(E)*len(MUS)*2; n_bad=0; bad_ids=set(); n_out=0; present=0
for mu in MUS:
    for fq in (50,100):
        cols=[f"{mu}|{p}|{fq}" for p in ("PhA","R","X")]
        present+=int(E[cols].notna().all(axis=1).sum())
        bad=(E[cols[1]]<=0)|(E[cols[2]]<=0)|(E[cols[0]]<=0); n_bad+=int(bad.sum()); bad_ids|=set(E.index[bad]); E.loc[bad,cols]=np.nan
        lp=np.log(E[cols[0]]); z=(lp-lp.mean())/lp.std(ddof=1); out=z.abs()>4; n_out+=int(out.sum()); E.loc[out,cols]=np.nan
print(f"\n通道觀測總數 {len(E)}×{len(MUS)}×2 = {total}；原始缺失前有效 {present}")
print(f"非正值（R/X/PhA ≤ 0）移除 {n_bad} 筆，來自 {len(bad_ids)} 位受試者：{sorted(bad_ids)}；其中受損者 {sum(1 for i in bad_ids if not d.loc[i,'healthy'])}")
print(f"4SD 移除 {n_out} 筆；剩餘 {total-n_bad-n_out}")
fore100=[c for c in E.columns if "腕肌" in c and c.split("|")[1] in ("R","X") and c.endswith("|100")]
print(f"前臂·100 kHz (R,X) 特徵 {len(fore100)}；134×{len(fore100)} 格中缺失 {int(E[fore100].isna().sum().sum())} 格（訓練折內中位數插補）")
