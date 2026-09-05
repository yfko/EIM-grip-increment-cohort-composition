# -*- coding: utf-8 -*-
"""分部位 + 性別專屬 LLN 的投影檢定。唯讀。"""
import csv, glob, os, re, math, statistics as st
import numpy as np
from collections import defaultdict
from scipy.stats import mannwhitneyu

ROOT="/Users/yfko/Library/Mobile Documents/com~apple~CloudDocs/Documents/New idea/EIM"
STATIC=os.path.join(ROOT,"20260401復刻_投影片0903更新/02_data/raw/static_output")
S1=os.path.join(ROOT,"stage1_rebuild/outputs/main_109")
rd=lambda p: list(csv.DictReader(open(p,encoding="utf-8-sig")))

cohort={r["id"]:dict(age=float(r["年齡"]),sex=r["性別"],grip=float(r["握力_最大值"]),
        inc=(r["分析納入"]=="True"),grip_ok=(r["握力正常"]=="True"),
        sppb_ok=(r["SPPB正常"]=="True")) for r in rd(os.path.join(S1,"cohort_audit.csv"))}

channels=set()
obs=defaultdict(dict)
for path in sorted(glob.glob(os.path.join(STATIC,"*.csv"))):
    freq=int(re.search(r"freq(\d+)kHz",os.path.basename(path)).group(1))
    for r in rd(path):
        for col in r:
            m=re.match(r"^(.+?)\(PhA\)_Static$",col)
            if not m: continue
            mu=m.group(1)
            try:
                pha=float(r[col]);R=float(r[f"{mu}(R)_Static"]);X=float(r[f"{mu}(X)_Static"])
            except (ValueError,KeyError): continue
            if R<=0 or X<=0 or pha<=0: continue
            obs[r["subject_id"]][(freq,mu)]=pha; channels.add((freq,mu))
channels=sorted(channels)

def region(mu):
    if "豎脊肌" in mu: return "軀幹"
    if "腕肌" in mu or "肱二" in mu or "肱三" in mu: return "上肢"
    return "下肢"

Z=1.65
def fit_lln(keys, train_ids):
    """在 train_ids 上重擬合每條通道的二次迴歸，回傳 lln(key,age) 函數。"""
    par={}
    for k in keys:
        xs=[cohort[s]["age"] for s in train_ids if k in obs.get(s,{})]
        ys=[math.log(obs[s][k]) for s in train_ids if k in obs.get(s,{})]
        if len(xs)<10: continue
        c=np.polyfit(xs,ys,2); res=np.array(ys)-np.polyval(c,xs)
        sd=float(np.sqrt((res**2).sum()/(len(xs)-3)))
        par[k]=(c,sd)
    return lambda k,age: math.exp(np.polyval(par[k][0],age)-Z*par[k][1]) if k in par else None

def rate(sid,keys,f):
    ch=[k for k in keys if k in obs.get(sid,{}) and f(k,cohort[sid]["age"]) is not None]
    if not ch: return None
    return sum(1 for k in ch if obs[sid][k]<f(k,cohort[sid]["age"]))/len(ch)

def run(title, train_filter):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(f"{'部位/頻率':<16}{'健康 n':>7}{'中位':>8}{'受損 n':>7}{'中位':>8}{'平均差':>9}{'p':>10}")
    for reg in ["軀幹","上肢","下肢","全部"]:
        for fq in [50,100]:
            keys=[k for k in channels if (reg=="全部" or region(k[1])==reg) and k[0]==fq]
            if not keys: continue
            res={}
            for sx in (["男","女"] if train_filter=="sex" else [None]):
                ids=[s for s in cohort if cohort[s]["inc"] and (sx is None or cohort[s]["sex"]==sx)]
                f=fit_lln(keys,ids)
                pool=[s for s in cohort if sx is None or cohort[s]["sex"]==sx]
                for s in pool:
                    v=rate(s,keys,f)
                    if v is not None: res[s]=v
            a=[v for s,v in res.items() if cohort[s]["inc"]]
            b=[v for s,v in res.items() if not cohort[s]["inc"]]
            if not a or not b: continue
            u,p=mannwhitneyu(a,b,alternative="less")
            print(f"{reg+' '+str(fq)+'kHz':<16}{len(a):>7}{st.median(a):>8.1%}{len(b):>7}{st.median(b):>8.1%}"
                  f"{st.mean(b)-st.mean(a):>+9.1%}{p:>10.3g}")

run("A. 性別合併 LLN（＝階段一的作法，僅重擬合以便同口徑比較）","pooled")
run("B. 性別專屬 LLN（男女各自建模）","sex")
