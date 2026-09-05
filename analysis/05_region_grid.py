# -*- coding: utf-8 -*-
"""部位 × 頻率 網格：每個部位區塊單獨加在人口學基準之上，增量多少？

回答：(1) 除了前臂，其他肌群有沒有訊號？
      (2) 階段一的頻率發現（100 kHz 承載年齡訊號）在這裡能不能用？
      (3) 100/50 頻率比值（研究生的「高低頻比例」）有沒有用？
全 134 人、慣用手最大握力、同一管線。唯讀。
"""
import re, glob, os, warnings, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, StratifiedKFold
warnings.filterwarnings("ignore")
ROOT="/Users/yfko/Library/Mobile Documents/com~apple~CloudDocs/Documents/New idea/EIM"
STATIC=f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/static_output"
SEED,N_REP,N_FOLD,N_BOOT=20260904,5,5,1000
ALPHAS=np.logspace(-4,2,100)

m=pd.read_excel(f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/完整資料.xlsx"); m["id"]=m["id"].astype(str)
dom=m["慣用手"].astype(str).str.contains("右").map({True:"右",False:"左"})
gc=lambda s:[f"{s}{i}({f})" for f in (50,100) for i in (1,2,3)]
m["grip_dom"]=[m.loc[i,gc(d)].max() for i,d in zip(m.index,dom)]
m["sex_male"]=(m["性別"]=="男").astype(int); m["dom"]=dom.values
demo=m.set_index("id")[["sex_male","年齡","BMI","grip_dom","dom"]].rename(columns={"年齡":"age"})

feat={}
for p in sorted(glob.glob(f"{STATIC}/*.csv")):
    fq=int(re.search(r"freq(\d+)kHz",os.path.basename(p)).group(1))
    d=pd.read_csv(p,encoding="utf-8-sig",dtype={"subject_id":str}).set_index("subject_id")
    for c in d.columns:
        mm=re.match(r"^(.+?)\((PhA|R|X)\)_Static$",c)
        if mm: feat[f"{mm.group(1)}|{mm.group(2)}|{fq}"]=d[c]
E=pd.DataFrame(feat)
MUS=sorted({c.split("|")[0] for c in E.columns})
for mu in MUS:
    for fq in (50,100):
        cols=[f"{mu}|{p}|{fq}" for p in ("PhA","R","X")]
        bad=(E[cols[1]]<=0)|(E[cols[2]]<=0)|(E[cols[0]]<=0); E.loc[bad,cols]=np.nan
# 頻率比值 100/50
for mu in MUS:
    for p in ("PhA","R","X"):
        E[f"{mu}|{p}|ratio"]=E[f"{mu}|{p}|100"]/E[f"{mu}|{p}|50"]
df=demo.join(E,how="inner"); DEMO=["sex_male","age","BMI"]

def region(mu):
    if "腕肌" in mu: return "前臂"
    if "肱" in mu: return "上臂"
    if "豎脊" in mu: return "軀幹"
    if "股" in mu: return "大腿"
    return "小腿"
def cols_for(reg,fq,side=None):
    return [c for c in E.columns
            if region(c.split("|")[0])==reg and c.split("|")[2]==str(fq)
            and (side is None or c.startswith(side))]

def pipe(s):
    return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",RobustScaler()),
        ("cl",FunctionTransformer(lambda A:np.nan_to_num(A,nan=0.,posinf=0.,neginf=0.))),
        ("las",LassoCV(alphas=ALPHAS,cv=KFold(5,shuffle=True,random_state=s),max_iter=50000,random_state=s,n_jobs=-1))])
y=df["grip_dom"].to_numpy(float)
strat=(df["sex_male"].astype(str)+"_"+(df["grip_dom"]>df["grip_dom"].median()).astype(int).astype(str))
def oof(cols):
    X=df[cols].to_numpy(float); P=np.empty((N_REP,len(y)))
    for r in range(N_REP):
        for tr,te in StratifiedKFold(N_FOLD,shuffle=True,random_state=SEED+r).split(X,strat):
            P[r,te]=pipe(SEED+r).fit(X[tr],y[tr]).predict(X[te])
    return P.mean(0)
r2=lambda yy,pp:1-((yy-pp)**2).sum()/((yy-yy.mean())**2).sum()
rng=np.random.default_rng(SEED)
def boot(p_base,p_new):
    dr,dm=[],[]
    for _ in range(N_BOOT):
        b=rng.choice(len(y),len(y),replace=True)
        dr.append(r2(y[b],p_new[b])-r2(y[b],p_base[b])); dm.append(np.abs(y[b]-p_base[b]).mean()-np.abs(y[b]-p_new[b]).mean())
    dr,dm=np.array(dr),np.array(dm)
    return (dr.mean(),*np.percentile(dr,[2.5,97.5]),dm.mean(),*np.percentile(dm,[2.5,97.5]),np.mean(dm>0))

pb=oof(DEMO); print(f"基準 sex+age+BMI  R²={r2(y,pb):.3f}  MAE={np.abs(y-pb).mean():.2f}\n")
print(f"{'區塊':<14}{'k':>4}{'R²':>8}{'ΔR² [95%CI]':>26}{'ΔMAE kg [95%CI]':>26}{'P(ΔMAE>0)':>11}")
grid=[]
for reg in ["前臂","上臂","軀幹","大腿","小腿"]:
    for fq in ["50","100","both","ratio"]:
        cols=cols_for(reg,"50")+cols_for(reg,"100") if fq=="both" else cols_for(reg,fq)
        grid.append((f"{reg}・{fq}",cols))
# 前臂只用慣用側（逐人）：以左右兩側各建模後依慣用側取值
grid.append(("前臂・both・全通道+基準參考", None))
for name,cols in grid:
    if cols is None: continue
    p=oof(DEMO+cols); d=boot(pb,p)
    print(f"{name:<14}{len(cols):>4}{r2(y,p):>8.3f}  {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]  {d[3]:+.2f} [{d[4]:+.2f},{d[5]:+.2f}]{d[6]:>11.3f}")
# 額外：全部 5 個部位的 100 kHz 相位角 + 比值（低維、帶階段一先驗）
for name,cols in [("全身・PhA@100",[c for c in E.columns if "|PhA|100" in c]),
                  ("全身・PhA ratio",[c for c in E.columns if "|PhA|ratio" in c]),
                  ("全身・全部@100",[c for c in E.columns if c.endswith("|100")])]:
    p=oof(DEMO+cols); d=boot(pb,p)
    print(f"{name:<14}{len(cols):>4}{r2(y,p):>8.3f}  {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]  {d[3]:+.2f} [{d[4]:+.2f},{d[5]:+.2f}]{d[6]:>11.3f}")
