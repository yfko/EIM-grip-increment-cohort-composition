# -*- coding: utf-8 -*-
"""EIM 的增量究竟來自誰？

在全 134 人上，用同一套重複 CV 取得模型③（人口學基準）與⑥（基準+EIM前臂）
的 out-of-fold 預測，再拆到子群比較。並以受試者層級 bootstrap 給出增量的
抽樣區間（修正先前只有「切分間區間」的限制）。唯讀。
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
SEED,N_REPEATS,N_FOLDS,N_BOOT = 20260904,10,5,2000
ALPHAS=np.logspace(-4,2,100)

m=pd.read_excel(f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/完整資料.xlsx"); m["id"]=m["id"].astype(str)
dom=m["慣用手"].astype(str).str.contains("右").map({True:"右",False:"左"})
gc=lambda s:[f"{s}{i}({f})" for f in (50,100) for i in (1,2,3)]
m["grip_dom"]=[m.loc[i,gc(d)].max() for i,d in zip(m.index,dom)]
m["sex_male"]=(m["性別"]=="男").astype(int)
demo=m.set_index("id")[["sex_male","年齡","BMI","grip_dom"]].rename(columns={"年齡":"age"})
a=pd.read_csv(f"{ROOT}/stage1_rebuild/outputs/main_109/cohort_audit.csv",encoding="utf-8-sig",dtype={"id":str}).set_index("id")
demo["healthy"]=a["分析納入"].astype(bool); demo["grip_ok"]=a["握力正常"].astype(bool)
demo["sppb_ok"]=a["SPPB正常"].astype(bool)

feat={}
for p in sorted(glob.glob(f"{STATIC}/*.csv")):
    fq=int(re.search(r"freq(\d+)kHz",os.path.basename(p)).group(1))
    d=pd.read_csv(p,encoding="utf-8-sig",dtype={"subject_id":str}).set_index("subject_id")
    for c in d.columns:
        mm=re.match(r"^(.+?)\((PhA|R|X)\)_Static$",c)
        if mm: feat[f"{mm.group(1)}|{mm.group(2)}|{fq}"]=d[c]
E=pd.DataFrame(feat)
for mu in {c.split("|")[0] for c in E.columns}:
    for fq in (50,100):
        cols=[f"{mu}|{p}|{fq}" for p in ("PhA","R","X") if f"{mu}|{p}|{fq}" in E]
        if len(cols)==3:
            bad=(E[f"{mu}|R|{fq}"]<=0)|(E[f"{mu}|X|{fq}"]<=0)|(E[f"{mu}|PhA|{fq}"]<=0)
            E.loc[bad,cols]=np.nan
df=demo.join(E,how="inner")
FORE=[c for c in E.columns if "腕肌" in c]; DEMO=["sex_male","age","BMI"]

def pipe(s):
    return Pipeline([("imp",SimpleImputer(strategy="median")),("sc",RobustScaler()),
        ("cl",FunctionTransformer(lambda A:np.nan_to_num(A,nan=0.,posinf=0.,neginf=0.))),
        ("las",LassoCV(alphas=ALPHAS,cv=KFold(5,shuffle=True,random_state=s),
                       max_iter=100000,random_state=s,n_jobs=-1))])

def oof(cols):
    """回傳 N_REPEATS × n 的 out-of-fold 預測矩陣。"""
    X=df[cols].to_numpy(float); y=df["grip_dom"].to_numpy(float)
    st=(df["sex_male"].astype(str)+"_"+(df["grip_dom"]>df["grip_dom"].median()).astype(int).astype(str))
    P=np.empty((N_REPEATS,len(y)))
    for r in range(N_REPEATS):
        for tr,te in StratifiedKFold(N_FOLDS,shuffle=True,random_state=SEED+r).split(X,st):
            P[r,te]=pipe(SEED+r).fit(X[tr],y[tr]).predict(X[te])
    return P

print("計算 out-of-fold 預測…")
P3,P6=oof(DEMO),oof(DEMO+FORE)
y=df["grip_dom"].to_numpy(float)
p3,p6=P3.mean(0),P6.mean(0)          # 跨重複平均，降低切分雜訊

groups={"全體 134":np.ones(len(df),bool),
        "健康 109":df["healthy"].to_numpy(),
        "受損 25":~df["healthy"].to_numpy(),
        "  └握力未達 21":~df["grip_ok"].to_numpy(),
        "  └SPPB未達 8":~df["sppb_ok"].to_numpy()}

print(f"\n{'='*76}\n各子群的絕對誤差（out-of-fold，跨 10 次重複平均）\n{'='*76}")
print(f"{'子群':<18}{'n':>5}{'③基準 MAE':>12}{'⑥+EIM MAE':>12}{'改善':>10}{'改善%':>9}")
for lab,msk in groups.items():
    e3,e6=np.abs(y-p3)[msk],np.abs(y-p6)[msk]
    print(f"{lab:<18}{msk.sum():>5}{e3.mean():>12.2f}{e6.mean():>12.2f}{e3.mean()-e6.mean():>+10.2f}{(e3.mean()-e6.mean())/e3.mean():>+9.1%}")

# 受試者層級 bootstrap：ΔR² 與 ΔMAE 的抽樣區間
rng=np.random.default_rng(SEED)
def r2(yy,pp): return 1-((yy-pp)**2).sum()/((yy-yy.mean())**2).sum()
print(f"\n{'='*76}\n受試者層級 bootstrap（{N_BOOT} 次重抽，修正先前只有切分間區間的限制）\n{'='*76}")
for lab,msk in [("全體 134",np.ones(len(df),bool)),("健康 109",df["healthy"].to_numpy()),
                ("受損 25",~df["healthy"].to_numpy())]:
    idx=np.flatnonzero(msk); dr2=[];dmae=[]
    for _ in range(N_BOOT):
        b=rng.choice(idx,len(idx),replace=True)
        dr2.append(r2(y[b],p6[b])-r2(y[b],p3[b]))
        dmae.append(np.abs(y[b]-p3[b]).mean()-np.abs(y[b]-p6[b]).mean())
    dr2=np.array(dr2);dmae=np.array(dmae)
    print(f"{lab:<12} ΔR² {dr2.mean():+.3f} [{np.percentile(dr2,2.5):+.3f},{np.percentile(dr2,97.5):+.3f}]"
          f"   ΔMAE {dmae.mean():+.2f} kg [{np.percentile(dmae,2.5):+.2f},{np.percentile(dmae,97.5):+.2f}]"
          f"   P(Δ>0)={np.mean(dr2>0):.3f}")
