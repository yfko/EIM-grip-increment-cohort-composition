# -*- coding: utf-8 -*-
"""下肢功能結果變項：18 部位的優勢能否在「對的結果變項」上兌現？

握力只有前臂帶訊號；這裡改測步行秒數與五次坐站秒數（對數尺度），
看大腿、小腿、軀幹是否對下肢功能有增量。同一管線、含 4SD 規則。唯讀。
"""
"""：每個部位區塊單獨加在人口學基準之上，增量多少？

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
SEED,N_REP,N_FOLD,N_BOOT=20260904,5,5,1000  # 06 版
ALPHAS=np.logspace(-4,2,100)

m=pd.read_excel(f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/完整資料.xlsx"); m["id"]=m["id"].astype(str)
dom=m["慣用手"].astype(str).str.contains("右").map({True:"右",False:"左"})
gc=lambda s:[f"{s}{i}({f})" for f in (50,100) for i in (1,2,3)]
m["grip_dom"]=[m.loc[i,gc(d)].max() for i,d in zip(m.index,dom)]
m["sex_male"]=(m["性別"]=="男").astype(int); m["dom"]=dom.values
m["gait_s"]=pd.to_numeric(m["步行測試(秒)"],errors="coerce"); m["chair_s"]=pd.to_numeric(m["椅子(秒)"],errors="coerce")
demo=m.set_index("id")[["sex_male","年齡","BMI","grip_dom","gait_s","chair_s","dom"]].rename(columns={"年齡":"age"})

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
# 階段一的雙尾 4SD 規則：以 ln(PhA) 為對象，每條通道獨立，逾界者該通道三個量測一律設缺失
n_out=0
for mu in MUS:
    for fq in (50,100):
        cols=[f"{mu}|{p}|{fq}" for p in ("PhA","R","X")]
        lp=np.log(E[cols[0]]); z=(lp-lp.mean())/lp.std(ddof=1)
        out=z.abs()>4; n_out+=int(out.sum()); E.loc[out,cols]=np.nan
print(f"4SD 離群規則另排除 {n_out} 筆通道觀察值")
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

print(f"椅子(秒)=60 的人數：{int((df['chair_s']>=60).sum())}（疑似上限碼）")
r2=lambda yy,pp:1-((yy-pp)**2).sum()/((yy-yy.mean())**2).sum()
rng=np.random.default_rng(SEED)
def run_outcome(yname,label):
    y=np.log(df[yname].to_numpy(float))
    strat=(df["sex_male"].astype(str)+"_"+(df[yname]>df[yname].median()).astype(int).astype(str))
    def oof(cols):
        X=df[cols].to_numpy(float); P=np.empty((N_REP,len(y)))
        for r in range(N_REP):
            for tr,te in StratifiedKFold(N_FOLD,shuffle=True,random_state=SEED+r).split(X,strat):
                P[r,te]=pipe(SEED+r).fit(X[tr],y[tr]).predict(X[te])
        return P.mean(0)
    def boot(pb,pn):
        dr,dm=[],[]
        for _ in range(N_BOOT):
            b=rng.choice(len(y),len(y),replace=True)
            dr.append(r2(y[b],pn[b])-r2(y[b],pb[b])); dm.append(np.abs(y[b]-pb[b]).mean()-np.abs(y[b]-pn[b]).mean())
        dr,dm=np.array(dr),np.array(dm); return dr.mean(),*np.percentile(dr,[2.5,97.5]),dm.mean(),*np.percentile(dm,[2.5,97.5]),np.mean(dm>0)
    pb=oof(DEMO)
    print(f"\n{'='*96}\n{label}｜log 尺度｜基準 sex+age+BMI R²={r2(y,pb):.3f}  MAE(log)={np.abs(y-pb).mean():.3f}\n{'='*96}")
    print(f"{'區塊':<16}{'k':>4}{'R²':>8}{'ΔR² [95%CI]':>26}{'ΔMAE(log) [95%CI]':>28}{'P(Δ>0)':>9}")
    blocks=[("前臂・100（對照）",cols_for("前臂","100")),
            ("軀幹・100",cols_for("軀幹","100")),("軀幹・50",cols_for("軀幹","50")),
            ("大腿・100",cols_for("大腿","100")),("大腿・50",cols_for("大腿","50")),("大腿・both",cols_for("大腿","50")+cols_for("大腿","100")),
            ("小腿・100",cols_for("小腿","100")),("小腿・50",cols_for("小腿","50")),("小腿・both",cols_for("小腿","50")+cols_for("小腿","100")),
            ("大腿+小腿・100",cols_for("大腿","100")+cols_for("小腿","100")),
            ("下肢+軀幹・100",cols_for("大腿","100")+cols_for("小腿","100")+cols_for("軀幹","100")),
            ("下肢+軀幹 PhA@100",[c for c in E.columns if "|PhA|100" in c and region(c.split("|")[0]) in ("大腿","小腿","軀幹")])]
    for name,cols in blocks:
        p=oof(DEMO+cols); d=boot(pb,p)
        print(f"{name:<16}{len(cols):>4}{r2(y,p):>8.3f}  {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]  {d[3]:+.4f} [{d[4]:+.4f},{d[5]:+.4f}]{d[6]:>9.3f}")
run_outcome("gait_s","步行測試秒數（4 m）")
run_outcome("chair_s","五次坐站秒數")
