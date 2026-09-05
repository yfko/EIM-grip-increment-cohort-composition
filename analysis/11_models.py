# -*- coding: utf-8 -*-
"""預註冊模型組（10_預註冊.md §5）。開發集 134 人。唯讀。

所有層共用：全 134、慣用手握力、(R,X) 去冗餘、4SD、受試者層級 10×5 折分層 CV、
FWL 兩階段殘差化（只在訓練折擬合）、ΔMAE 受試者 bootstrap。
"""
import re, glob, os, sys, warnings, json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, FunctionTransformer
from sklearn.linear_model import LassoCV, ElasticNetCV, LinearRegression, LogisticRegressionCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from group_lasso import GroupLasso
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
try:
    from xgboost import XGBRegressor; XGB_BACKEND="xgboost"
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor; XGB_BACKEND="sklearn-HGB"

ROOT="/Users/yfko/Library/Mobile Documents/com~apple~CloudDocs/Documents/New idea/EIM"
STATIC=f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/static_output"
OUT=f"{ROOT}/stage2_audit"
SEED,N_REP,N_REP_TREE,N_FOLD,N_BOOT=20260904,10,5,5,2000
TIERS=sys.argv[1] if len(sys.argv)>1 else "0,1a,1b,1c,2a,2b,2c,3,4"

# ── 資料 ──────────────────────────────────────────────────────────────
m=pd.read_excel(f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/完整資料.xlsx"); m["id"]=m["id"].astype(str)
dom=m["慣用手"].astype(str).str.contains("右").map({True:"右",False:"左"})
gc=lambda s:[f"{s}{i}({f})" for f in (50,100) for i in (1,2,3)]
m["grip_L"]=m[gc("左")].max(axis=1); m["grip_R"]=m[gc("右")].max(axis=1)
m["grip_dom"]=np.where(dom=="右",m["grip_R"],m["grip_L"]); m["grip_nd"]=np.where(dom=="右",m["grip_L"],m["grip_R"])
m["sex_male"]=(m["性別"]=="男").astype(int)
demo=m.set_index("id")[["sex_male","年齡","BMI","grip_dom","grip_nd","grip_L","grip_R"]].rename(columns={"年齡":"age"})
a=pd.read_csv(f"{ROOT}/stage1_rebuild/outputs/main_109/cohort_audit.csv",encoding="utf-8-sig",dtype={"id":str}).set_index("id")
demo["weak"]=(~a["握力正常"].astype(bool)).astype(int)

feat={}
for p in sorted(glob.glob(f"{STATIC}/*.csv")):
    fq=int(re.search(r"freq(\d+)kHz",os.path.basename(p)).group(1))
    d=pd.read_csv(p,encoding="utf-8-sig",dtype={"subject_id":str}).set_index("subject_id")
    for c in d.columns:
        mm=re.match(r"^(.+?)\((PhA|R|X)\)_Static$",c)
        if mm: feat[f"{mm.group(1)}|{mm.group(2)}|{fq}"]=d[c]
E=pd.DataFrame(feat); MUS=sorted({c.split("|")[0] for c in E.columns}); n_out=0
for mu in MUS:
    for fq in (50,100):
        cols=[f"{mu}|{p}|{fq}" for p in ("PhA","R","X")]
        bad=(E[cols[1]]<=0)|(E[cols[2]]<=0)|(E[cols[0]]<=0); E.loc[bad,cols]=np.nan
        lp=np.log(E[cols[0]]); z=(lp-lp.mean())/lp.std(ddof=1); out=z.abs()>4; n_out+=int(out.sum()); E.loc[out,cols]=np.nan
E=E[[c for c in E.columns if c.split("|")[1] in ("R","X")]]        # 去冗餘：只留 R、X
df=demo.join(E,how="inner"); DEMO=["sex_male","age","BMI"]
def region(mu):
    if "腕肌" in mu: return "前臂"
    if "肱" in mu: return "上臂"
    if "豎脊" in mu: return "軀幹"
    if "股" in mu: return "大腿"
    return "小腿"
FORE100=[c for c in E.columns if region(c.split("|")[0])=="前臂" and c.endswith("|100")]
GROUPS=[(r,f) for r in ["前臂","上臂","軀幹","大腿","小腿"] for f in ("50","100")]
ALLCOLS=[c for g in GROUPS for c in E.columns if region(c.split("|")[0])==g[0] and c.split("|")[2]==g[1]]
GID=np.array([GROUPS.index((region(c.split("|")[0]),c.split("|")[2])) for c in ALLCOLS])
y=df["grip_dom"].to_numpy(float); n=len(y)
strat=(df["sex_male"].astype(str)+"_"+(df["grip_dom"]>df["grip_dom"].median()).astype(int).astype(str)).to_numpy()
print(f"開發集 n={n}｜4SD 排除 {n_out} 筆｜前臂・100 (R,X) {len(FORE100)} 特徵｜SGL 全群 {len(ALLCOLS)} 特徵 / {len(GROUPS)} 群｜GBM 後端 {XGB_BACKEND}\n")

# ── FWL 兩階段 ───────────────────────────────────────────────────────
def fwl_split(tr,te,cols):
    """回傳 (Xstar_tr, Xstar_te, ystar_tr, pred_demo_te, D_tr, D_te)。殘差化只在訓練折擬合。"""
    Dm=df[DEMO].to_numpy(float); X=df[cols].to_numpy(float)
    imp=SimpleImputer(strategy="median").fit(X[tr]); Xtr,Xte=imp.transform(X[tr]),imp.transform(X[te])
    ols_y=LinearRegression().fit(Dm[tr],y[tr]); ystar=y[tr]-ols_y.predict(Dm[tr]); pdemo=ols_y.predict(Dm[te])
    ols_x=LinearRegression().fit(Dm[tr],Xtr); Xs_tr=Xtr-ols_x.predict(Dm[tr]); Xs_te=Xte-ols_x.predict(Dm[te])
    return Xs_tr,Xs_te,ystar,pdemo,Dm[tr],Dm[te]
def inner(s): return KFold(5,shuffle=True,random_state=s)
def splits(rep): return StratifiedKFold(N_FOLD,shuffle=True,random_state=SEED+rep).split(np.zeros(n),strat)

def oof_baseline(reps):
    P=np.empty((reps,n))
    for r in range(reps):
        for tr,te in splits(r):
            Dm=df[DEMO].to_numpy(float); P[r,te]=LinearRegression().fit(Dm[tr],y[tr]).predict(Dm[te])
    return P.mean(0)

def oof_model(make,cols,reps,with_demo=False,record=None):
    P=np.empty((reps,n))
    for r in range(reps):
        for tr,te in splits(r):
            Xs_tr,Xs_te,ystar,pdemo,Dtr,Dte=fwl_split(tr,te,cols)
            if with_demo: Xs_tr=np.hstack([Xs_tr,Dtr]); Xs_te=np.hstack([Xs_te,Dte])
            mdl=make(SEED+r); mdl.fit(Xs_tr,ystar)
            P[r,te]=pdemo+np.ravel(mdl.predict(Xs_te))
            if record is not None: record(mdl)
    return P.mean(0)

def oof_lasso_raw(reps):
    cols=DEMO+FORE100; X=df[cols].to_numpy(float); P=np.empty((reps,n))
    for r in range(reps):
        for tr,te in splits(r):
            pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",RobustScaler()),
                ("cl",FunctionTransformer(lambda A:np.nan_to_num(A,nan=0.,posinf=0.,neginf=0.))),
                ("las",LassoCV(alphas=np.logspace(-4,2,100),cv=inner(SEED+r),max_iter=100000,random_state=SEED+r))])
            P[r,te]=pipe.fit(X[tr],y[tr]).predict(X[te])
    return P.mean(0)

r2=lambda yy,pp:1-((yy-pp)**2).sum()/((yy-yy.mean())**2).sum()
rng=np.random.default_rng(SEED)
def boot_delta(pb,pn,yy=None,idx=None):
    yy=y if yy is None else yy; idx=np.arange(len(yy)) if idx is None else idx
    dm,dr=[],[]
    for _ in range(N_BOOT):
        b=rng.choice(idx,len(idx),replace=True)
        dm.append(np.abs(yy[b]-pb[b]).mean()-np.abs(yy[b]-pn[b]).mean()); dr.append(r2(yy[b],pn[b])-r2(yy[b],pb[b]))
    dm,dr=np.array(dm),np.array(dr)
    return dict(dmae=dm.mean(),dmae_lo=np.percentile(dm,2.5),dmae_hi=np.percentile(dm,97.5),p_pos=float(np.mean(dm>0)),
                dr2=dr.mean(),dr2_lo=np.percentile(dr,2.5),dr2_hi=np.percentile(dr,97.5))
def row(name,k,p,pb):
    d=boot_delta(pb,p)
    print(f"{name:<26}{k:>4}{r2(y,p):>8.3f}{np.abs(y-p).mean():>8.2f}  ΔMAE {d['dmae']:+.2f} [{d['dmae_lo']:+.2f},{d['dmae_hi']:+.2f}] P={d['p_pos']:.3f}  ΔR² {d['dr2']:+.3f} [{d['dr2_lo']:+.3f},{d['dr2_hi']:+.3f}]")
    return d

# ── 執行 ─────────────────────────────────────────────────────────────
PRED={}; RES={}
_prev=f"{OUT}/11_models_oof.npz"
if os.path.exists(_prev):
    _z=np.load(_prev,allow_pickle=True); PRED.update({k:_z[k] for k in _z.files if k not in ("y","ids")}); print(f"（續跑：載入既有 {len(PRED)} 組 out-of-fold 預測）")
_prevj=f"{OUT}/11_models_results.json"
if os.path.exists(_prevj): RES.update(json.load(open(_prevj)))
pb=oof_baseline(N_REP); PRED["baseline"]=pb
print(f"{'='*118}\n人口學基準 sex+age+BMI   R²={r2(y,pb):.3f}  MAE={np.abs(y-pb).mean():.2f} kg\n{'='*118}")
print(f"{'層／模型':<26}{'k':>4}{'R²':>8}{'MAE':>8}")

if "0" in TIERS.split(","):
    p=oof_lasso_raw(N_REP); PRED["0_lasso_raw"]=p; RES["0"]=row("0  Lasso 原始（文獻流程）",len(FORE100)+3,p,pb)
if "1a" in TIERS.split(","):
    mk=lambda s:Pipeline([("sc",StandardScaler()),("en",ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9],n_alphas=100,cv=inner(s),max_iter=50000,random_state=s))])
    p=oof_model(mk,FORE100,N_REP); PRED["1a_enet"]=p; RES["1a"]=row("1a Elastic net ★H1 主模型",len(FORE100),p,pb)
if "1b" in TIERS.split(","):
    sel=[]
    GR=np.logspace(-3,0,5); LR=np.logspace(-3,0,5)
    class SGLManual:
        """手寫內層 5 折網格（group_lasso 1.5 與 sklearn 1.6 GridSearchCV 不相容）。"""
        def __init__(self,seed): self.seed=seed
        def _fit_one(self,X,y_,g,l):
            gl=GroupLasso(groups=GID,group_reg=g,l1_reg=l,scale_reg="group_size",supress_warning=True,n_iter=3000,tol=1e-4,fit_intercept=True,frobenius_lipschitz=True)
            gl.fit(X,y_); return gl
        def fit(self,X,y_):
            self.sc=StandardScaler().fit(X); Xs=self.sc.transform(X); best=(np.inf,None)
            for g in GR:
                for l in LR:
                    errs=[]
                    for itr,ite in inner(self.seed).split(Xs):
                        try:
                            gl=self._fit_one(Xs[itr],y_[itr],g,l); p=np.ravel(gl.predict(Xs[ite]))
                            errs.append(np.abs(y_[ite]-p).mean() if np.all(np.isfinite(p)) else np.inf)
                        except Exception: errs.append(np.inf)
                    e=np.mean(errs)
                    if e<best[0]: best=(e,(g,l))
            self.best_=best[1]; self.gl=self._fit_one(Xs,y_,*self.best_); return self
        def predict(self,X): return np.ravel(self.gl.predict(self.sc.transform(X)))
    def rec(mdl):
        nz=np.flatnonzero(np.abs(np.ravel(mdl.gl.coef_))>1e-8); sel.append(sorted({GROUPS[g] for g in GID[nz]}))
    p=oof_model(lambda s:SGLManual(s),ALLCOLS,N_REP,record=rec); PRED["1b_sgl"]=p; RES["1b"]=row("1b Sparse-group lasso 10群",len(ALLCOLS),p,pb)
    from collections import Counter
    cnt=Counter(g for s_ in sel for g in s_); tot=len(sel)
    print("   各群被選入比例（跨 %d 個外層折）："%tot+"；".join(f"{r}・{f} {cnt[(r,f)]/tot:.0%}" for r,f in GROUPS))
    print("   全空模型比例：%.0f%%"%(100*sum(1 for s_ in sel if not s_)/tot))
if "1c" in TIERS.split(","):
    class PLSW(PLSRegression):
        def predict(self,X): return np.ravel(super().predict(X))
    mk=lambda s:GridSearchCV(Pipeline([("sc",StandardScaler()),("pls",PLSW())]),{"pls__n_components":list(range(1,7))},cv=inner(s),scoring="neg_mean_absolute_error")
    p=oof_model(mk,FORE100,N_REP); PRED["1c_pls"]=p; RES["1c"]=row("1c PLS",len(FORE100),p,pb)
XGRID=dict(max_depth=[2,3],learning_rate=[.03,.1],n_estimators=[100,300],min_child_weight=[3,6],reg_lambda=[1,10])
def mk_xgb(s,mono=None):
    if XGB_BACKEND=="xgboost":
        base=XGBRegressor(subsample=.8,colsample_bytree=.8,random_state=s,n_jobs=4,verbosity=0,**({"monotone_constraints":mono} if mono else {}))
        return GridSearchCV(base,XGRID,cv=inner(s),scoring="neg_mean_absolute_error",n_jobs=2)
    g=dict(max_depth=[2,3],learning_rate=[.03,.1],max_iter=[100,300],min_samples_leaf=[3,6],l2_regularization=[1,10])
    base=HistGradientBoostingRegressor(random_state=s,**({"monotonic_cst":list(mono)} if mono else {}))
    return GridSearchCV(base,g,cv=inner(s),scoring="neg_mean_absolute_error",n_jobs=-1)
MONO=tuple(-1 if c.split("|")[1]=="R" else 1 for c in FORE100)+(0,0,0)
if "2a" in TIERS.split(","):
    p=oof_model(lambda s:mk_xgb(s),FORE100,N_REP_TREE,with_demo=True); PRED["2a_xgb"]=p; RES["2a"]=row(f"2a {XGB_BACKEND}",len(FORE100)+3,p,pb)
if "2b" in TIERS.split(","):
    p=oof_model(lambda s:mk_xgb(s,MONO),FORE100,N_REP_TREE,with_demo=True); PRED["2b_xgb_mono"]=p; RES["2b"]=row(f"2b {XGB_BACKEND} 單調約束",len(FORE100)+3,p,pb)
if "2c" in TIERS.split(","):
    mk=lambda s:GridSearchCV(Pipeline([("sc",StandardScaler()),("kr",KernelRidge(kernel="rbf"))]),{"kr__alpha":[.1,1,10],"kr__gamma":[.01,.1,1]},cv=inner(s),scoring="neg_mean_absolute_error")
    p=oof_model(mk,FORE100,N_REP); PRED["2c_krr"]=p; RES["2c"]=row("2c Kernel ridge RBF",len(FORE100),p,pb)

# 決策規則：2a vs 1a
if "1a_enet" in PRED and "2a_xgb" in PRED:
    dm=[]
    for _ in range(N_BOOT):
        b=rng.choice(n,n,replace=True)
        dm.append(np.abs(y[b]-PRED["1a_enet"][b]).mean()-np.abs(y[b]-PRED["2a_xgb"][b]).mean())
    dm=np.array(dm); print(f"\n決策規則  2a 相對 1a 的 MAE 改善：{dm.mean():+.2f} kg [{np.percentile(dm,2.5):+.2f},{np.percentile(dm,97.5):+.2f}]  P(2a 較佳)={np.mean(dm>0):.3f}")

# ── 第 3 層：雙側混合效應 ───────────────────────────────────────────
if "3" in TIERS.split(","):
    print(f"\n{'='*118}\n第 3 層 雙側混合效應（268 手觀察值，同側前臂・100 (R,X)，受試者隨機截距）\n{'='*118}")
    rows=[]
    for sid in df.index:
        for side,gcol in (("左","grip_L"),("右","grip_R")):
            rec_={"subject":sid,"side":side,"y":df.loc[sid,gcol],"sex_male":df.loc[sid,"sex_male"],"age":df.loc[sid,"age"],"BMI":df.loc[sid,"BMI"]}
            for c in FORE100:
                if c.startswith(side): rec_["f"+re.sub(r"[^A-Za-z0-9]","_",c.split("|")[0][1:])+"_"+c.split("|")[1]]=df.loc[sid,c]
            rows.append(rec_)
    L=pd.DataFrame(rows); FC=[c for c in L.columns if c.startswith("f")]
    subj=np.array(sorted(df.index)); ysub={s:df.loc[s,"grip_dom"] for s in subj}
    PB=np.zeros(len(L)); PM=np.zeros(len(L)); cntv=np.zeros(len(L))
    for r in range(N_REP):
        for tr,te in splits(r):
            trs=set(df.index[tr]); Ltr=L[L.subject.isin(trs)].copy(); Lte=L[~L.subject.isin(trs)].copy()
            imp=SimpleImputer(strategy="median").fit(Ltr[FC]); Ltr[FC]=imp.transform(Ltr[FC]); Lte[FC]=imp.transform(Lte[FC])
            Dm_tr=Ltr[DEMO].to_numpy(float); Dm_te=Lte[DEMO].to_numpy(float)
            oy=LinearRegression().fit(Dm_tr,Ltr.y); Ltr["ystar"]=Ltr.y-oy.predict(Dm_tr); pd_te=oy.predict(Dm_te)
            ox=LinearRegression().fit(Dm_tr,Ltr[FC]); Xs_tr=Ltr[FC].to_numpy()-ox.predict(Dm_tr); Xs_te=Lte[FC].to_numpy()-ox.predict(Dm_te)
            sc=StandardScaler().fit(Xs_tr); Ltr[[c+"_s" for c in FC]]=sc.transform(Xs_tr); Lte[[c+"_s" for c in FC]]=sc.transform(Xs_te)
            f="ystar ~ "+" + ".join(c+"_s" for c in FC)
            try:
                mm_=smf.mixedlm(f,Ltr,groups=Ltr["subject"]).fit(reml=True,method="lbfgs",maxiter=500)
                pm=pd_te+np.asarray(mm_.predict(Lte))
            except Exception as ex:
                pm=pd_te
            idx=Lte.index.to_numpy(); PB[idx]+=pd_te; PM[idx]+=pm; cntv[idx]+=1
    PB/=cntv; PM/=cntv; yy=L.y.to_numpy(float)
    # 受試者層級 bootstrap（整人重抽，兩手一起）
    sidx={s:np.flatnonzero(L.subject.to_numpy()==s) for s in subj}; dm=[]
    for _ in range(N_BOOT):
        b=np.concatenate([sidx[s] for s in rng.choice(subj,len(subj),replace=True)])
        dm.append(np.abs(yy[b]-PB[b]).mean()-np.abs(yy[b]-PM[b]).mean())
    dm=np.array(dm)
    print(f"手層級  基準 MAE={np.abs(yy-PB).mean():.2f}  +EIM MAE={np.abs(yy-PM).mean():.2f}  ΔMAE {dm.mean():+.2f} [{np.percentile(dm,2.5):+.2f},{np.percentile(dm,97.5):+.2f}]  P={np.mean(dm>0):.3f}")
    print(f"對照 1a（單手 134）：ΔMAE {RES['1a']['dmae']:+.2f} [{RES['1a']['dmae_lo']:+.2f},{RES['1a']['dmae_hi']:+.2f}]" if "1a" in RES else "")
    RES["3"]=dict(dmae=dm.mean(),dmae_lo=np.percentile(dm,2.5),dmae_hi=np.percentile(dm,97.5),p_pos=float(np.mean(dm>0)))

# ── 第 4 層：AWGS 握力未達分類 ──────────────────────────────────────
if "4" in TIERS.split(","):
    print(f"\n{'='*118}\n第 4 層 AWGS 握力未達分類（陽性 {int(df.weak.sum())}/{n}）\n{'='*118}")
    w=df["weak"].to_numpy(int); Dm=df[DEMO].to_numpy(float); X=df[FORE100].to_numpy(float)
    PB=np.zeros(n); PE=np.zeros(n)
    for r in range(N_REP):
        for tr,te in StratifiedKFold(N_FOLD,shuffle=True,random_state=SEED+r).split(np.zeros(n),w):
            imp=SimpleImputer(strategy="median").fit(X[tr]); Xtr,Xte=imp.transform(X[tr]),imp.transform(X[te])
            ox=LinearRegression().fit(Dm[tr],Xtr); Xs_tr=Xtr-ox.predict(Dm[tr]); Xs_te=Xte-ox.predict(Dm[te])
            lb=Pipeline([("sc",StandardScaler()),("lr",LogisticRegressionCV(Cs=np.logspace(-3,2,6),cv=inner(SEED+r),max_iter=5000,scoring="roc_auc"))]).fit(Dm[tr],w[tr])
            le=Pipeline([("sc",StandardScaler()),("lr",LogisticRegressionCV(Cs=np.logspace(-3,2,6),cv=inner(SEED+r),max_iter=5000,scoring="roc_auc"))]).fit(np.hstack([Dm[tr],Xs_tr]),w[tr])
            PB[te]+=lb.predict_proba(Dm[te])[:,1]; PE[te]+=le.predict_proba(np.hstack([Dm[te],Xs_te]))[:,1]
    PB/=N_REP; PE/=N_REP; da=[]
    for _ in range(N_BOOT):
        b=rng.choice(n,n,replace=True)
        if w[b].sum() in (0,len(b)): continue
        da.append(roc_auc_score(w[b],PE[b])-roc_auc_score(w[b],PB[b]))
    da=np.array(da)
    print(f"AUC 人口學={roc_auc_score(w,PB):.3f}  人口學+EIM={roc_auc_score(w,PE):.3f}  ΔAUC {da.mean():+.3f} [{np.percentile(da,2.5):+.3f},{np.percentile(da,97.5):+.3f}]  P={np.mean(da>0):.3f}")
    RES["4"]=dict(auc_base=roc_auc_score(w,PB),auc_eim=roc_auc_score(w,PE),dauc=da.mean(),dauc_lo=np.percentile(da,2.5),dauc_hi=np.percentile(da,97.5))

np.savez(f"{OUT}/11_models_oof.npz",y=y,ids=np.array(df.index),**PRED)
json.dump({k:{kk:float(vv) for kk,vv in v.items()} for k,v in RES.items()},open(f"{OUT}/11_models_results.json","w"),ensure_ascii=False,indent=1)
print("\n已存：11_models_oof.npz、11_models_results.json")
