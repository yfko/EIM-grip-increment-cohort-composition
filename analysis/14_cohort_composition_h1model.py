# -*- coding: utf-8 -*-
"""族群設計對照，改用預註冊 H1 模型（elastic net、殘差化、前臂・100 (R,X) 8 特徵）
以消除 02/03 與 11 之間特徵集不一致的問題。輸出：
(a) 109 內訓練測試：ΔMAE、ΔR² 受試者 bootstrap
(b) 134 訓練、分子群評估：健康 109 / 受損 25 / 握力未達 21 的 ΔMAE 與 bootstrap
唯讀。"""
import sys,numpy as np
sys.argv=["11_models.py","none"]
src=open(__file__.replace("14_cohort_composition_h1model.py","11_models.py"),encoding="utf-8").read()
exec(src.split("# ── 執行")[0])
from sklearn.linear_model import ElasticNetCV
def mk(s): return Pipeline([("sc",StandardScaler()),("en",ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9],n_alphas=100,cv=inner(s),max_iter=50000,random_state=s))])
def boot(idx,yy,pb,pn):
    dm,dr=[],[]
    for _ in range(N_BOOT):
        b=rng.choice(idx,len(idx),replace=True)
        dm.append(np.abs(yy[b]-pb[b]).mean()-np.abs(yy[b]-pn[b]).mean()); dr.append(r2(yy[b],pn[b])-r2(yy[b],pb[b]))
    dm,dr=np.array(dm),np.array(dr)
    return f"ΔMAE {dm.mean():+.2f} [{np.percentile(dm,2.5):+.2f},{np.percentile(dm,97.5):+.2f}] P={np.mean(dm>0):.3f} | ΔR² {dr.mean():+.3f} [{np.percentile(dr,2.5):+.3f},{np.percentile(dr,97.5):+.3f}]"
healthy=a.loc[df.index,"分析納入"].astype(bool).to_numpy(); gripok=a.loc[df.index,"握力正常"].astype(bool).to_numpy()
# (b) 134 訓練（重用 oof_model 與 oof_baseline）
pb=oof_baseline(N_REP); pn=oof_model(mk,FORE100,N_REP)
print("=== (b) 全 134 訓練，H1 模型（EN 殘差化，8 特徵），分子群評估 ===")
for lab,msk in [("全體 134",np.ones(n,bool)),("健康 109",healthy),("受損 25",~healthy),("握力未達 21",~gripok)]:
    idx=np.flatnonzero(msk); print(f"{lab:<12} n={len(idx):3d}  基準 MAE {np.abs(y[idx]-pb[idx]).mean():.2f} → +EIM {np.abs(y[idx]-pn[idx]).mean():.2f}  |  {boot(idx,y,pb,pn)}")
# (a) 109 內訓練測試
sub=np.flatnonzero(healthy); y_h=y[sub]; Dm=df[DEMO].to_numpy(float)[sub]; X=df[FORE100].to_numpy(float)[sub]
strat_h=(df["sex_male"].astype(str)+"_"+(df["grip_dom"]>df["grip_dom"].iloc[sub].median()).astype(int).astype(str)).to_numpy()[sub]
PB=np.empty((N_REP,len(sub))); PN=np.empty((N_REP,len(sub)))
for r in range(N_REP):
    for tr,te in StratifiedKFold(N_FOLD,shuffle=True,random_state=SEED+r).split(np.zeros(len(sub)),strat_h):
        imp=SimpleImputer(strategy="median").fit(X[tr]); Xtr,Xte=imp.transform(X[tr]),imp.transform(X[te])
        oy=LinearRegression().fit(Dm[tr],y_h[tr]); ystar=y_h[tr]-oy.predict(Dm[tr]); PB[r,te]=oy.predict(Dm[te])
        ox=LinearRegression().fit(Dm[tr],Xtr); m_=mk(SEED+r).fit(Xtr-ox.predict(Dm[tr]),ystar)
        PN[r,te]=PB[r,te]+np.ravel(m_.predict(Xte-ox.predict(Dm[te])))
pb_h,pn_h=PB.mean(0),PN.mean(0)
print("\n=== (a) 健康 109 內訓練並測試，H1 模型 ===")
print(f"基準 R² {r2(y_h,pb_h):.3f} MAE {np.abs(y_h-pb_h).mean():.2f} → +EIM R² {r2(y_h,pn_h):.3f} MAE {np.abs(y_h-pn_h).mean():.2f}  |  {boot(np.arange(len(sub)),y_h,pb_h,pn_h)}")
# 交互作用：(Δ_134全體 − Δ_109內) 的 bootstrap（兩組獨立重抽）
dm=[]
for _ in range(N_BOOT):
    b1=rng.choice(n,n,replace=True); b2=rng.choice(len(sub),len(sub),replace=True)
    d134=np.abs(y[b1]-pb[b1]).mean()-np.abs(y[b1]-pn[b1]).mean(); d109=np.abs(y_h[b2]-pb_h[b2]).mean()-np.abs(y_h[b2]-pn_h[b2]).mean(); dm.append(d134-d109)
dm=np.array(dm); print(f"\n交互作用 Δ(134 訓練) − Δ(109 內)：{dm.mean():+.2f} kg [{np.percentile(dm,2.5):+.2f},{np.percentile(dm,97.5):+.2f}]  P(>0)={np.mean(dm>0):.3f}")
