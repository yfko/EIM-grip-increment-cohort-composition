# -*- coding: utf-8 -*-
"""缺的那一格：只用健康 109 訓練（H1 模型），去預測受損 25 人。
與「134 訓練 → 受損 25 (out-of-fold)」對照，回答『把受損者放進訓練集有沒有幫到受損者』。唯讀。"""
import sys,numpy as np
sys.argv=["11_models.py","none"]
src=open(__file__.replace("16_train109_predict25.py","11_models.py"),encoding="utf-8").read()
exec(src.split("# ── 執行")[0])
from sklearn.linear_model import ElasticNetCV
def mk(s): return Pipeline([("sc",StandardScaler()),("en",ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9],n_alphas=100,cv=inner(s),max_iter=50000,random_state=s))])
healthy=a.loc[df.index,"分析納入"].astype(bool).to_numpy(); gripok=a.loc[df.index,"握力正常"].astype(bool).to_numpy()
H=np.flatnonzero(healthy); I=np.flatnonzero(~healthy); G=np.flatnonzero(~gripok)
Dm=df[DEMO].to_numpy(float); X=df[FORE100].to_numpy(float)
PB=np.zeros(len(I)); PN=np.zeros(len(I))
for r in range(N_REP):                       # 10 個種子，全 109 訓練，預測 25（25 人從未進訓練）
    imp=SimpleImputer(strategy="median").fit(X[H]); Xh,Xi=imp.transform(X[H]),imp.transform(X[I])
    oy=LinearRegression().fit(Dm[H],y[H]); ystar=y[H]-oy.predict(Dm[H]); pb=oy.predict(Dm[I])
    ox=LinearRegression().fit(Dm[H],Xh); m_=mk(SEED+r).fit(Xh-ox.predict(Dm[H]),ystar)
    PB+=pb; PN+=pb+np.ravel(m_.predict(Xi-ox.predict(Dm[I])))
PB/=N_REP; PN/=N_REP; yi=y[I]
def boot(yy,pb,pn,idx):
    dm=[]
    for _ in range(N_BOOT):
        b=rng.choice(idx,len(idx),replace=True); dm.append(np.abs(yy[b]-pb[b]).mean()-np.abs(yy[b]-pn[b]).mean())
    dm=np.array(dm); return f"ΔMAE {dm.mean():+.2f} [{np.percentile(dm,2.5):+.2f},{np.percentile(dm,97.5):+.2f}] P={np.mean(dm>0):.3f}"
print("=== 只用健康 109 訓練 → 預測受損 25（真正的外部子群）===")
print(f"受損 25    基準 MAE {np.abs(yi-PB).mean():.2f} → +EIM {np.abs(yi-PN).mean():.2f}  |  {boot(yi,PB,PN,np.arange(len(I)))}")
gi=np.isin(I,G); print(f"握力未達 21 基準 MAE {np.abs(yi[gi]-PB[gi]).mean():.2f} → +EIM {np.abs(yi[gi]-PN[gi]).mean():.2f}  |  {boot(yi,PB,PN,np.flatnonzero(gi))}")
print("\n對照（14 號）：134 訓練 out-of-fold → 受損 25：8.25 → 7.09，ΔMAE +1.16 [+0.22,+2.14]；握力未達 21：8.17 → 6.75，+1.44 [+0.32,+2.53]")
print(f"\n109 訓練模型對受損 25 的基準預測偏差（預測−實際 平均）：{(PB-yi).mean():+.2f} kg（正值＝高估握力）")
