# -*- coding: utf-8 -*-
"""回應統計席 W1/W2：(1) 受損 vs 健康 子群增量差的配對 bootstrap（同一 134 訓練 out-of-fold 預測）；
(2) 正確標示的訓練族群交互作用：健康子群在 134 訓練 vs 109 內訓練的增量差；(3) 最小可偵測效應（MDE）。唯讀。"""
import sys,numpy as np
sys.argv=["11_models.py","none"]
src=open(__file__.replace("19_subgroup_difference_bootstrap.py","11_models.py"),encoding="utf-8").read()
exec(src.split("# ── 執行")[0])
from sklearn.linear_model import ElasticNetCV
def mk(s): return Pipeline([("sc",StandardScaler()),("en",ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9],n_alphas=100,cv=inner(s),max_iter=50000,random_state=s))])
healthy=a.loc[df.index,"分析納入"].astype(bool).to_numpy(); gripok=a.loc[df.index,"握力正常"].astype(bool).to_numpy()
H=np.flatnonzero(healthy); I=np.flatnonzero(~healthy); G=np.flatnonzero(~gripok)
pb=oof_baseline(N_REP); pn=oof_model(mk,FORE100,N_REP)
e_b=np.abs(y-pb); e_n=np.abs(y-pn); gain=e_b-e_n                      # 每人增量（kg）
def boot_diff(idxA,idxB,label):
    d=[]
    for _ in range(N_BOOT):
        bA=rng.choice(idxA,len(idxA),replace=True); bB=rng.choice(idxB,len(idxB),replace=True); d.append(gain[bA].mean()-gain[bB].mean())
    d=np.array(d); print(f"{label}: 差 {d.mean():+.2f} kg [{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}]  P(>0)={np.mean(d>0):.3f}")
print("=== (1) 134 訓練，子群增量差（絕對，kg）===")
boot_diff(I,H,"受損25 − 健康109")
boot_diff(G,H,"握力未達21 − 健康109")
# 比例版
def boot_ratio(idxA,idxB,label):
    d=[]
    for _ in range(N_BOOT):
        bA=rng.choice(idxA,len(idxA),replace=True); bB=rng.choice(idxB,len(idxB),replace=True)
        d.append(gain[bA].mean()/e_b[bA].mean()-gain[bB].mean()/e_b[bB].mean())
    d=np.array(d); print(f"{label}: 比例差 {100*d.mean():+.1f} 個百分點 [{100*np.percentile(d,2.5):+.1f},{100*np.percentile(d,97.5):+.1f}]  P(>0)={np.mean(d>0):.3f}")
print("=== (1b) 比例增量差（ΔMAE/基準 MAE）===")
boot_ratio(I,H,"受損25 − 健康109")
print(f"   點估計：健康 {100*gain[H].mean()/e_b[H].mean():.1f}%，受損 {100*gain[I].mean()/e_b[I].mean():.1f}%，握力未達 {100*gain[G].mean()/e_b[G].mean():.1f}%")
# (2) 正確的訓練族群交互作用：健康子群 134 訓練 vs 109 內訓練
sub=H; y_h=y[sub]; Dm=df[DEMO].to_numpy(float)[sub]; X=df[FORE100].to_numpy(float)[sub]
strat_h=(df["sex_male"].astype(str)+"_"+(df["grip_dom"]>df["grip_dom"].iloc[sub].median()).astype(int).astype(str)).to_numpy()[sub]
PB=np.empty((N_REP,len(sub))); PN=np.empty((N_REP,len(sub)))
for r in range(N_REP):
    for tr,te in StratifiedKFold(N_FOLD,shuffle=True,random_state=SEED+r).split(np.zeros(len(sub)),strat_h):
        imp=SimpleImputer(strategy="median").fit(X[tr]); Xtr,Xte=imp.transform(X[tr]),imp.transform(X[te])
        oy=LinearRegression().fit(Dm[tr],y_h[tr]); PB[r,te]=oy.predict(Dm[te]); ox=LinearRegression().fit(Dm[tr],Xtr)
        m_=mk(SEED+r).fit(Xtr-ox.predict(Dm[tr]),y_h[tr]-oy.predict(Dm[tr])); PN[r,te]=PB[r,te]+np.ravel(m_.predict(Xte-ox.predict(Dm[te])))
gain109=np.abs(y_h-PB.mean(0))-np.abs(y_h-PN.mean(0))                  # 109 內訓練的每人增量（同一批健康人）
d=[]
for _ in range(N_BOOT):
    b=rng.choice(len(sub),len(sub),replace=True); d.append(gain[H][b].mean()-gain109[b].mean())   # 配對（同一人兩種訓練）
d=np.array(d); print(f"\n=== (2) 訓練族群交互作用（健康子群，配對）：Δ(134 訓練) − Δ(109 內) = {d.mean():+.2f} kg [{np.percentile(d,2.5):+.2f},{np.percentile(d,97.5):+.2f}]  P(>0)={np.mean(d>0):.3f}")
print(f"   點估計：134 訓練 {gain[H].mean():+.2f}；109 內 {gain109.mean():+.2f}")
# (3) MDE：以 bootstrap SE 推 80% 檢定力、雙側 5% 的最小可偵測差
for lab,(iA,iB) in {"受損25 vs 健康109":(I,H)}.items():
    se=np.std([gain[rng.choice(iA,len(iA))].mean()-gain[rng.choice(iB,len(iB))].mean() for _ in range(2000)])
    print(f"\n=== (3) MDE（{lab}，80% power，α=0.05 雙側）≈ {2.80*se:.2f} kg（SE {se:.2f}）===")
