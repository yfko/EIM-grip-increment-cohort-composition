# -*- coding: utf-8 -*-
"""回應 DA M2：(a) 各子群的比例改善；(b) 置換零特徵對照——把前臂特徵在受試者間隨機置換（切斷與結果的關聯），
同一管線下健康／受損子群的 ΔMAE 是否仍出現「受損者較大」的差異。若零特徵在受損者也產生較大 ΔMAE，DA 的機械性解釋成立。唯讀。"""
import sys,numpy as np
sys.argv=["11_models.py","none"]
src=open(__file__.replace("18_da_m2_check.py","11_models.py"),encoding="utf-8").read()
exec(src.split("# ── 執行")[0])
from sklearn.linear_model import ElasticNetCV
def mk(s): return Pipeline([("sc",StandardScaler()),("en",ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9],n_alphas=100,cv=inner(s),max_iter=50000,random_state=s))])
healthy=a.loc[df.index,"分析納入"].astype(bool).to_numpy(); H=np.flatnonzero(healthy); I=np.flatnonzero(~healthy)
Dm=df[DEMO].to_numpy(float); X0=df[FORE100].to_numpy(float)
def fourcell(X,label):
    # (b)(c) 134 訓練 out-of-fold
    P=np.empty((N_REP,n)); PB=np.empty((N_REP,n))
    for r in range(N_REP):
        for tr,te in splits(r):
            imp=SimpleImputer(strategy="median").fit(X[tr]); Xtr,Xte=imp.transform(X[tr]),imp.transform(X[te])
            oy=LinearRegression().fit(Dm[tr],y[tr]); PB[r,te]=oy.predict(Dm[te]); ox=LinearRegression().fit(Dm[tr],Xtr)
            m_=mk(SEED+r).fit(Xtr-ox.predict(Dm[tr]),y[tr]-oy.predict(Dm[tr])); P[r,te]=PB[r,te]+np.ravel(m_.predict(Xte-ox.predict(Dm[te])))
    pb,pn=PB.mean(0),P.mean(0)
    out=[]
    for lab,idx in [("健康109",H),("受損25",I)]:
        b=np.abs(y[idx]-pb[idx]).mean(); m=np.abs(y[idx]-pn[idx]).mean(); out.append((lab,b,m,b-m,(b-m)/b))
    # (iv) 109 訓練 → 25
    imp=SimpleImputer(strategy="median").fit(X[H]); Xh,Xi=imp.transform(X[H]),imp.transform(X[I])
    PBi=np.zeros(len(I)); PNi=np.zeros(len(I))
    for r in range(N_REP):
        oy=LinearRegression().fit(Dm[H],y[H]); ox=LinearRegression().fit(Dm[H],Xh); m_=mk(SEED+r).fit(Xh-ox.predict(Dm[H]),y[H]-oy.predict(Dm[H]))
        PBi+=oy.predict(Dm[I]); PNi+=oy.predict(Dm[I])+np.ravel(m_.predict(Xi-ox.predict(Dm[I])))
    PBi/=N_REP; PNi/=N_REP; b=np.abs(y[I]-PBi).mean(); m=np.abs(y[I]-PNi).mean(); out.append(("109訓練→受損25",b,m,b-m,(b-m)/b))
    print(f"=== {label} ===")
    for lab,b,m,d,p in out: print(f"  {lab:<14} 基準 MAE {b:5.2f} → {m:5.2f}  ΔMAE {d:+.2f} kg  比例 {p:+.1%}")
    return out
real=fourcell(X0,"真實前臂特徵")
rs=np.random.default_rng(20260905); res=[]
for k in range(20):                       # 20 次受試者間置換（零特徵：與結果無關但保留邊際分布）
    Xp=X0[rs.permutation(n)]; res.append([o[3] for o in fourcell(Xp,f"置換 {k+1}")] if k<2 else None)
    if k>=2:
        # 靜默計算
        pass
# 為節省輸出，重跑置換 20 次但只彙整
import io,contextlib
vals=[]
for k in range(20):
    Xp=X0[rs.permutation(n)]
    with contextlib.redirect_stdout(io.StringIO()): o=fourcell(Xp,"")
    vals.append([x[3] for x in o])
vals=np.array(vals)
print("\n=== 置換零特徵 20 次的 ΔMAE 分布（健康109 / 受損25 / 109訓練→25）===")
for j,lab in enumerate(["健康109","受損25","109訓練→25"]):
    print(f"  {lab:<12} 平均 {vals[:,j].mean():+.2f}  範圍 [{vals[:,j].min():+.2f}, {vals[:,j].max():+.2f}]  ≥真實值的次數 {int((vals[:,j]>=real[j][3]).sum())}/20")
