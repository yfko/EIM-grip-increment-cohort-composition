# -*- coding: utf-8 -*-
"""補跑 1b：記錄每個外層折的群組係數 L2 範數，分辨「常被選」與「重要」。唯讀。"""
import sys,re,numpy as np
sys.argv=["11_models.py","none"]
src=open(__file__.replace("13_sgl_norms.py","11_models.py"),encoding="utf-8").read()
exec(src.split("# ── 執行")[0])                       # 載入資料、FWL、splits 等定義，不執行任何層
from group_lasso import GroupLasso
from sklearn.preprocessing import StandardScaler
GR=np.logspace(-3,0,5); LR=np.logspace(-3,0,5)
def fit_one(X,y_,g,l):
    gl=GroupLasso(groups=GID,group_reg=g,l1_reg=l,scale_reg="group_size",supress_warning=True,n_iter=3000,tol=1e-4,fit_intercept=True,frobenius_lipschitz=True); gl.fit(X,y_); return gl
norms=[]; sel=[]; best_params=[]
for r in range(N_REP):
    for tr,te in splits(r):
        Xs_tr,Xs_te,ystar,pdemo,_,_=fwl_split(tr,te,ALLCOLS)
        sc=StandardScaler().fit(Xs_tr); Xtr=sc.transform(Xs_tr); best=(np.inf,None)
        for g in GR:
            for l in LR:
                errs=[]
                for itr,ite in inner(SEED+r).split(Xtr):
                    try:
                        gl=fit_one(Xtr[itr],ystar[itr],g,l); p=np.ravel(gl.predict(Xtr[ite]))
                        errs.append(np.abs(ystar[ite]-p).mean() if np.all(np.isfinite(p)) else np.inf)
                    except Exception: errs.append(np.inf)
                e=np.mean(errs)
                if e<best[0]: best=(e,(g,l))
        gl=fit_one(Xtr,ystar,*best[1]); w=np.ravel(gl.coef_)
        norms.append([np.linalg.norm(w[GID==k]) for k in range(len(GROUPS))]); sel.append([bool(np.any(np.abs(w[GID==k])>1e-8)) for k in range(len(GROUPS))]); best_params.append(best[1])
N=np.array(norms); S=np.array(sel); share=N/np.maximum(N.sum(1,keepdims=True),1e-12)
print(f"外層折數 {len(N)}｜最常選的 (group_reg, l1_reg)：{max(set(best_params),key=best_params.count)}\n")
print(f"{'群':<10}{'選入%':>7}{'平均範數':>10}{'範數SD':>9}{'平均佔比':>9}{'佔比中位':>9}")
order=np.argsort(-N.mean(0))
for k in order:
    r_,f_=GROUPS[k]; print(f"{r_+'・'+f_:<10}{100*S[:,k].mean():>6.0f}%{N[:,k].mean():>10.3f}{N[:,k].std():>9.3f}{100*share[:,k].mean():>8.1f}%{100*np.median(share[:,k]):>8.1f}%")
np.savez(f"{OUT}/13_sgl_norms.npz",norms=N,sel=S,groups=np.array([f"{a}・{b}" for a,b in GROUPS]))
print("\n已存 13_sgl_norms.npz")
