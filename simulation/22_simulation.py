# -*- coding: utf-8 -*-
"""方法篇模擬（預註冊 22_模擬預註冊.md v1.0）。用法：python 22_simulation.py [S0|S1|S2|S3|all] [reps]
輸出 22_sim_<S>.json 與 22_sim_<S>_out.txt。合成資料，無受試者資料。"""
import sys, json, time, warnings, numpy as np
from sklearn.linear_model import LassoCV, ElasticNetCV, LinearRegression, Lasso
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
SEED=20260906; CAL=json.load(open("21_sim_calibration.json"))
WHICH=sys.argv[1] if len(sys.argv)>1 else "all"; REPS=int(sys.argv[2]) if len(sys.argv)>2 else 500
NJ=6
# ── 結構 ─────────────────────────────────────────────────────────────
REG={"forearm":4,"upper_arm":4,"trunk":2,"thigh":4,"calf":4}
GROUPS=[(r,f) for r in REG for f in (50,100)]            # 10 群
CH=[(r,f,k) for r in REG for f in (50,100) for k in range(REG[r])]   # 18 通道 × 2 頻率 = 36
def gid(r,f): return GROUPS.index((r,f))
# 校準用的負荷（S0 網格搜尋後固定；先給初值）
P=dict(a_sex=0.70,a_age=0.35,a_bmi=0.23,lam=0.35,phi=0.20,reg=0.20,rho=0.93,noise=0.73,b_x_sex=-0.2,psi=0.53)
def gen_demo(n,rng):
    sex=rng.binomial(1,CAL["demographics"]["p_male"],n)
    age=np.clip(rng.normal(CAL["demographics"]["age_mean"],CAL["demographics"]["age_sd"],n),18,87)
    bmi=rng.normal(CAL["demographics"]["bmi_mean"],CAL["demographics"]["bmi_sd"],n)
    D=np.c_[sex,(age-53.1)/20.7,(bmi-23.9)/3.9]; return D,sex,age,bmi
def gen_features(n,rng,D,p=P):
    """回傳 R (n×36), X (n×36), U (n×10 群因子，已去 D，作為真訊號來源)"""
    u=rng.normal(size=(n,len(GROUPS))); v=rng.normal(size=(n,2)); rreg=rng.normal(size=(n,len(REG))); w0=rng.normal(size=n)
    R=np.zeros((n,len(CH))); X=np.zeros((n,len(CH)))
    for j,(r,f,k) in enumerate(CH):
        g=gid(r,f); s=rng.choice([-1,1])
        core=p["lam"]*u[:,g]+p["phi"]*v[:,0 if f==50 else 1]+p["reg"]*rreg[:,list(REG).index(r)]+p["psi"]*w0
        R[:,j]=s*(p["a_sex"]*(D[:,0]-0.5)*2+p["a_age"]*D[:,1]+p["a_bmi"]*D[:,2])+core+p["noise"]*rng.normal(size=n)
        Rs=(R[:,j]-R[:,j].mean())/R[:,j].std()
        X[:,j]=p["rho"]*Rs+np.sqrt(1-p["rho"]**2)*rng.normal(size=n)+p["b_x_sex"]*(D[:,0]-0.5)*2
    R=R*3+20; X=X*1.5+8      # 正值尺度（R≈20±3, X≈8±1.5）
    return R,X,u
def feats(R,X,cols=("R","X"),chan=None):
    idx=range(len(CH)) if chan is None else chan; out=[]; names=[]
    for j in idx:
        d={"R":R[:,j],"X":X[:,j],"PhA":np.degrees(np.arctan2(X[:,j],R[:,j])),"Z":np.sqrt(R[:,j]**2+X[:,j]**2)}
        for c in cols: out.append(d[c]); names.append(f"{CH[j][0]}|{CH[j][1]}|{CH[j][2]}|{c}")
    return np.c_[tuple(out)] if out else np.zeros((len(R),0)), names
def gen_y(n,rng,D,u,gamma,impaired=None,kappa=0.0,b=(7.0,-3.5,1.4),total_var=10.35**2,intercept=30.0):
    """grip = 30 + b·D + Σ γ_g u_g + ε；b 使人口學 R² ≈ 0.60（var 63/107）；ε 的變異數自適應使總變異 ≈ 10.35²（若 γ 太大則 ε SD 下限 2.0）。
    impaired: 0/1 向量，受損者 −9 kg，κ 比例已由呼叫端反映到前臂·100 群因子"""
    var_demo=b[0]**2+b[1]**2+b[2]**2; var_u=float(np.sum(gamma**2)); eps=np.sqrt(max(total_var-var_demo-var_u,4.0))
    y=intercept+b[0]*(D[:,0]-0.5)*2+b[1]*D[:,1]+b[2]*D[:,2]+u@gamma+rng.normal(0,eps,n)
    if impaired is not None: y=y-9.0*impaired
    return y
# ── 評估 ─────────────────────────────────────────────────────────────
def cv_oof(Xf,D,y,model,nrep=3,nfold=5,rng=None,resid=False):
    np.seterr(all="ignore"); warnings.filterwarnings("ignore")
    """回傳 OOF 預測（人口學基準、基準+EIM）與每折選入指標（非零係數布林）"""
    n=len(y); pb=np.zeros(n); pe=np.zeros(n); sel=[]
    for r in range(nrep):
        kf=KFold(nfold,shuffle=True,random_state=SEED+r)
        for tr,te in kf.split(Xf):
            ols=LinearRegression().fit(D[tr],y[tr]); pb[te]+=ols.predict(D[te])/nrep
            if resid:
                ox=LinearRegression().fit(D[tr],Xf[tr]); Xtr=Xf[tr]-ox.predict(D[tr]); Xte=Xf[te]-ox.predict(D[te]); ytr=y[tr]-ols.predict(D[tr])
                sc=StandardScaler().fit(Xtr); Ztr=np.nan_to_num(sc.transform(Xtr),posinf=0,neginf=0); Zte=np.nan_to_num(sc.transform(Xte),posinf=0,neginf=0); m=model().fit(Ztr,ytr); pe[te]+=(ols.predict(D[te])+m.predict(Zte))/nrep
            else:
                Ztr=np.c_[Xf[tr],D[tr]]; Zte=np.c_[Xf[te],D[te]]; sc=RobustScaler().fit(Ztr); Ztr2=np.nan_to_num(sc.transform(Ztr),posinf=0,neginf=0); Zte2=np.nan_to_num(sc.transform(Zte),posinf=0,neginf=0); m=model().fit(Ztr2,y[tr]); pe[te]+=m.predict(Zte2)/nrep
            coef=np.asarray(getattr(m,"coef_",np.zeros(Xf.shape[1]))).ravel()[:Xf.shape[1]]; sel.append(np.abs(coef)>1e-8)
    return pb,pe,np.array(sel)
def metrics(y,pb,pe,rng,nboot=500):
    mae=lambda p:np.mean(np.abs(y-p)); r2=lambda p:1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)
    d=mae(pb)-mae(pe); dr=r2(pe)-r2(pb); n=len(y); bd=[]; br=[]
    for _ in range(nboot):
        i=rng.integers(0,n,n); yy,b1,e1=y[i],pb[i],pe[i]
        bd.append(np.mean(np.abs(yy-b1))-np.mean(np.abs(yy-e1))); bl=lambda p:1-np.sum((yy-p)**2)/np.sum((yy-yy.mean())**2); br.append(bl(e1)-bl(b1))
    return dict(dmae=d,dmae_lo=np.percentile(bd,2.5),dmae_hi=np.percentile(bd,97.5),dr2=dr,dr2_lo=np.percentile(br,2.5),dr2_hi=np.percentile(br,97.5),r2_base=r2(pb),r2_eim=r2(pe))
def jaccard(sel):
    s=[];
    for i in range(len(sel)):
        for j in range(i+1,len(sel)):
            a,b=sel[i],sel[j]; u=np.sum(a|b); s.append(np.sum(a&b)/u if u else 1.0)
    return float(np.mean(s))
def truth(gamma,cols,chan,resid,impaired_rate=0.0,kappa=0.0,rng=None,N=60000,intercept=30.0):
    np.seterr(all="ignore")
    """母體真值：人口學基準 vs 基準+EIM 的最佳線性預測（OLS on N）"""
    rng=np.random.default_rng(SEED+999); D,*_=gen_demo(N,rng); R,X,u=gen_features(N,rng,D)
    imp=None
    if impaired_rate>0:
        imp=(rng.random(N)<impaired_rate).astype(float); u[:,gid("forearm",100)]-=kappa*imp
    y=gen_y(N,rng,D,u,gamma,imp,intercept=intercept)
    Xf,_=feats(R,X,cols,chan); pb=LinearRegression().fit(D,y).predict(D); pe=LinearRegression().fit(np.c_[Xf,D],y).predict(np.c_[Xf,D])
    return dict(dmae=np.mean(np.abs(y-pb))-np.mean(np.abs(y-pe)),dr2=(1-np.sum((y-pe)**2)/np.sum((y-y.mean())**2))-(1-np.sum((y-pb)**2)/np.sum((y-y.mean())**2)),r2_base=1-np.sum((y-pb)**2)/np.sum((y-y.mean())**2))
def gamma_for(target_dmae,groups,cols=("R","X"),chan=None,resid=True,impaired_rate=0.0,kappa=0.0,intercept=30.0):
    """二分法：找標量 g 使 groups 上 gamma=g 的母體真值 ΔMAE ≈ target"""
    lo,hi=0.0,15.0
    for _ in range(16):
        mid=(lo+hi)/2; gamma=np.zeros(len(GROUPS)); gamma[list(groups)]=mid
        if truth(gamma,cols,chan,resid,impaired_rate,kappa,N=30000,intercept=intercept)["dmae"]<target_dmae: lo=mid
        else: hi=mid
    gamma=np.zeros(len(GROUPS)); gamma[list(groups)]=(lo+hi)/2; return gamma
LASSO=lambda: LassoCV(alphas=np.logspace(-3,1,40),cv=5,max_iter=5000)
EN=lambda: ElasticNetCV(l1_ratio=[0.2,0.5,0.8],alphas=np.logspace(-3,1,30),cv=5,max_iter=5000)
class SGLManual:
    def __init__(self,groups,l1=(0.05,0.2,0.5),l2=(0.1,0.3,1.0)): self.groups=groups; self.l1=l1; self.l2=l2   # v1.3：探針顯示 0.01–0.1 全在「全選」角落，改為涵蓋全選→空集的範圍
    def fit(self,X,y):
        from group_lasso import GroupLasso
        best=(np.inf,None); kf=KFold(3,shuffle=True,random_state=SEED)
        for a in self.l1:
            for b in self.l2:
                err=0
                for tr,te in kf.split(X):
                    m=GroupLasso(groups=self.groups,group_reg=b,l1_reg=a,n_iter=300,tol=1e-3,supress_warning=True,fit_intercept=True,scale_reg="none").fit(X[tr],y[tr]); err+=np.mean((y[te]-m.predict(X[te]).ravel())**2)
                if err<best[0]: best=(err,(a,b))
        a,b=best[1]; self.m=GroupLasso(groups=self.groups,group_reg=b,l1_reg=a,n_iter=500,tol=1e-4,supress_warning=True,fit_intercept=True,scale_reg="none").fit(X,y); self.coef_=self.m.coef_.ravel(); return self
    def predict(self,X): return self.m.predict(X).ravel()
# ── S0 校準 ───────────────────────────────────────────────────────────
def calib_stats(p,N=8000):
    rng=np.random.default_rng(1); D,*_=gen_demo(N,rng); R,X,_=gen_features(N,rng,D,p); Xf,names=feats(R,X)
    C=np.abs(np.corrcoef(Xf.T)); g=[gid(CH[j//2][0],CH[j//2][1]) for j in range(Xf.shape[1])]
    w=[C[i,j] for i in range(len(g)) for j in range(i+1,len(g)) if g[i]==g[j]]; b=[C[i,j] for i in range(len(g)) for j in range(i+1,len(g)) if g[i]!=g[j]]
    rrx=np.median([np.corrcoef(R[:,j],X[:,j])[0,1] for j in range(len(CH))])
    r2=np.median([LinearRegression().fit(D,Xf[:,j]).score(D,Xf[:,j]) for j in range(Xf.shape[1])])
    sx=np.median([abs(np.corrcoef(D[:,0],Xf[:,j])[0,1]) for j in range(Xf.shape[1])])
    Rz=Xf-LinearRegression().fit(D,Xf).predict(D); Cz=np.abs(np.corrcoef(Rz.T))
    wz=[Cz[i,j] for i in range(len(g)) for j in range(i+1,len(g)) if g[i]==g[j]]; bz=[Cz[i,j] for i in range(len(g)) for j in range(i+1,len(g)) if g[i]!=g[j]]
    return dict(r_RX=rrx,within=np.median(w),between=np.median(b),R2_demo=r2,sex_r=sx,within_res=np.median(wz),between_res=np.median(bz))
TARGET=dict(r_RX=0.903,within=0.721,between=0.497,R2_demo=0.397,sex_r=0.555,within_res=0.47,between_res=0.282)
def S0():
    best=(np.inf,None,None)
    for lam in (0.30,0.35,0.42):
      for psi in (0.48,0.53,0.60):
        for noise in (0.65,0.73,0.80):
          for asx in (0.62,0.70,0.78):
            p=dict(P,lam=lam,psi=psi,noise=noise,a_sex=asx); st=calib_stats(p,3000)
            loss=sum(((st[k]-TARGET[k])/TARGET[k])**2 for k in TARGET)
            if loss<best[0]: best=(loss,p,st)
    p=best[1]; st=calib_stats(p,20000); P.update(p)
    out=dict(params=p,achieved=st,target=TARGET,loss=best[0]); json.dump(out,open("22_sim_S0.json","w"),indent=1)
    print("S0 校準：",json.dumps({k:[round(TARGET[k],3),round(st[k],3)] for k in TARGET})); return p
def load_P():
    try: P.update(json.load(open("22_sim_S0.json"))["params"])
    except FileNotFoundError: S0()
# ── S1 ───────────────────────────────────────────────────────────────
FORE100=[j for j,(r,f,k) in enumerate(CH) if r=="forearm" and f==100]
def one_S1(rep,gamma,n=134):
    rng=np.random.default_rng(SEED+rep); D,*_=gen_demo(n,rng); R,X,u=gen_features(n,rng,D); y=gen_y(n,rng,D,u,gamma); res={}
    for lab,cols in (("RX",("R","X")),("RXPhAZ",("R","X","PhA","Z"))):
        Xf,names=feats(R,X,cols,FORE100); pb,pe,sel=cv_oof(Xf,D,y,LASSO,rng=rng); m=metrics(y,pb,pe,rng)
        true_idx=[i for i,nm in enumerate(names) if nm.endswith("|R") or nm.endswith("|X")]
        m.update(sel_true=float(sel[:,true_idx].mean()),sel_any=float(sel.mean()),jaccard=jaccard(sel),n_sel=float(sel.sum(1).mean())); res[lab]=m
    return res
def S1(reps):
    gamma=gamma_for(0.48,[gid("forearm",100)],chan=FORE100)
    tr=truth(gamma,("R","X"),FORE100,False); print("S1 gamma=%.2f"%gamma[gid("forearm",100)]); print("S1 真值 ΔMAE %.3f ΔR² %.3f R²base %.3f"%(tr["dmae"],tr["dr2"],tr["r2_base"]))
    t=time.time(); out=Parallel(n_jobs=NJ)(delayed(one_S1)(i,gamma) for i in range(reps)); print("S1 %d reps %.0fs"%(reps,time.time()-t))
    summ={}
    for lab in ("RX","RXPhAZ"):
        a={k:np.array([o[lab][k] for o in out]) for k in out[0][lab]}
        summ[lab]=dict(sel_true=float(a["sel_true"].mean()),jaccard=float(a["jaccard"].mean()),n_sel=float(a["n_sel"].mean()),
            ciw_dr2=float(np.median(a["dr2_hi"]-a["dr2_lo"])),ciw_dmae=float(np.median(a["dmae_hi"]-a["dmae_lo"])),
            dmae_mean=float(a["dmae"].mean()),dmae_bias=float(a["dmae"].mean()-tr["dmae"]),dr2_mean=float(a["dr2"].mean()),
            cover_dmae=float(np.mean((a["dmae_lo"]<=tr["dmae"])&(a["dmae_hi"]>=tr["dmae"]))),cover_dr2=float(np.mean((a["dr2_lo"]<=tr["dr2"])&(a["dr2_hi"]>=tr["dr2"]))),
            false_null_dr2=float(np.mean(a["dr2"]<=0)))
    d=np.array([o["RX"]["dmae"]-o["RXPhAZ"]["dmae"] for o in out]); summ["dmae_RX_minus_RXPhAZ"]=dict(mean=float(d.mean()),lo=float(np.percentile(d,2.5)),hi=float(np.percentile(d,97.5)))
    summ["truth"]=tr; summ["reps"]=reps; json.dump(summ,open("22_sim_S1.json","w"),indent=1); print(json.dumps(summ,indent=1))
# ── S2 ───────────────────────────────────────────────────────────────
def one_S2(rep,gamma,n):
    rng=np.random.default_rng(SEED+10000+rep); D,*_=gen_demo(n,rng); R,X,u=gen_features(n,rng,D); y=gen_y(n,rng,D,u,gamma)
    Xf,names=feats(R,X,("R","X")); g=np.array([gid(CH[j//2][0],CH[j//2][1]) for j in range(Xf.shape[1])]); true_g=set(np.where(gamma!=0)[0]); res={}
    for lab,model in (("Lasso",LASSO),("EN",EN),("SGL",lambda: SGLManual(g))):
        pb,pe,sel=cv_oof(Xf,D,y,model,nrep=2,rng=rng,resid=True); m=metrics(y,pb,pe,rng,nboot=300)
        gsel=np.array([[sel[f][g==k].any() for k in range(len(GROUPS))] for f in range(len(sel))])
        m.update(true_group_rate=float(np.mean([gsel[f][list(true_g)].mean() for f in range(len(gsel))])),false_group_rate=float(np.mean([gsel[f][[k for k in range(len(GROUPS)) if k not in true_g]].mean() for f in range(len(gsel))])),jaccard=jaccard(sel),mae_eim=float(np.mean(np.abs(y-pe)))); res[lab]=m
    return res
def S2(reps):
    summ={}
    for n in (134,300):
        for strength,tgt in (("weak",0.35),("strong",0.70)):
            gamma=gamma_for(tgt,[gid("forearm",100),gid("trunk",100),gid("calf",50)])
            tr=truth(gamma,("R","X"),None,True); t=time.time(); out=Parallel(n_jobs=NJ)(delayed(one_S2)(i,gamma,n) for i in range(reps)); print("S2 n=%d %s %d reps %.0fs"%(n,strength,reps,time.time()-t))
            cell={}
            for lab in ("Lasso","EN","SGL"):
                a={k:np.array([o[lab][k] for o in out]) for k in out[0][lab]}
                cell[lab]=dict(true_group_rate=float(a["true_group_rate"].mean()),false_group_rate=float(a["false_group_rate"].mean()),jaccard=float(a["jaccard"].mean()),mae=float(a["mae_eim"].mean()),dmae=float(a["dmae"].mean()),dmae_bias=float(a["dmae"].mean()-tr["dmae"]),cover_dmae=float(np.mean((a["dmae_lo"]<=tr["dmae"])&(a["dmae_hi"]>=tr["dmae"]))))
            d=np.array([o["Lasso"]["mae_eim"]-o["SGL"]["mae_eim"] for o in out]); cell["mae_Lasso_minus_SGL"]=dict(mean=float(d.mean()),lo=float(np.percentile(d,2.5)),hi=float(np.percentile(d,97.5))); cell["truth"]=tr
            summ[f"n{n}_{strength}"]=cell; print(f"n{n}_{strength}",json.dumps({k:(v if k=='mae_Lasso_minus_SGL' else {kk:round(vv,3) for kk,vv in v.items()}) for k,v in cell.items() if k!='truth'}))
    summ["reps"]=reps; json.dump(summ,open("22_sim_S2.json","w"),indent=1)
# ── S3 ───────────────────────────────────────────────────────────────
def one_S3(rep,gamma,kappa,n=134,intercept=35.5):
    rng=np.random.default_rng(SEED+20000+rep); D,sex,age,bmi=gen_demo(n,rng); R,X,u=gen_features(n,rng,D)
    imp=(rng.random(n)<0.2).astype(float); u[:,gid("forearm",100)]-=kappa*imp; y=gen_y(n,rng,D,u,gamma,imp,intercept=intercept)
    thr=np.where(sex==1,28.0,18.0); thr=np.where(age<65,np.where(sex==1,34.0,20.0),thr); healthy=y>=thr; res={}
    for coh,mask in (("all",np.ones(n,bool)),("healthy",healthy)):
        for lab,cols in (("RX",("R","X")),("RXPhA",("R","X","PhA"))):
            Xf,_=feats(R,X,cols,FORE100)
            for mod,model,resid in (("LassoRaw",LASSO,False),("ENres",EN,True),("LassoRes",LASSO,True),("ENraw",EN,False)):   # v1.4：拆開估計器與殘差化
                pb,pe,sel=cv_oof(Xf[mask],D[mask],y[mask],model,rng=rng,resid=resid); m=metrics(y[mask],pb,pe,rng); m["n"]=int(mask.sum()); res[f"{coh}|{lab}|{mod}"]=m
    return res
def S3(reps):
    kappa=1.0; INT=35.5; gamma=gamma_for(0.48,[gid("forearm",100)],chan=FORE100,impaired_rate=0.2,kappa=kappa,intercept=INT)
    tr_all=truth(gamma,("R","X"),FORE100,False,impaired_rate=0.2,kappa=kappa,intercept=INT); print("S3 真值（全族群）ΔMAE %.3f ΔR² %.3f"%(tr_all["dmae"],tr_all["dr2"]))
    t=time.time(); out=Parallel(n_jobs=NJ)(delayed(one_S3)(i,gamma,kappa,134,INT) for i in range(reps)); print("S3 %d reps %.0fs"%(reps,time.time()-t))
    summ={"truth_all":tr_all,"reps":reps}
    for key in out[0]:
        a={k:np.array([o[key][k] for o in out]) for k in out[0][key]}
        summ[key]=dict(n=float(a["n"].mean()),dr2_mean=float(a["dr2"].mean()),dmae_mean=float(a["dmae"].mean()),false_null_dr2=float(np.mean(a["dr2"]<=0)),false_null_dmae=float(np.mean(a["dmae_lo"]<=0)),ciw_dr2=float(np.median(a["dr2_hi"]-a["dr2_lo"])),cover_dmae_truthall=float(np.mean((a["dmae_lo"]<=tr_all["dmae"])&(a["dmae_hi"]>=tr_all["dmae"]))))
        print(key,json.dumps({k:round(v,3) for k,v in summ[key].items()}))
    json.dump(summ,open("22_sim_S3.json","w"),indent=1)
if __name__=="__main__":
    if WHICH in ("S0","all"): S0()
    else: load_P()
    if WHICH in ("S1","all"): S1(REPS)
    if WHICH in ("S2","all"): S2(REPS)
    if WHICH in ("S3","all"): S3(REPS)
