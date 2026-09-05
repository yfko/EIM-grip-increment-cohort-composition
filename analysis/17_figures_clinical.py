# -*- coding: utf-8 -*-
"""臨床篇圖 1–4 + 非慣用手 H1 模型敏感度 + ROC 機率重算。唯讀（只寫 stage2_paper/clinical/figures）。"""
import sys,json,numpy as np
sys.argv=["11_models.py","none"]
src=open(__file__.replace("17_figures_clinical.py","11_models.py"),encoding="utf-8").read()
exec(src.split("# ── 執行")[0])
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FIG="/Users/yfko/Library/Mobile Documents/com~apple~CloudDocs/Documents/New idea/EIM/stage2_paper/clinical/figures/"
plt.rcParams.update({"font.family":"Arial","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
def mk(s): return Pipeline([("sc",StandardScaler()),("en",ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9],n_alphas=100,cv=inner(s),max_iter=50000,random_state=s))])
def boot_dm(idx,yy,pb,pn):
    dm=[]
    for _ in range(N_BOOT):
        b=rng.choice(idx,len(idx),replace=True); dm.append(np.abs(yy[b]-pb[b]).mean()-np.abs(yy[b]-pn[b]).mean())
    dm=np.array(dm); return dm.mean(),np.percentile(dm,2.5),np.percentile(dm,97.5)
# ── 非慣用手 H1 敏感度 ──
y_nd=df["grip_nd"].to_numpy(float); Dm=df[DEMO].to_numpy(float); X=df[FORE100].to_numpy(float)
strat_nd=(df["sex_male"].astype(str)+"_"+(df["grip_nd"]>df["grip_nd"].median()).astype(int).astype(str)).to_numpy()
PB=np.empty((N_REP,n)); PN=np.empty((N_REP,n))
for r in range(N_REP):
    for tr,te in StratifiedKFold(N_FOLD,shuffle=True,random_state=SEED+r).split(np.zeros(n),strat_nd):
        imp=SimpleImputer(strategy="median").fit(X[tr]); Xtr,Xte=imp.transform(X[tr]),imp.transform(X[te])
        oy=LinearRegression().fit(Dm[tr],y_nd[tr]); PB[r,te]=oy.predict(Dm[te]); ox=LinearRegression().fit(Dm[tr],Xtr)
        m_=mk(SEED+r).fit(Xtr-ox.predict(Dm[tr]),y_nd[tr]-oy.predict(Dm[tr])); PN[r,te]=PB[r,te]+np.ravel(m_.predict(Xte-ox.predict(Dm[te])))
pb_nd,pn_nd=PB.mean(0),PN.mean(0); nd=boot_dm(np.arange(n),y_nd,pb_nd,pn_nd)
print(f"非慣用手 H1：基準 R² {r2(y_nd,pb_nd):.3f} MAE {np.abs(y_nd-pb_nd).mean():.2f} → +EIM R² {r2(y_nd,pn_nd):.3f} MAE {np.abs(y_nd-pn_nd).mean():.2f} | ΔMAE {nd[0]:+.2f} [{nd[1]:+.2f},{nd[2]:+.2f}]")
# ── ROC 機率（tier 4 重算）──
w=df["weak"].to_numpy(int); PBc=np.zeros(n); PEc=np.zeros(n)
for r in range(N_REP):
    for tr,te in StratifiedKFold(N_FOLD,shuffle=True,random_state=SEED+r).split(np.zeros(n),w):
        imp=SimpleImputer(strategy="median").fit(X[tr]); Xtr,Xte=imp.transform(X[tr]),imp.transform(X[te]); ox=LinearRegression().fit(Dm[tr],Xtr)
        lb=Pipeline([("sc",StandardScaler()),("lr",LogisticRegressionCV(Cs=np.logspace(-3,2,6),cv=inner(SEED+r),max_iter=5000,scoring="roc_auc"))]).fit(Dm[tr],w[tr])
        le=Pipeline([("sc",StandardScaler()),("lr",LogisticRegressionCV(Cs=np.logspace(-3,2,6),cv=inner(SEED+r),max_iter=5000,scoring="roc_auc"))]).fit(np.hstack([Dm[tr],Xtr-ox.predict(Dm[tr])]),w[tr])
        PBc[te]+=lb.predict_proba(Dm[te])[:,1]; PEc[te]+=le.predict_proba(np.hstack([Dm[te],Xte-ox.predict(Dm[te])]))[:,1]
PBc/=N_REP; PEc/=N_REP; print(f"ROC AUC 人口學 {roc_auc_score(w,PBc):.3f}  +EIM {roc_auc_score(w,PEc):.3f}")
R=json.load(open(f"{OUT}/11_models_results.json"))
# ── Fig 1 流程圖 ──
fig,ax=plt.subplots(figsize=(6.5,4.2)); ax.axis("off")
def box(x,y,w_,h,t,fc="white",lw=1.2,fs=9,bold=False):
    ax.add_patch(plt.Rectangle((x,y),w_,h,fc=fc,ec="black",lw=lw)); ax.text(x+w_/2,y+h/2,t,ha="center",va="center",fontsize=fs,fontweight=("bold" if bold else "normal"))
box(0.05,0.78,0.9,0.16,"Adults assessed and analysed\nn = 134 (68 men, 66 women), 18–87 years",fc="#e8e8e8",bold=True)
ax.annotate("",xy=(0.3,0.62),xytext=(0.3,0.78),arrowprops=dict(arrowstyle="-|>",lw=1.2)); ax.annotate("",xy=(0.7,0.62),xytext=(0.7,0.78),arrowprops=dict(arrowstyle="-|>",lw=1.2))
box(0.05,0.36,0.42,0.26,"Functionally healthy\nn = 109\nGrip ≥ AWGS 2025 threshold,\nSARC-F < 4, SPPB ≥ 10",fc="#f4f4f4")
box(0.53,0.36,0.42,0.26,"Functionally impaired\nn = 25\n21 grip below threshold,\n3 SARC-F ≥ 4, 8 SPPB < 10\n(criteria overlap)",fc="#f4f4f4",fs=8.5)
ax.text(0.5,0.22,"Primary analysis: all 134. Cohort-composition analysis: trained within the 109,\nor on all 134, and evaluated in each subgroup (Table 3).",ha="center",va="center",fontsize=8.5)
ax.text(0.5,0.06,"The 109 form the cohort of the companion reference-limit study.",ha="center",va="center",fontsize=8,color="#444444")
plt.savefig(FIG+"Figure1.png",dpi=600,bbox_inches="tight"); plt.savefig(FIG+"Figure1.tif",dpi=600,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"}); plt.close()
# ── Fig 2 森林圖 ──
rows=[("Lasso (published protocol), dominant",R["0"]),("Elastic net (pre-specified), dominant",R["1a"]),("XGBoost, dominant",R["2a"]),("Elastic net, non-dominant (sensitivity)",{"dmae":nd[0],"dmae_lo":nd[1],"dmae_hi":nd[2]})]
fig,ax=plt.subplots(figsize=(7.2,2.6))
for i,(lab,d) in enumerate(rows[::-1]):
    ax.errorbar(d["dmae"],i,xerr=[[d["dmae"]-d["dmae_lo"]],[d["dmae_hi"]-d["dmae"]]],fmt="o",color="black",capsize=3,ms=5); ax.text(-0.05,i,lab,ha="right",va="center",fontsize=8.5)
ax.axvline(0,color="#888888",lw=0.8,ls="--"); ax.set_yticks([]); ax.set_xlim(-0.3,1.2); ax.set_xlabel("Reduction in MAE vs demographic baseline, kg (95% CI)",fontsize=8.5); ax.spines["left"].set_visible(False)
plt.subplots_adjust(left=0.5); plt.savefig(FIG+"Figure2.png",dpi=600,bbox_inches="tight"); plt.savefig(FIG+"Figure2.tif",dpi=600,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"}); plt.close()
# ── Fig 3 六格 ──
cells=[("Train 109 → healthy 109 (CV)",0.33,0.06,0.58),("Train 134 → healthy 109",0.32,-0.01,0.66),("Train 134 → impaired 25",1.16,0.22,2.14),("Train 134 → grip below threshold 21",1.44,0.32,2.53),("Train 109 → impaired 25 (post hoc)",1.53,0.78,2.37),("Train 109 → grip below threshold 21 (post hoc)",1.87,1.00,2.80)]
fig,ax=plt.subplots(figsize=(7.6,3.2))
for i,(lab,m_,lo,hi) in enumerate(cells[::-1]):
    c="#b03a2e" if "impaired" in lab or "threshold" in lab else "black"
    ax.errorbar(m_,i,xerr=[[m_-lo],[hi-m_]],fmt="o",color=c,capsize=3,ms=5); ax.text(-0.1,i,lab,ha="right",va="center",fontsize=8.5)
ax.axvline(0,color="#888888",lw=0.8,ls="--"); ax.set_yticks([]); ax.set_xlim(-0.4,3.0); ax.set_xlabel("Reduction in MAE vs demographic baseline, kg (95% CI)",fontsize=8.5); ax.spines["left"].set_visible(False)
plt.subplots_adjust(left=0.55); plt.savefig(FIG+"Figure3.png",dpi=600,bbox_inches="tight"); plt.savefig(FIG+"Figure3.tif",dpi=600,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"}); plt.close()
# ── Fig 4 ROC ──
fig,ax=plt.subplots(figsize=(4.2,4.2))
for p,lab,c in [(PBc,f"Sex, age, BMI (AUC {roc_auc_score(w,PBc):.3f})","#888888"),(PEc,f"+ forearm EIM (AUC {roc_auc_score(w,PEc):.3f})","black")]:
    fpr,tpr,_=roc_curve(w,p); ax.plot(fpr,tpr,color=c,lw=1.6,label=lab)
ax.plot([0,1],[0,1],ls="--",color="#cccccc",lw=0.8); ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity"); ax.set_aspect("equal"); ax.legend(loc="lower right",frameon=False,fontsize=8)
plt.savefig(FIG+"Figure4.png",dpi=600,bbox_inches="tight"); plt.savefig(FIG+"Figure4.tif",dpi=600,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"}); plt.close()
json.dump({"nondominant_h1":{"r2_base":r2(y_nd,pb_nd),"r2_eim":r2(y_nd,pn_nd),"mae_base":float(np.abs(y_nd-pb_nd).mean()),"mae_eim":float(np.abs(y_nd-pn_nd).mean()),"dmae":nd[0],"lo":nd[1],"hi":nd[2]},"auc_base":roc_auc_score(w,PBc),"auc_eim":roc_auc_score(w,PEc)},open(f"{OUT}/17_figures_extra.json","w"),indent=1)
print("圖 1–4 已輸出（PNG + 600 dpi TIFF）")
