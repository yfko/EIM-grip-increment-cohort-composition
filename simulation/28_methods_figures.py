# -*- coding: utf-8 -*-
"""方法篇 Fig 1（流程示意）、Fig 5（實證層級森林圖＋SGL 群組）、Fig 6（三法部位×頻率熱圖）。灰階可辨。輸出 PNG + PDF（向量）。"""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT="/Users/yfko/Library/Mobile Documents/com~apple~CloudDocs/Documents/New idea/EIM"; FIG=f"{ROOT}/stage2_paper/methods/figures/"
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
def save(fig,name): fig.savefig(FIG+name+".png",dpi=300); fig.savefig(FIG+name+".pdf"); plt.close(fig)
# ── Fig 1 流程示意 ──
fig,ax=plt.subplots(figsize=(7.4,2.4)); ax.axis("off")
boxes=[(0.01,"Input\n18 channels × 2 frequencies\nR, X, phase angle, |Z|\n144 candidate features"),(0.26,"1  De-duplication\nkeep R and X only\n72 features"),(0.51,"2  Confound adjustment\nresidualize features and\noutcome on sex, age, BMI\ninside each training fold"),(0.76,"3  Penalized model\nlasso, elastic net or\nsparse-group lasso\n10 site × frequency groups")]
for i,(x,t) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((x,0.22),0.225,0.62,boxstyle="round,pad=0.008",fc="0.92" if i==0 else "white",ec="k",lw=0.9)); ax.text(x+0.1125,0.53,t,ha="center",va="center",fontsize=6.6)
    if i<3: ax.add_patch(FancyArrowPatch((x+0.228,0.53),(x+0.258,0.53),arrowstyle="-|>",mutation_scale=9,color="k"))
ax.text(0.5,0.07,"Subject-level 10 × 5-fold cross-validation; increment over the demographic baseline (ΔMAE) with subject bootstrap",ha="center",fontsize=7,style="italic"); ax.set_xlim(0,1); ax.set_ylim(0,1); save(fig,"Fig1_pipeline")
# ── Fig 5 森林圖 ──
R=json.load(open(f"{ROOT}/stage2_audit/11_models_results.json"))
tiers=[("0","Lasso, raw forearm + demographics"),("1a","Elastic net, residualized forearm (pre-specified)"),("1c","PLS, residualized forearm"),("2a","XGBoost, residualized forearm + demographics"),("2b","XGBoost, monotone"),("2c","Kernel ridge (RBF), residualized forearm"),("3","Bilateral mixed model, per hand"),("1d","Lasso, residualized all 72"),("1e","Elastic net, residualized all 72"),("1b","Sparse-group lasso, residualized all 72 (10 groups)")]
fig,ax=plt.subplots(1,2,figsize=(7.8,4.1),gridspec_kw={"width_ratios":[1.5,1]})
ys=np.arange(len(tiers))[::-1]
for y,(k,lab) in zip(ys,tiers):
    d=R[k]; ax[0].errorbar(d["dmae"],y,xerr=[[d["dmae"]-d["dmae_lo"]],[d["dmae_hi"]-d["dmae"]]],fmt="s" if k in ("1d","1e","1b") else "o",color="k",mfc="k" if k in ("1a","1b") else "white",capsize=2,ms=5)
ax[0].set_yticks(ys); ax[0].set_yticklabels([t[1] for t in tiers],fontsize=7.5); ax[0].axvline(0,ls=":",c="k",lw=0.8); ax[0].set_xlabel("ΔMAE versus demographic baseline (kg), 95% subject bootstrap"); ax[0].set_title("A  Empirical tiers (n = 134)",fontsize=9,loc="left")
sg=np.load(f"{ROOT}/stage2_audit/13_sgl_norms.npz",allow_pickle=True); groups=[str(g) for g in sg["groups"]]; sel=sg["sel"].mean(0); share=sg["norms"].mean(0)/sg["norms"].mean(0).sum()
order=np.argsort(share)[::-1]; yy=np.arange(len(groups))[::-1]
ax[1].barh(yy,share[order]*100,color="0.35",edgecolor="k",label="coefficient-norm share (%)"); ax[1].plot(sel[order]*100,yy,"o",mfc="white",mec="k",label="selection rate (%)")
ax[1].set_yticks(yy); ax[1].set_yticklabels([groups[i].replace("前臂","forearm").replace("上臂","upper arm").replace("軀幹","trunk").replace("大腿","thigh").replace("小腿","calf").replace("・"," · ").replace("|"," · ") for i in order],fontsize=7.5); ax[1].set_xlabel("%"); ax[1].legend(fontsize=7,loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=1,frameon=False); ax[1].set_title("B  SGL groups (50 folds)",fontsize=9,loc="left")
fig.tight_layout(); save(fig,"Fig5_empirical_tiers")
# ── Fig 6 三法熱圖 ──
t=pd.read_csv(f"{ROOT}/stage2_audit/26_site_freq_correspondence_table.csv",encoding="utf-8-sig",index_col=0)
fig,ax=plt.subplots(1,4,figsize=(7.4,2.9),gridspec_kw={"width_ratios":[1.6,1,1,1]})
for f,mk in ((50,"o"),(100,"s")):
    tt=t[t.freq==f]; ax[0].scatter(tt["S1_pct_20_70"],tt["XGB_shap_share"]*100,marker=mk,facecolors="white" if f==50 else "0.3",edgecolors="k",label=f"{f} kHz",s=22)
ax[0].axvline(0,ls=":",c="k",lw=0.7); ax[0].set_xlabel("Stage 1: change in LLN 20→70 y (%)",fontsize=8); ax[0].set_ylabel("Mean |SHAP| share (%)",fontsize=8); ax[0].legend(fontsize=7,loc="upper left",bbox_to_anchor=(0.0,1.0),frameon=True); ax[0].set_ylim(0,14); ax[0].set_title("A  36 channels",fontsize=9,loc="left")
regs=["前臂","上臂","軀幹","大腿","小腿"]; lab=["forearm","upper arm","trunk","thigh","calf"]
for a,(nm,ttl) in zip(ax[1:],(("EN_abscoef","B  Elastic net |β|"),("XGB_perm","C  XGB permutation"),("XGB_shap","D  XGB |SHAP|"))):
    H=(t.groupby(["region","freq"])[nm].sum().unstack().loc[regs]/t[nm].sum()*100)
    a.imshow(H.values,cmap="Greys",aspect="auto",vmin=0,vmax=32); a.set_xticks([0,1]); a.set_xticklabels(["50","100"],fontsize=8); a.set_yticks(range(5)); a.set_yticklabels(lab if nm=="EN_abscoef" else [""]*5,fontsize=8); a.set_title(ttl,fontsize=9,loc="left")
    for i in range(5):
        for j in range(2): a.text(j,i,f"{H.values[i,j]:.0f}",ha="center",va="center",fontsize=7.5,color="w" if H.values[i,j]>16 else "k")
    a.set_xlabel("kHz",fontsize=8)
fig.tight_layout(); save(fig,"Fig6_site_frequency"); print("figures written")
