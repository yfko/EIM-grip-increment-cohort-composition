# -*- coding: utf-8 -*-
"""模擬結果圖（方法篇草圖，灰階可辨，PM 規定不靠顏色）。輸出 stage2_paper/methods/figures/SimFig1-3.png"""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="../stage2_paper/methods/figures/"; plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
s1=json.load(open("22_sim_S1.json")); s2=json.load(open("22_sim_S2.json")); s3=json.load(open("22_sim_S3.json"))
# Fig 1: S1 selection stability & CI width
fig,ax=plt.subplots(1,3,figsize=(7.2,2.4)); labs=["{R, X}","{R, X, PhA, |Z|}"]; keys=["RX","RXPhAZ"]
for a,(m,tt) in zip(ax,[("sel_true","True-signal feature\nselection rate"),("jaccard","Between-fold Jaccard"),("ciw_dr2","Bootstrap 95% CI width\nfor ΔR²")]):
    v=[s1[k][m] for k in keys]; a.bar(labs,v,color=["0.35","0.75"],edgecolor="k",hatch=["","//"]); a.set_title(tt); a.set_ylim(0,max(v)*1.25)
    for i,x in enumerate(v): a.text(i,x*1.03,f"{x:.2f}",ha="center")
fig.suptitle("S1: algebraically redundant features (n = 134, Lasso, 500 replicates)",fontsize=9); fig.tight_layout(); fig.savefig(OUT+"SimFig1.png",dpi=300); fig.savefig(OUT+"SimFig1.pdf"); plt.close(fig)
# Fig 2: S2 true vs false group rate per model, 4 cells
fig,ax=plt.subplots(1,4,figsize=(7.4,2.5),sharey=True); cells=["n134_weak","n134_strong","n300_weak","n300_strong"]; models=["Lasso","EN","SGL"]; x=np.arange(3)
for a,c in zip(ax,cells):
    t=[s2[c][m]["true_group_rate"] for m in models]; f=[s2[c][m]["false_group_rate"] for m in models]
    a.bar(x-0.2,t,0.4,color="0.35",edgecolor="k",label="true groups"); a.bar(x+0.2,f,0.4,color="0.85",edgecolor="k",hatch="//",label="false groups")
    a.set_xticks(x); a.set_xticklabels(models); a.set_title(c.replace("_",", "),fontsize=9); a.set_ylim(0,1.05)
ax[0].set_ylabel("Group selection rate"); ax[0].legend(fontsize=7,loc="lower left"); fig.suptitle("S2: group-structured signal (300 replicates); SGL raises specificity, not sensitivity",fontsize=9); fig.tight_layout(); fig.savefig(OUT+"SimFig2.png",dpi=300); fig.savefig(OUT+"SimFig2.pdf"); plt.close(fig)
# Fig 3: S3 false-null rates, 2x2x(estimator x residualization)
fig,ax=plt.subplots(1,4,figsize=(7.6,2.7),sharey=True)
for a,(mod,ttl) in zip(ax,[("LassoRaw","Lasso, raw"),("LassoRes","Lasso, residualized"),("ENraw","Elastic net, raw"),("ENres","Elastic net, residualized")]):
    vals=np.array([[s3[f"{coh}|{fs}|{mod}"]["false_null_dr2"] for fs in ("RX","RXPhA")] for coh in ("all","healthy")])
    x=np.arange(2); a.bar(x-0.2,vals[0],0.4,color="0.35",edgecolor="k",label="all 134"); a.bar(x+0.2,vals[1],0.4,color="0.85",edgecolor="k",hatch="//",label="healthy only")
    a.set_xticks(x); a.set_xticklabels(["{R, X}","{R, X, PhA}"],fontsize=8); a.set_title(ttl,fontsize=8.5); a.axhline(0.05,ls=":",c="k",lw=0.8); a.set_ylim(0,0.45)
    for i in range(2):
        for j,off in ((0,-0.2),(1,0.2)): a.text(i+off,vals[j][i]+0.01,f"{vals[j][i]:.2f}",ha="center",fontsize=6.5)
ax[0].set_ylabel("P(estimated ΔR² ≤ 0 | true ΔR² > 0)",fontsize=8); ax[0].legend(fontsize=7,loc="upper left"); fig.suptitle("S3: spurious loss of a true increment (500 replicates); residualization, not the estimator, restores it",fontsize=8.5); fig.tight_layout(); fig.savefig(OUT+"SimFig3.png",dpi=300); fig.savefig(OUT+"SimFig3.pdf"); plt.close(fig)
print("figures written")
