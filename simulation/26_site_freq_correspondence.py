# -*- coding: utf-8 -*-
"""部位×頻率對照（探索性，PI 2026-09-06 概念）：階段一各通道年齡敏感度 vs 階段二各通道握力訊息（殘差化後）。
全部重算，不引用研究生交接輸出。設定同 11_models（10×5 折、FWL 殘差化、全 72 特徵）。
輸出：26_site_freq_correspondence.json / _out.txt / _table.csv，圖 stage2_paper/methods/figures/SiteFreqFig.png"""
import re, glob, os, json, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr, wilcoxon
from xgboost import XGBRegressor
import shap
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
ROOT="/Users/yfko/Library/Mobile Documents/com~apple~CloudDocs/Documents/New idea/EIM"
STATIC=f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/static_output"; OUT=f"{ROOT}/stage2_audit"; FIG=f"{ROOT}/stage2_paper/methods/figures"
SEED,N_REP,N_FOLD=20260904,10,5
# ── 資料（同 11_models） ──
m=pd.read_excel(f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/完整資料.xlsx"); m["id"]=m["id"].astype(str)
dom=m["慣用手"].astype(str).str.contains("右").map({True:"右",False:"左"})
gc=lambda s:[f"{s}{i}({f})" for f in (50,100) for i in (1,2,3)]
m["grip_L"]=m[gc("左")].max(axis=1); m["grip_R"]=m[gc("右")].max(axis=1); m["grip_dom"]=np.where(dom=="右",m["grip_R"],m["grip_L"]); m["sex_male"]=(m["性別"]=="男").astype(int)
demo=m.set_index("id")[["sex_male","年齡","BMI","grip_dom"]].rename(columns={"年齡":"age"})
feat={}
for p in sorted(glob.glob(f"{STATIC}/*.csv")):
    fq=int(re.search(r"freq(\d+)kHz",os.path.basename(p)).group(1)); d=pd.read_csv(p,encoding="utf-8-sig",dtype={"subject_id":str}).set_index("subject_id")
    for c in d.columns:
        mm=re.match(r"^(.+?)\((PhA|R|X)\)_Static$",c)
        if mm: feat[f"{mm.group(1)}|{mm.group(2)}|{fq}"]=d[c]
E=pd.DataFrame(feat); MUS=sorted({c.split("|")[0] for c in E.columns})
for mu in MUS:
    for fq in (50,100):
        cols=[f"{mu}|{p}|{fq}" for p in ("PhA","R","X")]; bad=(E[cols[1]]<=0)|(E[cols[2]]<=0)|(E[cols[0]]<=0); E.loc[bad,cols]=np.nan
        lp=np.log(E[cols[0]]); z=(lp-lp.mean())/lp.std(ddof=1); E.loc[z.abs()>4,cols]=np.nan
E=E[[c for c in E.columns if c.split("|")[1] in ("R","X")]]; df=demo.join(E,how="inner"); DEMO=["sex_male","age","BMI"]
def region(mu):
    if "腕肌" in mu: return "前臂"
    if "肱" in mu: return "上臂"
    if "豎脊" in mu: return "軀幹"
    if "股" in mu: return "大腿"
    return "小腿"
COLS=list(E.columns); CHAN=[f"{c.split('|')[0]}|{c.split('|')[2]}" for c in COLS]; CHANNELS=sorted(set(CHAN),key=lambda s:(region(s.split("|")[0]),s))
y=df["grip_dom"].to_numpy(float); n=len(y); Dm=df[DEMO].to_numpy(float); X=df[COLS].to_numpy(float)
strat=(df["sex_male"].astype(str)+"_"+(df["grip_dom"]>df["grip_dom"].median()).astype(int).astype(str)).to_numpy()
XGRID=dict(max_depth=[2,3],learning_rate=[0.05],n_estimators=[200],min_child_weight=[3,6],reg_lambda=[1,10])
imp_en=[]; imp_perm=[]; imp_shap=[]; sel_en=[]
for r in range(N_REP):
    for tr,te in StratifiedKFold(N_FOLD,shuffle=True,random_state=SEED+r).split(np.zeros(n),strat):
        imp=SimpleImputer(strategy="median").fit(X[tr]); Xtr,Xte=imp.transform(X[tr]),imp.transform(X[te])
        oy=LinearRegression().fit(Dm[tr],y[tr]); ystar=y[tr]-oy.predict(Dm[tr]); yte_star=y[te]-oy.predict(Dm[te])
        ox=LinearRegression().fit(Dm[tr],Xtr); Xs_tr=Xtr-ox.predict(Dm[tr]); Xs_te=Xte-ox.predict(Dm[te])
        sc=StandardScaler().fit(Xs_tr); Ztr,Zte=sc.transform(Xs_tr),sc.transform(Xs_te)
        en=ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9],n_alphas=100,cv=KFold(5,shuffle=True,random_state=SEED+r),max_iter=50000,random_state=SEED+r).fit(Ztr,ystar)
        imp_en.append(np.abs(en.coef_)); sel_en.append(np.abs(en.coef_)>1e-8)
        gs=GridSearchCV(XGBRegressor(objective="reg:squarederror",subsample=0.9,colsample_bytree=0.8,random_state=SEED+r,n_jobs=2,verbosity=0),XGRID,cv=KFold(5,shuffle=True,random_state=SEED+r),scoring="neg_mean_absolute_error",n_jobs=3).fit(Ztr,ystar)
        xgb=gs.best_estimator_
        pi=permutation_importance(xgb,Zte,yte_star,n_repeats=10,random_state=SEED+r,scoring="neg_mean_absolute_error"); imp_perm.append(np.maximum(pi.importances_mean,0))
        sv=shap.TreeExplainer(xgb).shap_values(Zte); imp_shap.append(np.abs(sv).mean(0))
    print(f"rep {r+1}/{N_REP} done",flush=True)
imp_en=np.array(imp_en); imp_perm=np.array(imp_perm); imp_shap=np.array(imp_shap); sel_en=np.array(sel_en)
def to_channel(M):
    ch=pd.DataFrame(M,columns=COLS).T; ch["chan"]=CHAN; return ch.groupby("chan").sum().loc[CHANNELS]
tab=pd.DataFrame(index=CHANNELS); tab["muscle"]=[c.split("|")[0] for c in CHANNELS]; tab["freq"]=[int(c.split("|")[1]) for c in CHANNELS]; tab["region"]=[region(c.split("|")[0]) for c in CHANNELS]
for nm,M in (("EN_abscoef",imp_en),("XGB_perm",imp_perm),("XGB_shap",imp_shap)):
    C=to_channel(M); tab[nm]=C.mean(1).values; tab[nm+"_share"]=tab[nm]/tab[nm].sum()
tab["EN_sel_rate"]=to_channel(sel_en.astype(float)).mean(1).values/2   # 兩個特徵各自的選入率平均
# ── 階段一：20→70 歲 LLN 變化 % 與線性年齡係數 ──
rv=pd.read_csv(f"{ROOT}/stage1_rebuild/outputs/main_109/reference_values_wide.csv",encoding="utf-8-sig"); rv["chan"]=rv["muscle"]+"|"+rv["frequency_khz"].astype(int).astype(str)
rv["pct_20_70"]=100*(rv["LLN_age_70"]-rv["LLN_age_20"])/rv["LLN_age_20"]; rv=rv.set_index("chan")
mc=pd.read_csv(f"{ROOT}/stage1_rebuild/outputs/main_109/model_coefficients_and_diagnostics.csv",encoding="utf-8-sig"); mc["chan"]=mc["muscle"]+"|"+mc["frequency_khz"].astype(int).astype(str); mc=mc.set_index("chan")
tab["S1_pct_20_70"]=rv.loc[tab.index,"pct_20_70"].values; tab["S1_coef_age1"]=mc.loc[tab.index,"coef_age^1"].values; tab["S1_r2"]=mc.loc[tab.index,"r_squared"].values
tab.to_csv(f"{OUT}/26_site_freq_correspondence_table.csv",encoding="utf-8-sig")
# ── 對照統計 ──
res={"n_channels":len(tab),"n_folds":len(imp_en)}
lines=[f"部位×頻率對照（36 通道；階段二重要性為 10×5 折平均；殘差化 72 特徵）\n"]
for nm in ("EN_abscoef","XGB_perm","XGB_shap"):
    r_all=spearmanr(tab["S1_pct_20_70"],tab[nm]); r50=spearmanr(tab.loc[tab.freq==50,"S1_pct_20_70"],tab.loc[tab.freq==50,nm]); r100=spearmanr(tab.loc[tab.freq==100,"S1_pct_20_70"],tab.loc[tab.freq==100,nm])
    res[nm]={"spearman_all":[r_all.statistic,r_all.pvalue],"spearman_50":[r50.statistic,r50.pvalue],"spearman_100":[r100.statistic,r100.pvalue]}
    lines.append(f"{nm:12s} Spearman(年齡 Δ% 20→70, 重要性)：全部 rho={r_all.statistic:+.3f} p={r_all.pvalue:.3f}｜50 kHz rho={r50.statistic:+.3f} p={r50.pvalue:.3f}｜100 kHz rho={r100.statistic:+.3f} p={r100.pvalue:.3f}")
    # 頻率邊際：同肌肉 100 vs 50 配對
    piv=tab.pivot(index="muscle",columns="freq",values=nm); w=wilcoxon(piv[100],piv[50]); n_hi=int((piv[100]>piv[50]).sum())
    res[nm]["freq_100_gt_50"]=[n_hi,len(piv)]; res[nm]["wilcoxon_p"]=w.pvalue; res[nm]["share_100"]=float(tab.loc[tab.freq==100,nm].sum()/tab[nm].sum())
    lines.append(f"{'':12s} 頻率邊際：100 kHz 重要性 > 50 kHz 的肌肉 {n_hi}/{len(piv)}，Wilcoxon p={w.pvalue:.3f}；100 kHz 佔總重要性 {res[nm]['share_100']:.2f}（階段一：17/18 肌肉 100 kHz 年齡斜率更負）")
    reg=tab.groupby(["region","freq"])[nm].sum().unstack(); res[nm]["region_by_freq"]=reg.round(4).to_dict()
    lines.append("             部位×頻率重要性總和：\n"+reg.round(3).to_string())
s1reg=tab.groupby(["region","freq"])["S1_pct_20_70"].mean().unstack(); lines.append("\n階段一 部位×頻率 平均 Δ% 20→70：\n"+s1reg.round(2).to_string()); res["S1_region_by_freq"]=s1reg.round(3).to_dict()
top=tab.sort_values("XGB_shap",ascending=False).head(8)[["region","S1_pct_20_70","EN_abscoef_share","XGB_perm_share","XGB_shap_share","EN_sel_rate"]]
lines.append("\nSHAP 前 8 通道（與階段一 Δ% 對照）：\n"+top.round(3).to_string())
for k in ("軀幹|100","前臂|100","小腿|50"):
    pass
open(f"{OUT}/26_site_freq_correspondence_out.txt","w",encoding="utf-8").write("\n".join(lines)); json.dump(res,open(f"{OUT}/26_site_freq_correspondence.json","w"),ensure_ascii=False,indent=1,default=float)
print("\n".join(lines))
# ── 圖：散布（分頻率）＋ 部位×頻率熱圖（灰階） ──
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"font.family":"Heiti TC"})
fig,ax=plt.subplots(1,2,figsize=(7.4,3.2))
for f,mk in ((50,"o"),(100,"s")):
    t=tab[tab.freq==f]; ax[0].scatter(t["S1_pct_20_70"],t["XGB_shap_share"]*100,marker=mk,facecolors="white" if f==50 else "0.3",edgecolors="k",label=f"{f} kHz")
ax[0].axvline(0,ls=":",c="k",lw=0.7); ax[0].set_xlabel("Stage 1: change in LLN 20→70 y (%)"); ax[0].set_ylabel("Stage 2: mean |SHAP| share (%)"); ax[0].legend(fontsize=8); ax[0].set_title("36 channels",fontsize=9)
H=tab.groupby(["region","freq"])["XGB_shap_share"].sum().unstack().loc[["前臂","上臂","軀幹","大腿","小腿"]]*100
im=ax[1].imshow(H.values,cmap="Greys",aspect="auto"); ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["50 kHz","100 kHz"]); ax[1].set_yticks(range(5)); ax[1].set_yticklabels(["forearm","upper arm","trunk","thigh","calf"])
for i in range(5):
    for j in range(2): ax[1].text(j,i,f"{H.values[i,j]:.1f}",ha="center",va="center",color="w" if H.values[i,j]>H.values.max()/2 else "k",fontsize=8)
ax[1].set_title("Grip-information share by site × frequency (%)",fontsize=9); fig.tight_layout(); fig.savefig(f"{FIG}/SiteFreqFig.png",dpi=300); print("figure saved")
