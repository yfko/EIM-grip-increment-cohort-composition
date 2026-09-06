# -*- coding: utf-8 -*-
"""方法篇模擬校準：從開發集 134 人抽出「彙總層級」的參數（不輸出任何受試者列）。
輸出：特徵數、群組結構、R/X/PhA/Z 代數冗餘的實測相關、群內／群間相關分布、人口學與特徵的相關、
人口學殘差化前後的特徵相關、握力對人口學的 R²。供 22_simulation.py 產生合成資料。"""
import re, glob, os, json, numpy as np, pandas as pd
ROOT="/Users/yfko/Library/Mobile Documents/com~apple~CloudDocs/Documents/New idea/EIM"
STATIC=f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/static_output"
m=pd.read_excel(f"{ROOT}/20260401復刻_投影片0903更新/02_data/raw/完整資料.xlsx"); m["id"]=m["id"].astype(str)
dom=m["慣用手"].astype(str).str.contains("右").map({True:"右",False:"左"})
gc=lambda s:[f"{s}{i}({f})" for f in (50,100) for i in (1,2,3)]
m["grip_L"]=m[gc("左")].max(axis=1); m["grip_R"]=m[gc("右")].max(axis=1)
m["grip_dom"]=np.where(dom=="右",m["grip_R"],m["grip_L"]); m["sex_male"]=(m["性別"]=="男").astype(int)
demo=m.set_index("id")[["sex_male","年齡","BMI","grip_dom"]].rename(columns={"年齡":"age"})
feat={}
for p in sorted(glob.glob(f"{STATIC}/*.csv")):
    fq=int(re.search(r"freq(\d+)kHz",os.path.basename(p)).group(1))
    d=pd.read_csv(p,encoding="utf-8-sig",dtype={"subject_id":str}).set_index("subject_id")
    for c in d.columns:
        mm=re.match(r"^(.+?)\((PhA|R|X)\)_Static$",c)
        if mm: feat[f"{mm.group(1)}|{mm.group(2)}|{fq}"]=d[c]
E=pd.DataFrame(feat); MUS=sorted({c.split("|")[0] for c in E.columns})
for mu in MUS:
    for fq in (50,100):
        cols=[f"{mu}|{p}|{fq}" for p in ("PhA","R","X")]
        bad=(E[cols[1]]<=0)|(E[cols[2]]<=0)|(E[cols[0]]<=0); E.loc[bad,cols]=np.nan
        lp=np.log(E[cols[0]]); z=(lp-lp.mean())/lp.std(ddof=1); E.loc[z.abs()>4,cols]=np.nan
df=demo.join(E,how="inner"); n=len(df)
def region(mu):
    if "腕肌" in mu: return "forearm"
    if "肱" in mu: return "upper_arm"
    if "豎脊" in mu: return "trunk"
    if "股" in mu: return "thigh"
    return "calf"
out={"n":n,"n_muscles":len(MUS),"channels":len(MUS)*2}
# 1. 代數冗餘：同通道 R、X、PhA、|Z| 的相關
rr=[]
for mu in MUS:
    for fq in (50,100):
        R,X,P=(df[f"{mu}|{p}|{fq}"] for p in ("R","X","PhA")); Z=np.sqrt(R**2+X**2)
        sub=pd.DataFrame({"R":R,"X":X,"PhA":P,"Z":Z}).dropna(); c=sub.corr()
        rr.append({"ch":f"{region(mu)}|{fq}","r_RX":c.loc["R","X"],"r_RPhA":c.loc["R","PhA"],"r_XPhA":c.loc["X","PhA"],"r_RZ":c.loc["R","Z"],"r_XZ":c.loc["X","Z"]})
rr=pd.DataFrame(rr); out["within_channel_corr_median"]=rr.drop(columns="ch").median().round(3).to_dict()
out["within_channel_corr_range"]={k:[round(rr[k].min(),3),round(rr[k].max(),3)] for k in rr.columns if k!="ch"}
# 2. 群組結構（R、X 只留）：群內 vs 群間相關
RX=[c for c in E.columns if c.split("|")[1] in ("R","X")]
grp={c:f"{region(c.split('|')[0])}|{c.split('|')[2]}" for c in RX}
C=df[RX].corr().abs(); within=[]; between=[]
for i,a in enumerate(RX):
    for b in RX[i+1:]:
        (within if grp[a]==grp[b] else between).append(C.loc[a,b])
out["groups"]=sorted(set(grp.values())); out["p_RX"]=len(RX)
out["abs_corr_within_group"]={"median":round(float(np.median(within)),3),"q25":round(float(np.percentile(within,25)),3),"q75":round(float(np.percentile(within,75)),3)}
out["abs_corr_between_group"]={"median":round(float(np.median(between)),3),"q25":round(float(np.percentile(between,25)),3),"q75":round(float(np.percentile(between,75)),3)}
# 3. 人口學與特徵：每個特徵對 sex+age+BMI 的 R²（阻抗編碼人口學的程度）
from sklearn.linear_model import LinearRegression
D=df[["sex_male","age","BMI"]].to_numpy(float); r2=[]
for c in RX:
    y=df[c].to_numpy(float); ok=~np.isnan(y); r2.append(LinearRegression().fit(D[ok],y[ok]).score(D[ok],y[ok]))
out["feature_R2_on_demographics"]={"median":round(float(np.median(r2)),3),"q25":round(float(np.percentile(r2,25)),3),"q75":round(float(np.percentile(r2,75)),3),"max":round(float(np.max(r2)),3)}
sexr=[abs(np.corrcoef(df["sex_male"],df[c].fillna(df[c].median()))[0,1]) for c in RX]
out["abs_corr_feature_sex"]={"median":round(float(np.median(sexr)),3),"max":round(float(np.max(sexr)),3)}
# 4. 殘差化後的群內／群間相關
Rz=pd.DataFrame(index=df.index)
for c in RX:
    y=df[c].to_numpy(float); ok=~np.isnan(y); lr=LinearRegression().fit(D[ok],y[ok]); res=np.full(n,np.nan); res[ok]=y[ok]-lr.predict(D[ok]); Rz[c]=res
Cz=Rz.corr().abs(); w2=[]; b2=[]
for i,a in enumerate(RX):
    for b in RX[i+1:]:
        (w2 if grp[a]==grp[b] else b2).append(Cz.loc[a,b])
out["abs_corr_within_group_residualized"]=round(float(np.median(w2)),3); out["abs_corr_between_group_residualized"]=round(float(np.median(b2)),3)
# 5. 結果變項：握力對人口學 R²、殘差 SD
y=df["grip_dom"].to_numpy(float); lr=LinearRegression().fit(D,y); out["grip_R2_demographics_insample"]=round(lr.score(D,y),3); out["grip_resid_sd"]=round(float(np.std(y-lr.predict(D),ddof=4)),2); out["grip_sd"]=round(float(np.std(y,ddof=1)),2)
out["demographics"]={"p_male":round(float(df.sex_male.mean()),3),"age_mean":round(float(df.age.mean()),1),"age_sd":round(float(df.age.std()),1),"bmi_mean":round(float(df.BMI.mean()),1),"bmi_sd":round(float(df.BMI.std()),1)}
out["missing_cells_RX"]=int(df[RX].isna().sum().sum())
json.dump(out,open(f"{ROOT}/stage2_audit/21_sim_calibration.json","w"),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False,indent=1))
