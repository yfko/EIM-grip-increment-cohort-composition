# EIM 在人口學之上對握力預測的增量——分析程式

對應手稿：Kao H-H, Ko Y-F. *Small incremental value of electrical impedance myography over sex, age and body mass index for grip-strength prediction*（2026 年 9 月投稿 Muscle & Nerve）。

- 134 位 18–87 歲成人（功能健康 109、功能受損 25），前臂 100 kHz 的 R、X 共 8 個特徵。
- 特徵在訓練折內對性別、年齡、BMI 殘差化；受試者層級 10 × 5 折交叉驗證；ΔMAE 附 2,000 次受試者 bootstrap 區間。
- 預註冊（2026-09-04）在 `PREREGISTRATION_v1.1_zh-TW.md`；偏離事項在 `SUPPLEMENTARY_NOTES.md` Note 1。

**受試者層級資料一律不在此庫**（IRB CMUH114-REC3-142）。`.gitignore` 排除所有 `.csv`、`.xlsx`、`.npz`；每次 push 前執行 `python check_no_participant_data.py`。原始資料與 `cohort_audit.csv` 由本機路徑讀取，不隨程式散佈。

`analysis/` 的編號沿用開發時的順序（02–20），`outputs/` 為各程式的終端輸出與結果整理（只含彙總值）。權利聲明見 `README.md`。
