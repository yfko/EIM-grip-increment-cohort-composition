# S3 v1.3（run2）vs v1.5（κ=1，與 v1.4 同 seed）逐欄對照

相異欄位 17，最大絕對差 0.032

| Cell | Field | v1.3 | v1.4/v1.5 (κ=1) | Diff |
|---|---|---|---|---|
| all|RX|LassoRaw | n | 134.000 | 134.000 | +0.000 |
| all|RX|LassoRaw | dr2_mean | 0.046 | 0.046 | +0.000 |
| all|RX|LassoRaw | dmae_mean | 0.296 | 0.296 | +0.000 |
| all|RX|LassoRaw | false_null_dr2 | 0.088 | 0.088 | +0.000 |
| all|RX|LassoRaw | false_null_dmae | 0.812 | 0.812 | +0.000 |
| all|RX|LassoRaw | ciw_dr2 | 0.133 | 0.133 | +0.000 |
| all|RX|LassoRaw | cover_dmae_truthall | 0.794 | 0.794 | +0.000 |
| all|RX|ENres | n | 134.000 | 134.000 | +0.000 |
| all|RX|ENres | dr2_mean | 0.053 | 0.053 | +0.000 |
| all|RX|ENres | dmae_mean | 0.343 | 0.343 | +0.000 |
| all|RX|ENres | false_null_dr2 | 0.046 | 0.046 | +0.000 |
| all|RX|ENres | false_null_dmae | 0.660 | 0.660 | +0.000 |
| all|RX|ENres | ciw_dr2 | 0.107 | 0.107 | +0.000 |
| all|RX|ENres | cover_dmae_truthall | 0.768 | 0.768 | +0.000 |
| all|RXPhA|LassoRaw | n | 134.000 | 134.000 | +0.000 |
| all|RXPhA|LassoRaw | dr2_mean | 0.036 | 0.036 | +0.000 |
| all|RXPhA|LassoRaw | dmae_mean | 0.229 | 0.229 | +0.000 |
| all|RXPhA|LassoRaw | false_null_dr2 | 0.164 | 0.164 | +0.000 |
| all|RXPhA|LassoRaw | false_null_dmae | 0.902 | 0.902 | +0.000 |
| all|RXPhA|LassoRaw | ciw_dr2 | 0.142 | 0.142 | -0.001 |
| all|RXPhA|LassoRaw | cover_dmae_truthall | 0.762 | 0.758 | -0.004 |
| all|RXPhA|ENres | n | 134.000 | 134.000 | +0.000 |
| all|RXPhA|ENres | dr2_mean | 0.049 | 0.049 | +0.000 |
| all|RXPhA|ENres | dmae_mean | 0.320 | 0.320 | +0.000 |
| all|RXPhA|ENres | false_null_dr2 | 0.060 | 0.060 | +0.000 |
| all|RXPhA|ENres | false_null_dmae | 0.642 | 0.674 | +0.032 |
| all|RXPhA|ENres | ciw_dr2 | 0.102 | 0.101 | -0.001 |
| all|RXPhA|ENres | cover_dmae_truthall | 0.732 | 0.736 | +0.004 |
| healthy|RX|LassoRaw | n | 104.416 | 104.416 | +0.000 |
| healthy|RX|LassoRaw | dr2_mean | 0.030 | 0.030 | +0.000 |
| healthy|RX|LassoRaw | dmae_mean | 0.205 | 0.205 | +0.000 |
| healthy|RX|LassoRaw | false_null_dr2 | 0.178 | 0.178 | +0.000 |
| healthy|RX|LassoRaw | false_null_dmae | 0.838 | 0.840 | +0.002 |
| healthy|RX|LassoRaw | ciw_dr2 | 0.111 | 0.112 | +0.001 |
| healthy|RX|LassoRaw | cover_dmae_truthall | 0.612 | 0.614 | +0.002 |
| healthy|RX|ENres | n | 104.416 | 104.416 | +0.000 |
| healthy|RX|ENres | dr2_mean | 0.036 | 0.036 | +0.000 |
| healthy|RX|ENres | dmae_mean | 0.244 | 0.244 | +0.000 |
| healthy|RX|ENres | false_null_dr2 | 0.098 | 0.098 | +0.000 |
| healthy|RX|ENres | false_null_dmae | 0.700 | 0.692 | -0.008 |
| healthy|RX|ENres | ciw_dr2 | 0.089 | 0.088 | -0.001 |
| healthy|RX|ENres | cover_dmae_truthall | 0.564 | 0.570 | +0.006 |
| healthy|RXPhA|LassoRaw | n | 104.416 | 104.416 | +0.000 |
| healthy|RXPhA|LassoRaw | dr2_mean | 0.021 | 0.021 | +0.000 |
| healthy|RXPhA|LassoRaw | dmae_mean | 0.153 | 0.153 | +0.000 |
| healthy|RXPhA|LassoRaw | false_null_dr2 | 0.304 | 0.304 | +0.000 |
| healthy|RXPhA|LassoRaw | false_null_dmae | 0.902 | 0.904 | +0.002 |
| healthy|RXPhA|LassoRaw | ciw_dr2 | 0.119 | 0.119 | -0.000 |
| healthy|RXPhA|LassoRaw | cover_dmae_truthall | 0.586 | 0.582 | -0.004 |
| healthy|RXPhA|ENres | n | 104.416 | 104.416 | +0.000 |
| healthy|RXPhA|ENres | dr2_mean | 0.032 | 0.032 | +0.000 |
| healthy|RXPhA|ENres | dmae_mean | 0.222 | 0.222 | +0.000 |
| healthy|RXPhA|ENres | false_null_dr2 | 0.124 | 0.124 | +0.000 |
| healthy|RXPhA|ENres | false_null_dmae | 0.730 | 0.720 | -0.010 |
| healthy|RXPhA|ENres | ciw_dr2 | 0.084 | 0.085 | +0.001 |
| healthy|RXPhA|ENres | cover_dmae_truthall | 0.514 | 0.518 | +0.004 |
