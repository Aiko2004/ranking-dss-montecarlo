"""all3_R30.csv = full_R30 + abl_R30 + r3_R30 (повторяющиеся контрольные методы совпадают и берутся из full).
python merge_all3.py ../results"""
import sys
import pandas as pd

R = sys.argv[1]
full = pd.read_csv(f"{R}/full_R30.csv.gz")
abl = pd.read_csv(f"{R}/abl_R30.csv.gz")
r3 = pd.read_csv(f"{R}/r3_R30.csv.gz")
abl = abl[~abl.method.isin(["WSM-entropy", "FIS-entropy", "Ridge"])]
r3 = r3[r3.method != "WSM-entropy"]
cols = full.columns
pd.concat([full, abl[[c for c in cols if c in abl]], r3[[c for c in cols if c in r3]]], ignore_index=True)[cols].to_csv(f"{R}/all3_R30.csv.gz", index=False)
