import numpy as np, pandas as pd
from common import *
d = load_d(); dl = d[d.lab].copy(); A = np.load(JM + "/submit_TS_P600G10/model/tables.npz", allow_pickle=False)
R = pd.Series(A["r12_val"], index=A["r12_idx"]); P(f"표 {len(R)} · sd {R.std():.4f} · 범위 {R.min():+.3f}~{R.max():+.3f}")
dl["sh"] = (dl.pitcher_hand == dl.batter_hand).astype(int); dl["ptc"] = dl.pt - dl.groupby("season").pt.transform("mean") + dl.pt.mean()
def recon(v, lam, rows):
    x = rows; key = x.pitcher_id.astype(str) + "|" + x.cnt.astype(int).astype(str) + "|" + x.sh.astype(str)
    g = x[v].groupby(key).agg(["mean", "size"]); pm = x.groupby("pitcher_id")[v].mean(); pid = pd.Series(g.index.str.split("|").str[0], index=g.index)
    lc = x[v].groupby(x.cnt.astype(int).astype(str) + "|" + x.sh.astype(str)).mean(); ck = pd.Series(g.index.str.split("|").str[1] + "|" + g.index.str.split("|").str[2], index=g.index)
    Z = g["size"] / (g["size"] + lam) * ((g["mean"] - pid.map(pm)) - (ck.map(lc) - x[v].mean())); return Z
for v, rows, tag in (("pt", dl, "pt·전행"), ("ptc", dl, "시즌중심·전행"), ("pt", dl[dl.season >= 2023], "pt·2023+")):
    Z = recon(v, 600, rows); j = R.to_frame("t").join(Z.rename("z"), how="inner"); P(f"{tag}: 교집합 {len(j)} · 최대차 {(j.t-j.z).abs().max():.2e} · 상관 {np.corrcoef(j.t, j.z)[0,1]:.5f}")
