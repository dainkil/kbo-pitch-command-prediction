import os, numpy as np, pandas as pd
from common import *
d = load_d(); A = np.load(DN + "/tmp_O57/model/tables.npz", allow_pickle=False)
pg = pd.DataFrame({"sum": A["pgr_sum"], "n": A["pgr_n"]}, index=A["pgr_idx"]); pg["m"] = pg["sum"] / pg["n"]
for ctr in (0, 1):
    os.environ["PGR_CTR"] = str(ctr); import importlib; importlib.reload(T7)
    g, r2 = T7.proxy_pitcher(d); g["m"] = g["sum"] / g["size"]; j = pg.join(g, rsuffix="_r"); big = j[j.n > 500]
    P(f"PGR_CTR={ctr}: 대이력(n>500) {len(big)}명 · n 일치 {np.isclose(big.n, big['size']).mean()*100:.1f}% · 평균 최대차 {(big.m-big.m_r).abs().max():.2e}")
dl = d[d.lab]; gl = dl.groupby("pitcher_id").pt.agg(["sum", "size"]); j2 = pg.join(gl, rsuffix="_l")
P(f"라벨행만: pgr_n/라벨n 평균 {(j2.n/j2['size']).mean():.3f} · 라벨행 비중 분포 {(j2['size']/j2.n).describe().round(3).to_dict()}")
