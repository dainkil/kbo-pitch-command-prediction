# -*- coding: utf-8 -*-
"""as-of 패널: 이력 <Y 로 신호를 만들고 Y 시즌 실제 평균(50구+)과의 τ_b — 인샘플 누수 없는 방향 점검 (팀 규율상 채택 근거는 아님)."""
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from common import *
import signals as SG
d = load_d(); meta = base_meta(); dl = d[d.lab]
TAU = lambda a, b: float(kendalltau(np.asarray(a, float), np.asarray(b, float), variant="b").statistic)
KS = ["DIR", "DSHs", "SH", "SIT", "PLT", "GEOF", "GEOF23", "DIRc", "DSHsc", "SITb", "CNTH", "CPs"]
rows = []
for Y in (2022, 2023, 2024):
    H = d[d.season < Y]; S = SG.build(H, meta)
    n, fsh, L164p, Z, cw = S["n"], S["fsh"], S["L164p"], S["Z"], S["cw"]
    E = dl[dl.season == Y].groupby("pitcher_id").agg(m=("pt", "mean"), c=("pt", "size")); E = E[E.c >= 50]
    act = E.index.intersection(n.index); y = E.m.reindex(act)
    A_ = np.c_[np.ones(len(n)), L164p.to_numpy(), np.log1p(n.to_numpy()), fsh.reindex(n.index).to_numpy()]; W = np.sqrt(n.to_numpy())
    orth = lambda z: z - A_ @ np.linalg.lstsq(A_ * W[:, None], z.to_numpy() * W, rcond=None)[0]
    b = TAU(L164p.reindex(act), y); r = {"Y": Y, "n_act": len(act), "base": round(b, 4)}
    for k in KS:
        z = orth(Z[k].reindex(n.index).fillna(0.0)); z = cw(z) / float(z.reindex(act).std())
        for kap in (0.12, 0.25):
            r[f"{k}@{kap}"] = round(TAU((L164p + kap * z).reindex(act), y) - b, 4)
    rows.append(r); P(r)
R = pd.DataFrame(rows).set_index("Y"); P("\nΔτ (as-of, 활동투수 전체):\n" + R.T.to_string())
