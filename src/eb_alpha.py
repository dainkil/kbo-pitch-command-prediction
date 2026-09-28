"""E4 — 라벨 비율 EB 의 잡음 감소: 클래스별 α (Shadow 는 투수간 참분산≈0 → 강하게 수축). 위치 정보 미사용. 3폴드 τ/Total(anchor v3)."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
d = load()
def eb_rows_alpha(ev, E, alphas):
    """L2 EB 이지만 클래스별 α: p_c = (n·rate_c + α_c·prior_c)/(n+α_c) 후 정규화."""
    pr = E["prior"][cells(ev)]; loc = E["pidx"].get_indexer(pd.Index(ev.pitcher_id.to_numpy())); have = loc >= 0; safe = np.where(have, loc, 0)
    n = np.where(have, E["pn"][safe], 0.0)[:, None]; rate = np.where(have[:, None], E["prate"][safe], 0.0); a = np.asarray(alphas, float)[None, :]
    p = (rate * n + a * pr) / (n + a); p = np.clip(p, 1e-6, None); return p / p.sum(1, keepdims=True)
rows = []
for Y in (2022, 2023, 2024):
    lab, ev = fold(d, Y); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy(); E = eb_tables(lab, 2000)
    z = np.load(OUT + f"p_val_v3_{Y}.npz", allow_pickle=True); assert (z["pid"] == pid).all(); Pg = z["P"]
    head(f"{Y}")
    for nm, al in (("기준 2000/2000/2000", (2000, 2000, 2000)), ("S∞ 1e6/2000/2000", (1e6, 2000, 2000)), ("S 8000/2000/2000", (8000, 2000, 2000)), ("S 4000/2000/2000", (4000, 2000, 2000)),
                   ("S∞ H1000 F2000", (1e6, 1000, 2000)), ("S∞ H2000 F1000", (1e6, 2000, 1000)), ("S∞ 1000/1000", (1e6, 1000, 1000)), ("S∞ 4000/4000", (1e6, 4000, 4000)),
                   ("1000/1000/1000", (1000, 1000, 1000)), ("S 4000 H 1000 F 1000", (4000, 1000, 1000)), ("S 8000 H 1000 F 2000", (8000, 1000, 2000))):
        r = score(anchor(Pg, eb_rows_alpha(ev, E, al), 1.0), pid, y, nm); rows.append(dict(Y=Y, m=nm, tot=r[0], pitch=r[1], tau=r[2]))
R = pd.DataFrame(rows).set_index(["m", "Y"]).unstack("Y"); base = R.loc["기준 2000/2000/2000"]
R["gate"] = [("✓" if ((R.loc[m, "tau"] > base["tau"]).all() and (R.loc[m, "tot"] > base["tot"]).all()) else "✗") for m in R.index]
for k in ("tot", "tau"): R[(k, "mean")] = R[k].mean(1)
head("3폴드 요약"); print(R[[("tau", 2022), ("tau", 2023), ("tau", 2024), ("tau", "mean"), ("tot", "mean"), ("gate", "")]].round(4).sort_values(("tau", "mean"), ascending=False).to_string())
