"""EB(L2형) × GBDT 확률 결합 — 3폴드 게이트. 게이트(사전): 채택 = EB 단독 대비 Total ↑ AND τ ↑ 가 2022·2023·2024 모두 성립."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
d = load(); rows = []
for Y in (2022, 2023, 2024):
    lab, ev = fold(d, Y); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy()
    head(f"평가 시즌 {Y}")
    for a in (1000, 2000, 4000):
        E = eb_tables(lab, alpha=a); Pe = eb_predict(ev, E); r = score(Pe, pid, y, f"EB 단독 α={a}"); rows.append(dict(Y=Y, m=f"EB_a{a}", tot=r[0], pitch=r[1], tau=r[2]))
    E = eb_tables(lab, alpha=2000); Pe = eb_predict(ev, E)
    for tag in ("noF", "noseason", "v3"):
        z = np.load(OUT + f"p_val_{tag}_{Y}.npz", allow_pickle=True); assert (z["pid"] == pid).all() and (z["y"] == y).all(), "row alignment"
        Pg = z["P"]; sub(f"GBDT {tag}")
        r = score(Pg, pid, y, f"  GBDT {tag} 단독"); rows.append(dict(Y=Y, m=f"{tag}_w0", tot=r[0], pitch=r[1], tau=r[2]))
        for w in (0.5, 0.6, 0.7, 0.8, 0.9):
            Pb = (1 - w) * Pg + w * Pe; Pb /= Pb.sum(1, keepdims=True)
            r = score(Pb, pid, y, f"  (1−w)·GBDT + w·EB  w={w}"); rows.append(dict(Y=Y, m=f"{tag}_w{w}", tot=r[0], pitch=r[1], tau=r[2]))
R = pd.DataFrame(rows).set_index(["m", "Y"]).unstack("Y")
head("3폴드 요약 — 게이트: EB_a2000 대비 Total·τ 모두 3/3 ↑")
base = R.loc["EB_a2000"]
R["gate"] = [("✓" if ((R.loc[m, "tot"] > base["tot"]).all() and (R.loc[m, "tau"] > base["tau"]).all()) else "✗") for m in R.index]
R[("tot", "mean")] = R["tot"].mean(1); R[("tau", "mean")] = R["tau"].mean(1); R[("pitch", "mean")] = R["pitch"].mean(1)
print(R.round(4).to_string())
