"""25: 배포 스케일(EB 확률·수축 포함)에서 목표 재가중 — LOSO(시즌) 회귀: act_p ~ pS_p·100 + pH_p·30 + z_rev_p + 이력구간 더미.
평가: 남긴 시즌 τ (전체 행 채점 / R 행 채점). 기준 = 100pS+30pH."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from _common import load, PT, head, sub
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); from pipeline import eb_tables, eb_predict
d = load(labeled_only=True)
def tau(a, b): r = kendalltau(np.round(a, 10), np.round(b, 10), variant="b").statistic; return 0.0 if np.isnan(r) else r
rows = {}
for Y in (2021, 2022, 2023, 2024):
    lab = d[d.season < Y]; cur = d[d.season == Y]; E = eb_tables(lab, alpha=1300.0); P = eb_predict(cur, E)
    loc = E["pidx"].get_indexer(pd.Index(cur.pitcher_id.astype(str))); pn = np.where(loc >= 0, E["pn"][np.where(loc >= 0, loc, 0)], 0.0)
    rs = lab[lab.game_type == "R"]; gr = rs.groupby("pitcher_id").rev.agg(["sum", "size"]); rp = rs.rev.mean(); ebrev = (gr["sum"] + 1300 * rp) / (gr["size"] + 1300)
    z = cur.pitcher_id.map(ebrev).fillna(rp).to_numpy() - rp
    df = pd.DataFrame({"pid": cur.pitcher_id.to_numpy(), "S": 100 * P[:, 0], "H": 30 * P[:, 1], "z": 100 * z, "act": cur.pt.to_numpy(), "pn": pn, "R": (cur.game_type == "R").to_numpy()})
    for tag, dd in (("all", df), ("R", df[df.R])):
        g = dd.groupby("pid").agg(n=("act", "size"), S=("S", "mean"), H=("H", "mean"), z=("z", "mean"), act=("act", "mean"), pn=("pn", "first")); g = g[g.n >= 50]
        g["d0"] = (g.pn == 0).astype(float); g["d1"] = ((g.pn > 0) & (g.pn <= 300)).astype(float); g["d2"] = ((g.pn > 300) & (g.pn <= 1000)).astype(float); g["Y"] = Y
        rows.setdefault(tag, []).append(g)
SPECS = {"기준 100pS+30pH": None, "S,H 재가중": ["S", "H"], "S,H,rev": ["S", "H", "z"], "S,H,구간더미": ["S", "H", "d0", "d1", "d2"], "S,H,rev,구간더미": ["S", "H", "z", "d0", "d1", "d2"]}
for tag in ("all", "R"):
    A = pd.concat(rows[tag]); head(f"25. LOSO 재가중 — 채점 행: {tag} · 학습 = 다른 시즌 투수(≥50구) · 시즌 2022/2023/2024 τ")
    print(f"{'모형':<22}" + "".join(f"{Y:>10}" for Y in (2022, 2023, 2024)) + "   계수(2024 폴드 학습)")
    for nm, cols in SPECS.items():
        out = []; beta_last = None
        for Y in (2022, 2023, 2024):
            te = A[A.Y == Y]; tr = A[(A.Y != Y)]
            if cols is None: pred = te.S + te.H
            else:
                X = np.column_stack([tr[c] for c in cols] + [np.ones(len(tr))]); beta = np.linalg.lstsq(X, tr.act, rcond=None)[0]
                pred = np.column_stack([te[c] for c in cols] + [np.ones(len(te))]) @ beta; beta_last = beta
            out.append(tau(te.act, pred))
        print(f"{nm:<22}" + "".join(f"{t:>+10.4f}" for t in out) + ("" if beta_last is None else "   " + str(np.round(beta_last[:-1], 2))))
print("\n25 done")
