"""24: anchor 목표의 성분 재가중 — t' = ws·100·pS + wh·30·pH − κ·100·(EBrev − 리그) (EB α1300 셀 prior 확률에서). 22번 회귀 가중 vs 점수 정의 가중. 3폴드."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from _common import load, PT, head, sub
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); from pipeline import eb_tables, eb_predict
d = load(labeled_only=True)
def tau(a, b): r = kendalltau(np.round(a, 10), np.round(b, 10), variant="b").statistic; return 0.0 if np.isnan(r) else r
V = [("기준 (1,1,0)", 1, 1, 0), ("회귀 (.283,.587,19.3)", .283, .587, 19.3), ("회귀 κ=0 (.283,.587,0)", .283, .587, 0), ("(1,1.5,0)", 1, 1.5, 0), ("(1,2,0)", 1, 2, 0), ("(1,1,5)", 1, 1, 5), ("(.5,.5,10)", .5, .5, 10)]
head("24. 목표 성분 재가중 — 3폴드 τ (전체 / ≤300 / 3001+)")
print(f"{'변형':<26}" + "".join(f"{Y:>26}" for Y in (2022, 2023, 2024)))
res = {}
for Y in (2022, 2023, 2024):
    lab = d[d.season < Y]; cur = d[d.season == Y]; E = eb_tables(lab, alpha=1300.0); P = eb_predict(cur, E)
    loc = E["pidx"].get_indexer(pd.Index(cur.pitcher_id.astype(str))); pn = np.where(loc >= 0, E["pn"][np.where(loc >= 0, loc, 0)], 0.0)
    rs = lab[lab.game_type == "R"]; gr = rs.groupby("pitcher_id").rev.agg(["sum", "size"]); rp = rs.rev.mean(); ebrev = (gr["sum"] + 1300 * rp) / (gr["size"] + 1300)
    z = cur.pitcher_id.map(ebrev).fillna(rp).to_numpy() - rp
    base = pd.DataFrame({"pid": cur.pitcher_id.to_numpy(), "pS": P[:, 0], "pH": P[:, 1], "z": z, "act": cur.pt.to_numpy(), "pn": pn}).groupby("pid").agg(n=("act", "size"), pS=("pS", "mean"), pH=("pH", "mean"), z=("z", "mean"), act=("act", "mean"), pn=("pn", "first")); base = base[base.n >= 50]
    for nm, ws, wh, k in V:
        t = ws * 100 * base.pS + wh * 30 * base.pH - k * 100 * base.z
        res.setdefault(nm, []).append((tau(base.act, t), tau(base.act[base.pn <= 300], t[base.pn <= 300]), tau(base.act[base.pn > 3000], t[base.pn > 3000])))
for nm, *_ in V: print(f"{nm:<26}" + "".join(f"   {a:+.3f} / {b:+.3f} / {c:+.3f}" for a, b, c in res[nm]))
