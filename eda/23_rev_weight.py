"""23: 반대투구율(rev)을 별도 가중한 anchor 목표 — t' = t(EB α1300 셀prior) + κ·(EB_rev_p − rev_prior)·100.
rev 표는 R 행(체제 무관)으로, 수축 α_r. 3폴드 τ, 이력 구간별. 22번의 계수(−19/단위 rev) 근처 κ 스캔."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from _common import load, PT, head, sub
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); from pipeline import eb_tables, eb_predict
d = load(labeled_only=True)
def tau(a, b): r = kendalltau(np.round(a, 10), np.round(b, 10), variant="b").statistic; return 0.0 if np.isnan(r) else r
head("23. 반대투구율 별도 가중 — 3폴드 (기준: EB α1300 셀 prior, adj 없음)")
for src_tag, src_f in (("rev 표 = R 행", lambda x: x[x.game_type == "R"]), ("rev 표 = 전체 행", lambda x: x)):
    sub(src_tag)
    print(f"{'Y':<6}{'기준τ':>8}  " + "  ".join(f"κ={k:>4}" for k in (-5, -10, -15, -20, -30)) + "   | 소이력(≤300) 기준 → κ=−15 | 3001+ 기준 → κ=−15")
    for Y in (2022, 2023, 2024):
        lab = d[d.season < Y]; cur = d[d.season == Y]; E = eb_tables(lab, alpha=1300.0); t = eb_predict(cur, E) @ PT
        loc = E["pidx"].get_indexer(pd.Index(cur.pitcher_id.astype(str))); pn = np.where(loc >= 0, E["pn"][np.where(loc >= 0, loc, 0)], 0.0)
        rs = src_f(lab); gr = rs.groupby("pitcher_id").rev.agg(["sum", "size"]); rp = rs.rev.mean(); a_r = 1300.0
        ebrev = ((gr["sum"] + a_r * rp) / (gr["size"] + a_r)); z = cur.pitcher_id.map(ebrev).fillna(rp).to_numpy() - rp
        g = pd.DataFrame({"pid": cur.pitcher_id.to_numpy(), "t": t, "z": z, "act": cur.pt.to_numpy(), "pn": pn}).groupby("pid").agg(n=("act", "size"), t=("t", "mean"), z=("z", "mean"), act=("act", "mean"), pn=("pn", "first")); g = g[g.n >= 50]
        base = tau(g.act, g.t); row = f"{Y:<6}{base:>+8.4f}  " + "  ".join(f"{tau(g.act, g.t + k*100*g.z)-base:>+7.4f}" for k in (-5, -10, -15, -20, -30))
        sm = g.pn <= 300; lg = g.pn > 3000
        row += f"   | {tau(g.act[sm], g.t[sm]):+.3f} → {tau(g.act[sm], (g.t - 15*100*g.z)[sm]):+.3f} | {tau(g.act[lg], g.t[lg]):+.3f} → {tau(g.act[lg], (g.t - 15*100*g.z)[lg]):+.3f}"
        print(row, flush=True)
print("κ 단위: 점/rev비율 (22번 회귀 계수 −19 ≈ κ=−19 · z 는 EB 반대투구율 − 리그평균). 행 독립: 투수표 조회.")
