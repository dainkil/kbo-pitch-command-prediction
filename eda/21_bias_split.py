"""21: E3 후속 — 소이력 과대예측이 F 체제 탓인가. (a) 투수의 평가시즌 F 비중별로 편향 분해 (b) R 행만으로 채점했을 때의 편향·오라클 δ
(c) prior 수준: 풀링(2019~Y-1) vs 최근 2시즌 셀 prior 의 수준 차. 학습 없음."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from _common import load, PT, head, sub
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); from pipeline import eb_tables, cells, eb_predict
d = load(labeled_only=True)
def tau(a, b): r = kendalltau(np.round(a, 10), np.round(b, 10), variant="b").statistic; return 0.0 if np.isnan(r) else r
BK = [("이력 0", lambda n: n == 0), ("1~300", lambda n: (n > 0) & (n <= 300)), ("301~1000", lambda n: (n > 300) & (n <= 1000)), ("1001~3000", lambda n: (n > 1000) & (n <= 3000)), ("3001+", lambda n: n > 3000)]
for Y in (2023, 2024):
    lab = d[d.season < Y]; cur = d[d.season == Y]
    E = eb_tables(lab, alpha=1300.0); t = eb_predict(cur, E) @ PT
    loc = E["pidx"].get_indexer(pd.Index(cur.pitcher_id.astype(str))); pn = np.where(loc >= 0, E["pn"][np.where(loc >= 0, loc, 0)], 0.0)
    df = pd.DataFrame({"pid": cur.pitcher_id.to_numpy(), "t": t, "act": cur.pt.to_numpy(), "pn": pn, "isF": (cur.game_type == "F").to_numpy(float)})
    head(f"{Y} · (a) 이력구간 × 평가시즌 F비중 별 편향 (예측−실제) · 투수 ≥50구")
    g = df.groupby("pid").agg(n=("act", "size"), t=("t", "mean"), act=("act", "mean"), pn=("pn", "first"), fsh=("isF", "mean")); g = g[g.n >= 50]
    fb = pd.cut(g.fsh, [-0.01, 0.1, 0.5, 1.0], labels=["F≤10%", "10~50%", "F>50%"])
    tb = g.assign(bias=g.t - g.act, fb=fb).groupby([pd.cut(g.pn, [-1, 0, 300, 1000, 3000, 1e9], labels=[b[0] for b in BK]), "fb"], observed=True).agg(n=("bias", "size"), bias=("bias", "mean"), act=("act", "mean"), t=("t", "mean"))
    print(tb.round(2).to_string())
    # (b) R 행만 채점 (투수 R 행 ≥50)
    sub("(b) R 행만으로 채점 — F 체제와 무관한 편향인가")
    r = df[df.isF == 0]; gr = r.groupby("pid").agg(n=("act", "size"), t=("t", "mean"), act=("act", "mean"), pn=("pn", "first")); gr = gr[gr.n >= 50]; base = tau(gr.act, gr.t)
    print(f"   R 행만 투수 {len(gr)} · 기준 τ {base:+.4f} · 리그 R 실제 {gr.act.mean():.2f}")
    for nm, f in BK:
        m = f(gr.pn.to_numpy()); q = gr[m]
        if len(q) < 5: continue
        best = max(((tau(gr.act, gr.t + dl * m), dl) for dl in np.arange(-4, 4.01, 0.25)), key=lambda x: x[0])
        print(f"   {nm:<10}{len(q):>4}  예측 {q.t.mean():.2f} 실제 {q.act.mean():.2f} 편향 {(q.t-q.act).mean():+.2f}  구간τ {tau(q.act, q.t):+.3f}  오라클 δ={best[1]:+.2f} → {best[0]-base:+.4f}")
    r0 = (gr.pn == 0).to_numpy(); r1 = ((gr.pn > 0) & (gr.pn <= 300)).to_numpy()
    best2 = max(((tau(gr.act, gr.t + a * r0 + b * r1), a, b) for a in np.arange(-4, 2.01, 0.5) for b in np.arange(-4, 2.01, 0.5)), key=lambda x: x[0])
    print(f"   2D 오라클(R만): 신인 {best2[1]:+.1f} · 1~300 {best2[2]:+.1f} → {best2[0]-base:+.4f}")
    # (c) prior 수준
    sub("(c) 배치셀 prior 수준 — 풀링 vs 최근 2시즌 vs 최근 2시즌 R 만")
    for tag, sel in (("풀링 2019~Y-1", lab), ("최근 2시즌", lab[lab.season >= Y - 2]), ("최근 2시즌 R", lab[(lab.season >= Y - 2) & (lab.game_type == "R")]), ("R 전체", lab[lab.game_type == "R"])):
        Ei = eb_tables(sel, alpha=1300.0); pr = Ei["prior"][cells(cur)] @ PT
        print(f"   {tag:<14} 평가행 prior 평균 {pr.mean():.2f} · 셀 sd {np.std(Ei['prior'] @ PT):.2f}   (평가시즌 실제 평균 {cur.pt.mean():.2f} · R {cur[cur.game_type=='R'].pt.mean():.2f} · F {cur[cur.game_type=='F'].pt.mean():.2f})")
    # (d) 신인 prior 를 '첫 시즌 투수-시즌' 평균으로 — 첫시즌 투수의 실제 평균 vs 전체
    fs = lab.groupby("pitcher_id").season.min(); first = lab[lab.season == lab.pitcher_id.map(fs)]
    a1 = first.groupby(["pitcher_id", "season"]).pt.agg(["mean", "size"]); a1 = a1[a1["size"] >= 50]
    aa = lab.groupby(["pitcher_id", "season"]).pt.agg(["mean", "size"]); aa = aa[aa["size"] >= 50]
    print(f"   (d) 과거 첫시즌 투수-시즌 평균 {a1['mean'].mean():.2f} (n={len(a1)}) vs 전체 투수-시즌 {aa['mean'].mean():.2f} (n={len(aa)}) → 신인 δ 후보 {a1['mean'].mean()-aa['mean'].mean():+.2f}")
print("\n21 done", flush=True)
