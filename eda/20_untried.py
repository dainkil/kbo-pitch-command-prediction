"""20: 4명이 안 본 각도 — (E3) 이력 구간별 '수준' 편향과 오라클 δ · (E1) 등급 성분별 해간 신뢰도(어느 성분이 다음 시즌 점수를 더 잘 예측하나)
· (E7) 신인의 월/레벨별 실제 점수 (row 열만으로 잡히는 신인 prior 후보). 학습 없음, 표만."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from _common import load, PT, head, sub
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); from pipeline import eb_tables, cells, eb_predict
d = load(labeled_only=True)
def tau(a, b): r = kendalltau(np.round(a, 10), np.round(b, 10), variant="b").statistic; return 0.0 if np.isnan(r) else r
BK = [("이력 0", lambda n: n == 0), ("1~300", lambda n: (n > 0) & (n <= 300)), ("301~1000", lambda n: (n > 300) & (n <= 1000)), ("1001~3000", lambda n: (n > 1000) & (n <= 3000)), ("3001+", lambda n: n > 3000)]

head("E3. 이력 구간별 수준 편향 — EB(배치셀 prior, α=1300) 예측 평균 vs 실제 평균, 오라클 δ")
for Y in (2022, 2023, 2024):
    lab = d[d.season < Y]; cur = d[d.season == Y]
    E = eb_tables(lab, alpha=1300.0); t = eb_predict(cur, E) @ PT
    loc = E["pidx"].get_indexer(pd.Index(cur.pitcher_id.astype(str))); pn = np.where(loc >= 0, E["pn"][np.where(loc >= 0, loc, 0)], 0.0)
    g = pd.DataFrame({"pid": cur.pitcher_id.to_numpy(), "t": t, "act": cur.pt.to_numpy(), "pn": pn, "isF": (cur.game_type == "F").to_numpy(float)}).groupby("pid").agg(n=("act", "size"), t=("t", "mean"), act=("act", "mean"), pn=("pn", "first"), fsh=("isF", "mean"))
    g = g[g.n >= 50]; base = tau(g.act, g.t)
    sub(f"{Y} · 투수 {len(g)} · 기준 τ {base:+.4f} · 리그 실제평균 {g.act.mean():.2f}")
    print(f"{'구간':<10}{'명':>5}{'예측평균':>9}{'실제평균':>9}{'편향(예−실)':>12}{'구간내τ':>9}   오라클 δ(구간만 이동) → τ")
    for nm, f in BK:
        m = f(g.pn.to_numpy()); q = g[m]
        if len(q) < 5: continue
        best = max(((tau(g.act, g.t + dl * m), dl) for dl in np.arange(-4, 4.01, 0.25)), key=lambda x: x[0])
        print(f"{nm:<10}{len(q):>5}{q.t.mean():>9.2f}{q.act.mean():>9.2f}{(q.t - q.act).mean():>+12.2f}{tau(q.act, q.t):>+9.3f}   δ={best[1]:+.2f} → {best[0]:+.4f} ({best[0]-base:+.4f})")
    # 신인 δ 와 1~300 δ 동시 (2D 격자)
    r0 = (g.pn == 0).to_numpy(); r1 = ((g.pn > 0) & (g.pn <= 300)).to_numpy()
    best2 = max(((tau(g.act, g.t + a * r0 + b * r1), a, b) for a in np.arange(-4, 2.01, 0.5) for b in np.arange(-4, 2.01, 0.5)), key=lambda x: x[0])
    print(f"   2D 오라클: 신인 δ0={best2[1]:+.1f} · 1~300 δ1={best2[2]:+.1f} → τ {best2[0]:+.4f} ({best2[0]-base:+.4f})")
    # 신인 안에서 F 비중과 실제
    q = g[r0]; print(f"   신인 {len(q)}명: corr(2025 F비중, 실제) {np.corrcoef(q.fsh, q.act)[0,1]:+.3f} · F비중>0.5 {int((q.fsh>0.5).sum())}명 실제 {q[q.fsh>0.5].act.mean():.2f} vs ≤0.5 실제 {q[q.fsh<=0.5].act.mean():.2f}")

head("E1. 등급 성분별 해간 신뢰도 — 투수-시즌 (연속 시즌 둘 다 ≥200구). 어느 성분이 '다음 시즌 점수'를 가장 잘 예측하나")
oh = np.eye(3)[d.y.to_numpy().astype(int)]
ps = pd.DataFrame({"pid": d.pitcher_id.to_numpy(), "s": d.season.to_numpy(), "sh": oh[:, 0], "he": oh[:, 1], "fa": oh[:, 2], "rev": d.rev.to_numpy(float), "pt": d.pt.to_numpy(), "R": (d.game_type == "R").to_numpy(float)})
ps["fa_nr"] = ps.fa * (1 - ps.rev)   # 반대투구 아닌 Failure (위치 이탈)
for tag, sel in (("전체 행", ps), ("R 행만", ps[ps.R == 1]), ("2023+ 만", ps[ps.s >= 2023])):
    a = sel.groupby(["pid", "s"]).agg(n=("pt", "size"), sh=("sh", "mean"), he=("he", "mean"), fa=("fa", "mean"), rev=("rev", "mean"), fa_nr=("fa_nr", "mean"), pt=("pt", "mean")).reset_index()
    a = a[a.n >= 200]; b = a.copy(); b["s"] = b.s - 1
    j = a.merge(b, on=["pid", "s"], suffixes=("", "_next"))
    sub(f"{tag}: 연속 쌍 {len(j)} · 시즌 {sorted(j.s.unique())}")
    print(f"{'성분':<8}{'자기상관(해간)':>14}{'→다음 점수 corr':>16}{'→다음 점수 τ':>14}{'투수간 sd':>10}")
    for c in ("sh", "he", "fa", "rev", "fa_nr", "pt"):
        print(f"{c:<8}{np.corrcoef(j[c], j[c+'_next'])[0,1]:>+14.3f}{np.corrcoef(j[c], j['pt_next'])[0,1]:>+16.3f}{tau(j['pt_next'], j[c]):>+14.3f}{a[c].std():>10.4f}")
    # 성분 조합: 다음 점수를 he·fa 로 회귀 (sh 제외) vs pt 단독
    X = np.column_stack([j.sh, j.he, np.ones(len(j))]); beta = np.linalg.lstsq(X, j.pt_next, rcond=None)[0]; pred = X @ beta
    print(f"   다음점수 ~ sh·he 회귀: 계수 sh {beta[0]:+.1f} he {beta[1]:+.1f} (점수 가중 100/30 과 비교) · τ {tau(j.pt_next, pred):+.3f} vs pt 단독 {tau(j.pt_next, j.pt):+.3f}")

head("E7. 신인(as-of 이력 0) 의 실제 점수 — 행 열(월·레벨·이닝·li)만으로 갈리는가 (2023~2024 시즌)")
for Y in (2023, 2024):
    lab = d[d.season < Y]; cur = d[d.season == Y]; seen = set(lab.pitcher_id.unique())
    rk = cur[~cur.pitcher_id.isin(seen)]; ok = rk.groupby("pitcher_id").pt.transform("size") >= 50; rk = rk[ok]
    sub(f"{Y} 신인 {rk.pitcher_id.nunique()}명 · {len(rk)}행 · 실제 평균 {rk.pt.mean():.2f} (리그 {cur.pt.mean():.2f})")
    for col, bins in (("game_month", None), ("game_type", None), ("inning", [0, 3, 6, 8, 20]), ("li", [0, 0.6, 1.4, 99]), ("balls_before", None)):
        k = rk[col] if bins is None else pd.cut(rk[col], bins, labels=False)
        tb = rk.groupby(k).pt.agg(["mean", "size"]); print(f"   {col:<12} " + " · ".join(f"{i}:{m:.1f}({n})" for i, (m, n) in tb.iterrows()))
    # 신인 투수 단위: 투수별 (월 평균, F 비중, 평균 이닝, 평균 li) vs 실제
    gp = rk.groupby("pitcher_id").agg(act=("pt", "mean"), mon=("game_month", "mean"), fsh=("game_type", lambda s: (s == "F").mean()), inn=("inning", "mean"), li=("li", "mean"), balls=("balls_before", "mean"), sd=("score_diff_pitcher_team", "mean"))
    print("   투수 단위 τ(실제, x): " + " · ".join(f"{c} {tau(gp.act, gp[c]):+.3f}" for c in ("mon", "fsh", "inn", "li", "balls", "sd")))
print("\n20 done", flush=True)
