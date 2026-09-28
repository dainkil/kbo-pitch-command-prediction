"""번들의 adj.json 에 erank 참조분포를 넣는다. 사용: build_erank.py <model_dir> [--upto 2025] [--w 1.0]
참조 = 그 번들 eb_tables 의 학습 투수: m_q = ((prate·n + α·gm)/(n+α))@G (전역 prior 사후평균), v_q = 1/(n/σ² + 1/τ²).
σ² = 라벨행 투구단위 점수 분산, τ² = 투수간 참 분산 (모멘트법: 투수-시즌 평균(n≥200) 분산 − 평균(σ²/n))."""
import sys, os, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eda")); from _common import load, PT
ap = argparse.ArgumentParser(); ap.add_argument("mdir"); ap.add_argument("--upto", type=int, default=2025); ap.add_argument("--w", type=float, default=1.0); A = ap.parse_args()
d = load(labeled_only=False); L = d[d.y.notna() & (d.season < A.upto)]
sc = PT[L.y.to_numpy().astype(int)]; s2 = float(sc.var())
ps = pd.DataFrame({"pid": L.pitcher_id.to_numpy(), "season": L.season.to_numpy(), "sc": sc}).groupby(["pid", "season"]).sc.agg(["mean", "size"]); ps = ps[ps["size"] >= 200]
t2 = float(ps["mean"].var() - (s2 / ps["size"]).mean()); t2 = max(t2, 0.5)
E = np.load(os.path.join(A.mdir, "eb_tables.npz"), allow_pickle=False); em = json.load(open(os.path.join(A.mdir, "eb_meta.json")))
gm = np.asarray(em["gm"], float); alpha = float(em["alpha"]); G = np.array([100.0, 30.0, 0.0])
pn = E["pn"].astype(float); prate = E["prate"].astype(float)
ref_m = ((prate * pn[:, None] + alpha * gm) / (pn[:, None] + alpha)) @ G; ref_v = 1.0 / (pn / s2 + 1.0 / t2)
adj = json.load(open(os.path.join(A.mdir, "adj.json"), encoding="utf-8"))
adj["erank"] = {"sigma2": s2, "tau2": t2, "w": A.w, "ref_m": [float(x) for x in ref_m], "ref_v": [float(x) for x in ref_v],
                "note": f"참조 학습 투수 {len(ref_m)} · σ²={s2:.1f} (sd {s2**.5:.2f}) · τ²={t2:.3f} (sd {t2**.5:.2f}) · α_theory=σ²/τ²={s2/t2:.0f} · upto {A.upto}"}
json.dump(adj, open(os.path.join(A.mdir, "adj.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(adj["erank"]["note"]); print(f"  ref_m: mean {ref_m.mean():.2f} sd {ref_m.std():.2f} 범위 [{ref_m.min():.1f},{ref_m.max():.1f}] · ref_v sd 범위 [{ref_v.min()**.5:.2f},{ref_v.max()**.5:.2f}] · 신인 v sd {t2**.5:.2f}")
