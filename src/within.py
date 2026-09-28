"""집단별 τ — EB / GBDT noF / noseason / v3 를 평가 투수의 F 비중 집단으로 나눠 본다 (2023·2024)."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
d = load()
def tau(a, b): return kendalltau(np.asarray(a).round(10), np.asarray(b).round(10), variant="b").statistic
for Y in (2023, 2024):
    lab, ev = fold(d, Y); E = eb_tables(lab, 2000); Pe = eb_predict(ev, E)
    g = pd.DataFrame({"pid": ev.pitcher_id.to_numpy(), "act": ev.pt.to_numpy(), "EB": Pe @ PT, "F": (ev.game_type == "F").to_numpy().astype(float)})
    for tag in ("noF", "noseason", "v3"):
        z = np.load(OUT + f"p_val_{tag}_{Y}.npz", allow_pickle=True); g[tag] = z["P"] @ PT
    a = g.groupby("pid").agg(n=("act", "size"), act=("act", "mean"), F=("F", "mean"), EB=("EB", "mean"), noF=("noF", "mean"), noseason=("noseason", "mean"), v3=("v3", "mean")); a = a[a.n >= 50]
    head(f"{Y}")
    for lo, hi, lab_ in ((0, 0.0001, "F=0% (R전용)"), (0, 0.1, "F<10%"), (0.1, 0.5, "F10–50%"), (0.5, 1.01, "F≥50%"), (0, 1.01, "전체")):
        m = (a.F >= lo) & (a.F < hi); print(f"  {lab_:14s} n={int(m.sum()):3d}  " + "  ".join(f"{c} {tau(a.act[m], a[c][m]):+.3f}" for c in ("EB", "noF", "noseason", "v3")))
    # R 전용 투수만 τ 를 재면 (test 가 R 위주라는 가설) — 실제 평균 pt 도 집단별로
    print("  실제 pt 평균: " + "  ".join(f"{lab_} {a.act[(a.F>=lo)&(a.F<hi)].mean():.2f}" for lo, hi, lab_ in ((0, 0.1, "F<10%"), (0.1, 0.5, "F10–50%"), (0.5, 1.01, "F≥50%"))))
