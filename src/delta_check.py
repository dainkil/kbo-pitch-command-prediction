"""마진 투수 δ 가 배포 경로(base α1300+adj / Cf skill prior) 위에서도 남는가 — 2024 가짜 test, 이력 구간별 편향·오라클 δ (전체 / R 행만)."""
import sys, os, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import kendalltau
sys.path.insert(0, "submit"); from script import load_tables, predict_all, ID, load_eb
sys.path.insert(0, "../eda"); from _common import CLASSES, PT
def tau(a, b): r = kendalltau(np.round(a, 10), np.round(b, 10), variant="b").statistic; return 0.0 if np.isnan(r) else r
test = pd.read_csv("../out/fake_test_2024.csv", dtype={ID: str}); lab = pd.read_csv("../out/fake_labels_2024.csv", dtype={ID: str})
m = test[[ID, "pitcher_id", "game_type"]].merge(lab, on=ID, how="left"); y = m.control_grade.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(); ok = ~np.isnan(y)
for tag, mdir in (("base α1300+adj", "submit_val_d10base/model"), ("Cf skill prior", "submit_val_d10Cf/model"), ("Af", "submit_val_d10Af/model")):
    T = load_tables(Path(mdir)); P = predict_all(test, Path(mdir), T); E = load_eb(Path(mdir))
    loc = E["pidx"].get_indexer(pd.Index(test.pitcher_id.astype(str))); pn = np.where(loc >= 0, E["pn"][np.where(loc >= 0, loc, 0)], 0.0)
    df = pd.DataFrame({"pid": m.pitcher_id, "t": P @ PT, "act": PT[np.nan_to_num(y).astype(int)], "pn": pn, "R": (m.game_type == "R")})[ok]
    print(f"\n### {tag}")
    for sub_tag, dd in (("전체 행", df), ("R 행만", df[df.R])):
        g = dd.groupby("pid").agg(n=("act", "size"), t=("t", "mean"), act=("act", "mean"), pn=("pn", "first")); g = g[g.n >= 50]; base = tau(g.act, g.t)
        r1 = ((g.pn > 0) & (g.pn <= 300)).to_numpy(); r0 = (g.pn == 0).to_numpy()
        best = max(((tau(g.act, g.t + dl * r1), dl) for dl in np.arange(-4, 2.01, 0.25)), key=lambda x: x[0])
        best2 = max(((tau(g.act, g.t + a * r0 + b * r1), a, b) for a in np.arange(-3, 3.01, 0.5) for b in np.arange(-4, 2.01, 0.5)), key=lambda x: x[0])
        bias = {nm: (g.t - g.act)[mk].mean() for nm, mk in (("0", r0), ("1~300", r1), (">3000", (g.pn > 3000).to_numpy()))}
        print(f"  {sub_tag}: 투수 {len(g)} τ {base:+.4f} · 편향 신인 {bias['0']:+.2f} / 1~300 {bias['1~300']:+.2f} / 3001+ {bias['>3000']:+.2f} · 오라클 δ(1~300)={best[1]:+.2f} → {best[0]-base:+.4f} · 2D 신인 {best2[1]:+.1f}/1~300 {best2[2]:+.1f} → {best2[0]-base:+.4f}")
