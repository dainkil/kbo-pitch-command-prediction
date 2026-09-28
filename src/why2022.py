"""2022 폴드에서 주입이 왜 손해인가 — 평가 투수를 그 시즌 F 비중으로 나눠 τ 변화를 본다 (진단만, 채택 판단에 쓰지 않음).
F 비중 낮은 투수 집단에서도 손해면 '이력 길이·잡음', F 비중 높은 집단에서만 손해면 'F 라벨 체제'."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
d_all = load(labeled_only=False); tm = load_tm(); d = d_all[d_all.y.notna()]
def tau(a, b): return kendalltau(np.asarray(a).round(10), np.asarray(b).round(10), variant="b").statistic
for Y in (2022, 2023, 2024):
    lab, ev = fold(d, Y); T = build_tables(lab, d_all[d_all.season < Y], tm[tm.season < Y], Y); E = eb_tables(lab, 2000); sd_e = eb_level_sd(lab, E)
    Pe = eb_predict(ev, E); g = pd.DataFrame({"pid": ev.pitcher_id.to_numpy(), "act": ev.pt.to_numpy(), "eb": Pe @ PT, "F": (ev.game_type == "F").to_numpy().astype(float)})
    g = g.groupby("pid").agg(n=("act", "size"), act=("act", "mean"), eb=("eb", "mean"), F=("F", "mean")); g = g[g.n >= 50]
    head(f"{Y}: 투수 {len(g)} · F 비중 분포 0%:{(g.F==0).sum()} <10%:{((g.F>0)&(g.F<.1)).sum()} 10–50%:{((g.F>=.1)&(g.F<.5)).sum()} ≥50%:{(g.F>=.5).sum()}")
    for nm, sigs in SIGSETS.items():
        z = inject_z(T["pitcher"], sigs, sd_e).reindex(g.index).fillna(0.0); s = g.eb + 0.1 * sd_e * z
        line = f"  {nm:5s} ε=0.1  전체 Δτ {tau(g.act, s) - tau(g.act, g.eb):+.4f}"
        for lo, hi, lab_ in ((0, 0.1, "F<10%"), (0.1, 0.5, "F10–50%"), (0.5, 1.01, "F≥50%")):
            m = (g.F >= lo) & (g.F < hi)
            if m.sum() >= 20: line += f"  | {lab_}(n={int(m.sum())}) EB {tau(g.act[m], g.eb[m]):+.3f} → 주입 {tau(g.act[m], s[m]):+.3f}"
        print(line)
