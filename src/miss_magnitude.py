"""E1 — 빗나간 크기: 등급이 버린 연속 위치 정보가 투수 순위를 올리나. 3폴드 × {해간 지속성, 단독 τ, EB 주입 Δτ(원본/중립화), 집단별}."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
d_all = load(labeled_only=False); tm = load_tm(); d = d_all[d_all.y.notna()]
def tau(a, b): return kendalltau(np.asarray(a).round(10), np.asarray(b).round(10), variant="b").statistic
SIG = ["mr_far15", "mr_far30", "mr_far50", "mr_waste", "mr_chase", "mr_outpos", "mr_outsq", "mr_cdist", "raw_far30", "raw_waste", "raw_outpos", "sd_sx", "sd_sy"]
SIGN = {s: -1 for s in SIG}; SIGN["mr_chase"] = +1; SIGN["raw_chase"] = +1   # 크게 빠질수록 나쁨; chase(살짝 밖)는 waste 대비 좋음
F = {}
head("0. 해간 지속성 (투수 100투구+ 양 시즌, Pearson r) — 라벨비율 r≈.45 와 비교")
for Y in (2022, 2023, 2024):
    a = miss_table(d[d.season == Y - 1]); b = miss_table(d[d.season == Y])
    na = d[d.season == Y - 1].groupby("pitcher_id").size(); nb = d[d.season == Y].groupby("pitcher_id").size()
    ok = na[na >= 100].index.intersection(nb[nb >= 100].index).intersection(a.index).intersection(b.index)
    pa = d[d.season == Y - 1].groupby("pitcher_id").pt.mean(); pb = d[d.season == Y].groupby("pitcher_id").pt.mean()
    line = f"  {Y-1}→{Y} n={len(ok)}  pt {np.corrcoef(pa[ok], pb[ok])[0,1]:+.3f}  "
    line += "  ".join(f"{s.replace('mr_','')} {np.corrcoef(a.loc[ok, s], b.loc[ok, s])[0,1]:+.3f}" for s in ("raw_far30", "raw_waste", "raw_outpos", "sd_sx", "sd_sy", "mr_far30", "mr_outpos"))
    print(line)
for Y in (2022, 2023, 2024):
    lab, ev = fold(d, Y); T = build_tables(lab, d_all[d_all.season < Y], tm[tm.season < Y], Y); E = eb_tables(lab, 2000); sd_e = eb_level_sd(lab, E)
    Pe = eb_predict(ev, E)
    g = pd.DataFrame({"pid": ev.pitcher_id.to_numpy(), "act": ev.pt.to_numpy(), "eb": Pe @ PT, "F": (ev.game_type == "F").to_numpy().astype(float)}).groupby("pid").agg(n=("act", "size"), act=("act", "mean"), eb=("eb", "mean"), F=("F", "mean"))
    g = g[g.n >= 50]; F[Y] = (g, T["pitcher"], sd_e)
    print(f"[{Y}] 투수 {len(g)} · EB τ {tau(g.act, g.eb):+.4f} · sd_e {sd_e:.3f}", flush=True)
def evaluate(sigs, eps, neutral):
    out = []
    for Y in (2022, 2023, 2024):
        g, Tp, sd_e = F[Y]; z = inject_z(Tp, {s: SIGN[s] for s in sigs}, sd_e, neutral=neutral).reindex(g.index).fillna(0.0)
        s = g.eb + eps * sd_e * z; base = tau(g.act, g.eb); R = g.F < 0.1; Fh = g.F >= 0.5
        out.append((tau(g.act, s) - base, tau(g.act[R], s[R]) - tau(g.act[R], g.eb[R]), tau(g.act[Fh], s[Fh]) - tau(g.act[Fh], g.eb[Fh])))
    return np.array(out)   # (3폴드, [전체, within-R, F≥50%])
def show(label, A):
    ok_all = (A[:, 0] > 0).all(); ok_R = (A[:, 1] > 0).all()
    print(f"  {label:34s} Δτ전체 [{A[0,0]:+.4f} {A[1,0]:+.4f} {A[2,0]:+.4f}] mean {A[:,0].mean():+.4f} | within-R [{A[0,1]:+.4f} {A[1,1]:+.4f} {A[2,1]:+.4f}] | F≥50% [{A[0,2]:+.4f} {A[1,2]:+.4f} {A[2,2]:+.4f}]  {'✓' if ok_all else '·'}{'R' if ok_R else '·'}")
head("1. 신호별 단독 주입 ε=0.1 — 원본 vs 중립화  (✓=전체 3/3, R=within-R 3/3)")
for s in SIG:
    for neu in (False, True): show(f"{s} {'중립' if neu else '원본'}", evaluate([s], 0.1, neu))
head("2. 세트 × ε")
SETS = {"far30+waste+sy": ["mr_far30", "mr_waste", "sd_sy"], "far30": ["mr_far30"], "outpos+sy": ["mr_outpos", "sd_sy"], "far50": ["mr_far50"],
        "ext(far50 a2000)": ["mr_far50"], "all_mr": [s for s in SIG if s.startswith("mr_") and s != "mr_chase"], "sd만": ["sd_sx", "sd_sy"]}
for nm, sigs in SETS.items():
    for neu in (False, True):
        for eps in (0.05, 0.1, 0.2, 0.3): show(f"{nm} {'중립' if neu else '원본'} ε={eps}", evaluate(sigs, eps, neu))
