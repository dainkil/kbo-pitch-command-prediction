# -*- coding: utf-8 -*-
"""종민 제안(05:00) 오프라인판: 투수 × 카운트12 × 같은손여부 조건부 편차 표.
Δ[p,c,h] = n/(n+λ) · [ (s_pch − s_p) − (L_ch − L) ]   (투수 자신의 평균 대비 셀 편차에서 리그 셀 효과를 뺀 것 · λ=200 부분풀링)
행은 자기 (pitcher_id, balls, strikes, 투수손==타자손) 로 조회 → t += s·Δ. 투수 평균 = 2025 카운트·매치업 구성으로 가중된 조건부 실력.
사용: build_pch.py --out pch.json [--upto 2025] [--lam 200] [--min_n 1]"""
import numpy as np, pandas as pd, sys, json, argparse
sys.path.insert(0, "/home/ubuntu/workspace/산출물/dain/eda"); from _common import load, PT
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--upto", type=int, default=2025); ap.add_argument("--lam", type=float, default=200.0); ap.add_argument("--since", type=int, default=2019); A = ap.parse_args()
d = load(labeled_only=False); L = d[d.y.notna() & (d.season < A.upto) & (d.season >= A.since)].copy(); L["sc"] = PT[L.y.to_numpy().astype(int)]
L["c"] = (L.balls_before.astype(int) * 3 + L.strikes_before.astype(int)); L["h"] = (L.pitcher_hand.astype(int) == L.batter_hand.astype(int)).astype(int); L["ch"] = L.c * 2 + L.h
Lmean = float(L.sc.mean()); Lch = L.groupby("ch").sc.mean()
g = L.groupby(["pitcher_id", "ch"]).sc.agg(["size", "mean"]); sp = L.groupby("pitcher_id").sc.mean()
g["dev"] = (g["mean"] - g.index.get_level_values(0).map(sp).to_numpy()) - (g.index.get_level_values(1).map(Lch).to_numpy() - Lmean)
g["delta"] = g["size"] / (g["size"] + A.lam) * g["dev"]
keys = [f"{p}|{int(ch)}" for p, ch in g.index]; vals = [float(v) for v in g["delta"].to_numpy()]
out = {"lam": A.lam, "upto": A.upto, "since": A.since, "keys": keys, "vals": vals, "n_pitcher": int(g.index.get_level_values(0).nunique()), "note": "ch = (balls*3+strikes)*2 + [투수손==타자손] · Δ = n/(n+λ)·[(s_pch−s_p)−(L_ch−L)]"}
json.dump(out, open(A.out, "w")); print(f"saved {A.out} · 투수 {out['n_pitcher']} · 셀 {len(keys)} · Δ sd {np.std(vals):.3f} 범위 [{min(vals):+.2f},{max(vals):+.2f}] · n 중앙값 {g['size'].median():.0f} · 리그 셀 효과 범위 [{(Lch-Lmean).min():+.2f},{(Lch-Lmean).max():+.2f}]")
