"""d9 배선 검산: 2024 폴드 가짜 test 에서 (d5-val) vs (d5-val + adj) 의 점수·τ 와, 행별 점수 이동이 Δt 공식과 일치하는지."""
import sys, os, json
sys.path.insert(0, "submit"); from script import load_tables, predict_all, ID, load_adj
import numpy as np, pandas as pd; from pathlib import Path
sys.path.insert(0, "../eda"); from _common import score, CLASSES
test = pd.read_csv("../out/fake_test_2024.csv", dtype={ID: str}); lab = pd.read_csv("../out/fake_labels_2024.csv", dtype={ID: str})
m = test[[ID, "pitcher_id"]].merge(lab, on=ID, how="left"); y = m.control_grade.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(); ok = ~np.isnan(y); pid = m.pitcher_id.to_numpy()
out = {}
for tag, d in (("d5val(base)", "submit_val_d9_base/model"), ("d9val(+adj)", "submit_val_d9/model")):
    T = load_tables(Path(d)); P = predict_all(test, Path(d), T); out[tag] = P; score(P[ok], pid[ok], y[ok], tag)
G = np.array([100., 30., 0.]); s5 = out["d5val(base)"] @ G; s9 = out["d9val(+adj)"] @ G
A = load_adj(Path("submit_val_d9/model")); isF = (test.game_type.astype(str) == "F").to_numpy(); loc = A["team_index"].get_indexer(pd.Index(test.pitcher_team_id.astype(str))); dev = np.where(loc >= 0, A["team_dev_arr"][np.where(loc >= 0, loc, 0)], 0.0)
dt = A["lvl_g"] * A["lvl_diff"] * isF + A["lvl_team_g"] * dev
print(f"F행 Δt = {A['lvl_g']*A['lvl_diff']:+.4f}점 · 팀항 범위 [{(A['lvl_team_g']*A['team_dev_arr']).min():+.3f}, {(A['lvl_team_g']*A['team_dev_arr']).max():+.3f}] · 팀 미매칭 행 {(loc<0).sum()}")
e = np.abs((s9 - s5) - dt); print(f"행별 (점수이동 − Δt): max {e.max():.3g} · 99.9% {np.quantile(e, .999):.3g} · >1e-6 인 행 {(e > 1e-6).sum()} (clip 영향)")
g = pd.DataFrame({"pid": pid, "d": s9 - s5, "dt": dt}).groupby("pid").mean(); print(f"투수별 평균 이동 vs 기대: max|Δ| {(g.d - g.dt).abs().max():.3g} · 이동 sd {g.d.std():.3f} · F행 비율 {isF.mean():.3f}")
