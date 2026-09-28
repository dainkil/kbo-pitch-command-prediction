"""d11 배선 검산: val_d9(+adj L25) vs val_d11(+adj L25 + lvln +0.3) — 행별 Δ = 0.3·lvl_diff·isF·z_p 재현, 2024 폴드 점수(참고: 레짐축이라 거울상 예상)."""
import sys, os, json, shutil
sys.path.insert(0, "submit"); from script import load_tables, predict_all, ID, load_adj
import numpy as np, pandas as pd; from pathlib import Path
sys.path.insert(0, "../eda"); from _common import score, CLASSES
shutil.rmtree("submit_val_d11", ignore_errors=True); shutil.copytree("submit_val_d9", "submit_val_d11"); shutil.copy("submit_d11/model/adj.json", "submit_val_d11/model/adj.json")
test = pd.read_csv("../out/fake_test_2024.csv", dtype={ID: str}); lab = pd.read_csv("../out/fake_labels_2024.csv", dtype={ID: str})
m = test[[ID, "pitcher_id"]].merge(lab, on=ID, how="left"); y = m.control_grade.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(); ok = ~np.isnan(y); pid = m.pitcher_id.to_numpy()
out = {}
for tag, d in (("d9val", "submit_val_d9/model"), ("d11val", "submit_val_d11/model")):
    T = load_tables(Path(d)); P = predict_all(test, Path(d), T); out[tag] = P; score(P[ok], pid[ok], y[ok], tag)
G = np.array([100., 30., 0.]); s9 = out["d9val"] @ G; s11 = out["d11val"] @ G
A = load_adj(Path("submit_val_d11/model")); isF = (test.game_type.astype(str) == "F").to_numpy(float)
loc = A["hist_index"].get_indexer(pd.Index(test.pitcher_id.astype(str))); z = np.where(loc >= 0, A["hist_z_arr"][np.where(loc >= 0, loc, 0)], A["hist_z_none"])
dt = A["lvln_g"] * A["lvl_diff"] * isF * z; e = np.abs((s11 - s9) - dt)
print(f"lvln 항: F행 {int(isF.sum())} · z 범위 [{z.min():.2f},{z.max():.2f}] · 미매칭(신인) 행 {(loc<0).sum()} · Δt 범위 [{dt.min():+.2f},{dt.max():+.2f}]")
print(f"행별 (점수이동 − Δt): max {e.max():.3g} · >1e-6 행 {(e>1e-6).sum()}")
