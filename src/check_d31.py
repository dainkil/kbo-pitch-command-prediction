"""d30/d31 검산 (2024 가짜 test): d30 행별 점수 == L236 단독 · d31−d30 == w·Σβ·x · Pitch/τ 변화 · 투수 평균 이동 sd"""
import sys, os, json, numpy as np, pandas as pd; from pathlib import Path
sys.path.insert(0, os.path.abspath("script_hybrid")); from script import load_tables, predict_all, ID
sys.path.insert(0, "../eda"); from _common import score, CLASSES
test = pd.read_csv("../out/fake_test_2024.csv", dtype={ID: str}); lab = pd.read_csv("../out/fake_labels_2024.csv", dtype={ID: str})
m = test[[ID, "pitcher_id"]].merge(lab, on=ID, how="left"); y = m.control_grade.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(); ok = ~np.isnan(y); pid = m.pitcher_id.to_numpy()
G = np.array([100., 30., 0.]); P = {}
for t in ("d30", "d31"):
    T = load_tables(Path(f"submit_{t}/model")); P[t] = predict_all(test, Path(f"submit_{t}/model"), T); score(P[t][ok], pid[ok], y[ok], f"{t}")
L236 = pd.read_csv("/tmp/ens/L236/output/submission.csv")[["prob_shadow", "prob_heart", "prob_failure"]].to_numpy()
print(f"d30 행별 점수 − L236 단독: max |Δ| {np.abs(P['d30'] @ G - L236 @ G).max():.3g}")
U = json.load(open("submit_d31/model/usage.json")); T31 = load_tables(Path("submit_d31/model"))
b = test.balls_before.to_numpy(); st = test.strikes_before.to_numpy(); fidx = pd.Index(U["fs_idx"]); floc = fidx.get_indexer(pd.Index(test.pitcher_id.astype(str))); fz = np.array(U["fs_z"])
cols = []
for k, mu, sd in zip(U["keys"], U["mu"], U["sd"]):
    if k.startswith("cnt_b"): cols.append((((b == int(k[5])) & (st == int(k[7]))).astype(float) - mu) / sd)
    else: cols.append(np.where(floc >= 0, fz[np.where(floc >= 0, floc, 0)], 0.0))
X = np.column_stack(cols); n = test[["pitcher_id"]].merge(T31["pitcher"][["p_n"]], left_on="pitcher_id", right_index=True, how="left")["p_n"].fillna(0).to_numpy(float)
dt = (X @ np.array(U["beta"])) * (U["alpha"] / (n + U["alpha"])); d = P["d31"] @ G - P["d30"] @ G
print(f"d31−d30 배선: max |Δ − dt| {np.abs(d - dt).max():.3g} · dt sd {dt.std():.3f} 범위 [{dt.min():+.2f},{dt.max():+.2f}] · fs 미등록 행 {(floc < 0).sum()}")
g = pd.DataFrame({"pid": pid, "d": d}).groupby("pid").d.mean(); print(f"투수별 평균 이동 sd {g.std():.3f} 범위 [{g.min():+.2f},{g.max():+.2f}]")
