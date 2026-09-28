"""d20 하이브리드 검산 (2024 가짜 test): (1) 행별 제구점수 = L87 단독 예측의 점수와 동일한가 (2) Player 동일 · Pitch 는 dain 쪽인가."""
import sys, os, json, time, importlib.util, numpy as np, pandas as pd; from pathlib import Path
S = os.path.abspath("script_hybrid"); sys.path.insert(0, S); from script import load_tables, predict_all, ID
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eda")); from _common import score, CLASSES
test = pd.read_csv("../out/fake_test_2024.csv", dtype={ID: str}); lab = pd.read_csv("../out/fake_labels_2024.csv", dtype={ID: str})
m = test[[ID, "pitcher_id"]].merge(lab, on=ID, how="left"); y = m.control_grade.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(); ok = ~np.isnan(y); pid = m.pitcher_id.to_numpy()
mdir = Path("submit_d20/model"); T = load_tables(mdir); t0 = time.time(); P = predict_all(test, mdir, T); print(f"hybrid 예측 {time.time()-t0:.1f}s")
spec = importlib.util.spec_from_file_location("jm_script", str(mdir / "jm_root" / "script.py")); jm = importlib.util.module_from_spec(spec); spec.loader.exec_module(jm)
B = jm.make_bundle(mdir / "jm_root"); Pj = np.empty((len(test), 3))
for s0 in range(0, len(test), 200000): e0 = min(s0 + 200000, len(test)); Pj[s0:e0] = jm.predict(test.iloc[s0:e0].reset_index(drop=True), B)
G = np.array([100., 30., 0.]); print(f"행별 점수 |hybrid − L87| max {np.abs(P @ G - Pj @ G).max():.3g}")
score(Pj[ok], pid[ok], y[ok], "L87 단독 (그의 q + 그의 t)"); score(P[ok], pid[ok], y[ok], "d20 하이브리드 (dain q + 그의 t)")
d14 = np.load("/tmp/_d14_val_P.npy") if os.path.exists("/tmp/_d14_val_P.npy") else None
