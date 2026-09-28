"""verify.py 와 동일하되 script.py 위치를 인자로 받는다 (다른 작업에서 편집 중인 submit/script.py 대신 배포 zip 의 script 를 씀).
사용: verify_with.py <script_dir> <model_dir> <test_csv> [labels_csv]"""
import sys, os, time
import numpy as np, pandas as pd
from pathlib import Path
sdir, mdir, test_csv = sys.argv[1], sys.argv[2], sys.argv[3]; labels_csv = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, sdir); from script import load_tables, predict_all, ID
T = load_tables(Path(mdir)); meta = T["meta"]; has_adj = (Path(mdir) / "adj.json").exists()
test = pd.read_csv(test_csv, dtype={ID: str})
print(f"script {sdir} · model {mdir} · test {test.shape} · 피처 {len(meta['features'])} · 모델 {len(meta['models'])} · mode {meta.get('mode', 'gbdt')} · inject {meta.get('inject_eps', 0)} · pre23f {meta.get('pre23f', 'keep')} · adj {'Y' if has_adj else 'N'}", flush=True)
def pred(df): return predict_all(df.reset_index(drop=True), Path(mdir), T)
t0 = time.time(); P = pred(test); print(f"전체 예측 {time.time()-t0:.1f}s · 평균 {P.mean(0).round(4)}", flush=True)
rng = np.random.default_rng(0); idx = np.sort(rng.choice(len(test), min(5000, len(test)), replace=False))
d1 = np.abs(pred(test.iloc[idx]) - P[idx]).max(); d2 = np.abs(pred(test.iloc[::-1]) - P[::-1]).max()
single = [int(i) for i in rng.choice(len(test), 20, replace=False)]; d3 = max(np.abs(pred(test.iloc[[i]]) - P[i]).max() for i in single)
print(f"① 행 독립  부분집합 최대차 {d1:.3g} · 역순 {d2:.3g} · 단일행 {d3:.3g}  → {'PASS (비트 동일)' if d1 == 0 and d2 == 0 and d3 == 0 else 'FAIL'}", flush=True)
ok = np.isfinite(P).all() and (P >= 0).all() and (P <= 1).all() and np.allclose(P.sum(1), 1, atol=1e-5)
print(f"④ 형식     유한·[0,1]·합=1 → {'PASS' if ok else 'FAIL'}", flush=True)
if labels_csv:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eda")); from _common import score, CLASSES
    lab = pd.read_csv(labels_csv, dtype={ID: str}); m = test[[ID, "pitcher_id"]].merge(lab, on=ID, how="left")
    y = m.control_grade.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(); ok_ = ~np.isnan(y)
    score(P[ok_], m.pitcher_id.to_numpy()[ok_], y[ok_], "② 배포 동일 (script.py 경로 채점)")
