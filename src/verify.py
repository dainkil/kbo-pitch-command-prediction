"""제출 전 검산 (공통규칙 §2). 사용: verify.py <model_dir> <test_csv> [labels_csv]
① 행 독립: 전체 / 무작위 부분집합 / 단일행 / 역순 예측이 비트 단위로 같은가
② 배포 동일: labels_csv 가 있으면 script.py 경로의 확률로 Score/Pitch/Player 를 채점 (train.py val 과 비교)"""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submit")); from script import load_tables, predict_all, ID
mdir, test_csv = sys.argv[1], sys.argv[2]; labels_csv = sys.argv[3] if len(sys.argv) > 3 else None
from pathlib import Path
T = load_tables(Path(mdir)); meta = T["meta"]
test = pd.read_csv(test_csv, dtype={ID: str}); print(f"test {test.shape} · 피처 {len(meta['features'])} · 모델 {len(meta['models'])} · mode {meta.get('mode','gbdt')} · inject {meta.get('inject_eps',0)}", flush=True)
def pred(df): return predict_all(df.reset_index(drop=True), Path(mdir), T)
t0 = time.time(); P = pred(test); print(f"전체 예측 {time.time()-t0:.1f}s · 평균 {P.mean(0).round(4)}", flush=True)
rng = np.random.default_rng(0); idx = np.sort(rng.choice(len(test), min(5000, len(test)), replace=False))
d1 = np.abs(pred(test.iloc[idx]) - P[idx]).max()
d2 = np.abs(pred(test.iloc[::-1]) - P[::-1]).max()
single = [int(i) for i in rng.choice(len(test), 20, replace=False)]
d3 = max(np.abs(pred(test.iloc[[i]]) - P[i]).max() for i in single)
print(f"① 행 독립  부분집합 최대차 {d1:.3g} · 역순 {d2:.3g} · 단일행 {d3:.3g}  → {'PASS (비트 동일)' if d1 == 0 and d2 == 0 and d3 == 0 else 'FAIL'}", flush=True)
ok = np.isfinite(P).all() and (P >= 0).all() and (P <= 1).all() and np.allclose(P.sum(1), 1, atol=1e-5)
print(f"④ 형식     유한·[0,1]·합=1 → {'PASS' if ok else 'FAIL'}", flush=True)
if labels_csv:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eda")); from _common import score, CLASSES
    lab = pd.read_csv(labels_csv, dtype={ID: str}); m = test[[ID, "pitcher_id"]].merge(lab, on=ID, how="left")
    y = m.control_grade.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(); ok_ = ~np.isnan(y)
    score(P[ok_], m.pitcher_id.to_numpy()[ok_], y[ok_], "② 배포 동일 (script.py 경로 채점)")
    # d10: 이력 구간별 투수 τ (방향 참고용 — Player 폴드는 LB 부호를 못 맞힌다)
    from scipy.stats import kendalltau; PT = np.array([100., 30., 0.])
    g = pd.DataFrame({"pid": m.pitcher_id.to_numpy()[ok_], "pred": P[ok_] @ PT, "act": PT[y[ok_].astype(int)]}).groupby("pid").agg(n=("act", "size"), pred=("pred", "mean"), act=("act", "mean"))
    g["pn"] = T["pitcher"]["p_n"].reindex(g.index).fillna(0.0).to_numpy(); g = g[g.n >= 50]
    for nm, mk in (("전체", g.pn >= 0), ("이력 0", g.pn == 0), ("1~300", (g.pn > 0) & (g.pn <= 300)), ("≤300", g.pn <= 300), (">300", g.pn > 300)):
        q = g[mk]; t = kendalltau(q.act.round(10), q.pred.round(10), variant="b").statistic if len(q) >= 5 else float("nan")
        print(f"    τ {nm:<8} {len(q):>4}명  {t:+.4f}", flush=True)
