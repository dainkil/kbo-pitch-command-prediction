# -*- coding: utf-8 -*-
"""추론 — Player 목표값(t)은 동봉한 종민 TS 번들(model/jm_root)로 **그대로**, 등급 배분 q 만 [종민 5시드 ⊕ dain 5시드] 평균 + 온도 T.
anchor(proj) 가 행 기대점수를 t 로 정확히 고정하므로 투수 평균 = TS 와 비트 동일 → Player 불변, Pitch 만 변한다. 모든 단계 행 독립."""
import json, os, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
import dain_feats as DF                      # dain 표 → 62피처 (행 독립 join)
ID = "row_id"; PROB_COLS = ["prob_shadow", "prob_heart", "prob_failure"]

def _load_jm(root):
    sp = importlib.util.spec_from_file_location("jm_script", str(root / "script.py")); jm = importlib.util.module_from_spec(sp); sp.loader.exec_module(jm); return jm

def predict_all(test, mdir):
    cfg = json.load(open(mdir / "hyb.json", encoding="utf-8")); w, T = float(cfg["w_dain"]), float(cfg["q_T"])
    jm = _load_jm(mdir / "jm_root"); B = jm.make_bundle(mdir / "jm_root")
    TD = DF.load_tables(mdir); meta_d = TD["meta"]
    out = np.empty((len(test), 3), dtype=float)
    for s0 in range(0, len(test), 200000):                    # 메모리 단위 배치 · 행별 독립
        e0 = min(s0 + 200000, len(test)); te = test.iloc[s0:e0].reset_index(drop=True)
        d = jm.prep(te); pri = B["tables"]["prior"][jm.cells(d, B.get("use_isf", False))]
        X = jm.feats(d, B["tables"], pri, B["cols"], a_p=B.get("a_p"), beta=B.get("beta")); Xn = X.to_numpy(dtype=np.float64)
        qj = np.zeros((len(d), 3))
        for m_ in B["mains"]: qj += np.asarray(m_.predict(Xn)).reshape(len(d), 3)
        qj /= float(len(B["mains"]))
        EB = X[[f"pit_c{j}" for j in range(3)]].to_numpy(dtype=float); EB = np.clip(EB, 1e-9, None); EB /= EB.sum(axis=1, keepdims=True)
        t = jm.target_grade(d, B["tables"], pri, B, EB)          # ★ TS 의 Player 목표값 그대로
        qd = DF.predict_ensemble(mdir, DF.apply_tables(te, TD), meta_d) if w > 0 else qj
        q = w * qd + (1.0 - w) * qj
        if T != 1.0: q = np.clip(q, 1e-12, None) ** (1.0 / T); q = q / q.sum(axis=1, keepdims=True)
        out[s0:e0] = jm.anchor(q, t, 1.0, mode="proj")
    return out

def main():
    t0 = time.time(); root = Path("./"); mdir = root / "model"
    test = pd.read_csv(root / "data" / "test.csv", dtype={ID: str}); test.columns = [c.lstrip("﻿") for c in test.columns]
    sample = pd.read_csv(root / "data" / "sample_submission.csv", dtype={ID: str}); sample.columns = [c.lstrip("﻿") for c in sample.columns]
    if not test[ID].reset_index(drop=True).equals(sample[ID].reset_index(drop=True)): raise ValueError("row_id 순서 불일치")
    P = predict_all(test.reset_index(drop=True), mdir)
    if not np.isfinite(P).all(): raise ValueError("비정상 확률")
    if ((P < 0) | (P > 1)).any(): raise ValueError("범위")
    if not np.allclose(P.sum(1), 1.0, atol=1e-5): raise ValueError("확률 합 != 1")
    sub = sample.copy()
    for i, c in enumerate(PROB_COLS): sub[c] = P[:, i]
    (root / "output").mkdir(exist_ok=True); sub.to_csv(root / "output" / "submission.csv", index=False)
    print(f"saved output/submission.csv ({len(sub):,} rows) in {time.time()-t0:.1f}s · mean probs {P.mean(0).round(4)}")

if __name__ == "__main__": main()
