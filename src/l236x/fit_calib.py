# -*- coding: utf-8 -*-
"""후보정 적합 (표본외): hold24(≤2023 학습) 의 2024 q 에 대해 anchor(proj) 후 Brier 를 최소화하는 온도 T·클래스 편향 b — Player 는 anchor 로 불변."""
import json, importlib.util, numpy as np, pandas as pd
from scipy.optimize import minimize
from common import *
HO = X + "/hold24"; sp = importlib.util.spec_from_file_location("o57s", HO + "/script.py"); O = importlib.util.module_from_spec(sp); sp.loader.exec_module(O)
B = O.make_bundle(HO); d = load_d(); dl = d[d.lab]; L24 = dl[dl.season == 2024]
X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); Y = np.eye(3)[y24]; dd = O.prep(X24)
pri = B["tables"]["prior"][O.cells(dd, False)]; Xf = O.feats(dd, B["tables"], pri, B["cols"], a_p=B.get("a_p"), beta=B.get("beta"))
q = np.mean([np.asarray(mm.predict(Xf.to_numpy(np.float64))).reshape(len(dd), 3) for mm in B["mains"]], axis=0)
EB = Xf[[f"pit_c{j}" for j in range(3)]].to_numpy(float); EB = np.clip(EB, 1e-9, None); EB /= EB.sum(1, keepdims=True); t = O.target_grade(dd, B["tables"], pri, B, EB)
def cal(q, T, b):
    z = np.log(np.clip(q, 1e-12, None)) / T + np.array([b[0], b[1], 0.0]); z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)
brier = lambda p: 100 * (1 - ((p - Y) ** 2).sum(1).mean() / 2)
base = brier(O.anchor(q, t, 1.0, mode="proj")); P(f"hold24 2024 anchored Pitch 기준 {base:.4f} · q 단독 {brier(q):.4f}")
for T in (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2): P(f"  T={T:<4} anchored Pitch {brier(O.anchor(cal(q, T, [0, 0]), t, 1.0, mode='proj')) - base:+.4f} · q 단독 {brier(cal(q, T, [0, 0])) - brier(q):+.4f}")
res = minimize(lambda w: -brier(O.anchor(cal(q, w[0], w[1:]), t, 1.0, mode="proj")), x0=[1.0, 0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6))
T_, bS, bH = res.x; P(f"  최적(anchored): T={T_:.3f} bS={bS:+.4f} bH={bH:+.4f} · ΔPitch {-res.fun - base:+.4f}")
res2 = minimize(lambda w: -brier(cal(q, w[0], w[1:])), x0=[1.0, 0.0, 0.0], method="Nelder-Mead"); P(f"  최적(q 단독): T={res2.x[0]:.3f} b={np.round(res2.x[1:], 4)} · Δ {-res2.fun - brier(q):+.4f}")
json.dump({"T": float(T_), "bS": float(bS), "bH": float(bH)}, open(X + "/calib.json", "w")); P("saved calib.json")
