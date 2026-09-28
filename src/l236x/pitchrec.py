# -*- coding: utf-8 -*-
"""Pitch 회수 실험 (as-of: 이력≤2023 홀드아웃 번들 · 2024 라벨행). O57 구조(cmp −0.15) 위에서
   ① resid κ: t += κ·[(s−cmp) − m_p]   (q 의 투수×상황 상호작용만 · 투수 2025 평균 ≈ 0 → Player 중립 설계)
   ② drift κ: t += κ·(cmp − cmpbar_p)  (2025 구성 − 이력 구성)
   ③ full  κ: t += κ·(s − sbar_p)      (d22 κ anchor)
   ④ ncut"""
import os, sys, json, importlib.util, numpy as np, pandas as pd
from common import *
HO = X + "/hold24"
spec = importlib.util.spec_from_file_location("o57s", HO + "/script.py"); O = importlib.util.module_from_spec(spec); spec.loader.exec_module(O)
import shutil; shutil.copy(DN + "/tmp_O57/model/comp.txt", HO + "/model/comp.txt")
B = O.make_bundle(HO); m57 = json.load(open(DN + "/tmp_O57/model/meta.json")); B["cmp_g"] = m57["cmp_g"]; B["cmp_mu"] = m57["cmp_mu"]
P(f"홀드아웃 번들: a_t={B.get('a_t')} use_g={B.get('use_g')} cmp_g={B['cmp_g']} poff={'poff_val' in B['tables']}")
d = load_d(); dl = d[d.lab]; H = d[d.season <= 2023]; L24 = dl[dl.season == 2024]
X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); pid24 = L24.pitcher_id.to_numpy()
def parts(df):
    dd = O.prep(df); pri = B["tables"]["prior"][O.cells(dd, B.get("use_isf", False))]
    Xf = O.feats(dd, B["tables"], pri, B["cols"], a_p=B.get("a_p"), beta=B.get("beta")); Xn = Xf.to_numpy(np.float64)
    q = np.mean([np.asarray(mm.predict(Xn)).reshape(len(dd), 3) for mm in B["mains"]], axis=0)
    EB = Xf[[f"pit_c{j}" for j in range(3)]].to_numpy(float); EB = np.clip(EB, 1e-9, None); EB /= EB.sum(1, keepdims=True)
    t = O.target_grade(dd, B["tables"], pri, B, EB); cmp = O.B_CMP["m"].predict(O._cmp_feats(dd).to_numpy(np.float32)) @ G
    return q, t, q @ G, cmp
q, t, s, cmp = parts(X24)
# 이력(≤2023) 행에서 투수별 상수: m_p = mean(s−cmp) · cmpbar_p · sbar_p
Hs = H[B29].reset_index(drop=True); qh, th, sh, ch = parts(Hs)
hp = pd.DataFrame({"p": Hs.pitcher_id.to_numpy(), "r": sh - ch, "c": ch, "s": sh}).groupby("p").agg(r=("r", "mean"), c=("c", "mean"), s=("s", "mean"), n=("r", "size"))
gm_r, gm_c, gm_s = float((sh - ch).mean()), float(ch.mean()), float(sh.mean())
pid = pd.Series(pid24); has = pid.isin(hp.index).to_numpy()
m_p = pid.map(hp.r).fillna(gm_r).to_numpy(); cb_p = pid.map(hp.c).fillna(gm_c).to_numpy(); sb_p = pid.map(hp.s).fillna(gm_s).to_numpy()
P(f"이력 있는 2024 행 비율 {has.mean()*100:.1f}% · (s−cmp) 행 sd {np.std(s-cmp):.3f} · 상호작용 잔차 sd {np.std((s-cmp)-m_p):.3f} · drift 행 sd {np.std(cmp-cb_p):.3f}")
def ev(tag, tt): return score(y24, O.anchor(q, tt, 1.0, mode="proj"), pid24, tag)
b = ev("O57형 홀드아웃 base", t)
score(y24, q, pid24, "  (참고) q 비앵커")
res = []
for k in (0.25, 0.5, 1.0):
    r1 = ev(f"resid κ={k}", t + k * ((s - cmp) - m_p) * has); res.append(("resid", k, r1[1] - b[1], r1[2] - b[2]))
for k in (0.25, 0.5, 1.0):
    r2 = ev(f"drift κ={k} (이력 있는 투수만)", t + k * (cmp - cb_p) * has); res.append(("drift", k, r2[1] - b[1], r2[2] - b[2]))
for k in (0.25, 0.5):
    r3 = ev(f"full κ={k} (s − sbar_p)", t + k * (s - sb_p) * has); res.append(("full", k, r3[1] - b[1], r3[2] - b[2]))
for k in (0.5,):
    r4 = ev(f"resid κ={k} + drift 0.5", t + k * ((s - cmp) - m_p) * has + 0.5 * (cmp - cb_p) * has); res.append(("resid+drift", k, r4[1] - b[1], r4[2] - b[2]))
P("\n요약 ΔPitch / ΔPlayer (as-of 2024, base 대비):"); [P(f"   {a:12s} κ={k:<5} {dp:+.4f}  {dpl:+.4f}") for a, k, dp, dpl in res]
np.savez(X + "/pitchrec_2024.npz", s=s, cmp=cmp, t=t, m_p=m_p, cb_p=cb_p, has=has)
