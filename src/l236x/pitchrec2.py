# -*- coding: utf-8 -*-
"""drift 항 정밀: cmpbar_p 정의(전이력 / 직전시즌 / 직전시즌≥50 else 전이력) × κ · 투수 단위 항 sd(Player 교란 크기) · as-of 2024"""
import os, sys, json, importlib.util, numpy as np, pandas as pd
from common import *
HO = X + "/hold24"
spec = importlib.util.spec_from_file_location("o57s", HO + "/script.py"); O = importlib.util.module_from_spec(spec); spec.loader.exec_module(O)
B = O.make_bundle(HO); m57 = json.load(open(DN + "/tmp_O57/model/meta.json")); B["cmp_g"] = m57["cmp_g"]; B["cmp_mu"] = m57["cmp_mu"]
d = load_d(); dl = d[d.lab]; H = d[d.season <= 2023]; L24 = dl[dl.season == 2024]
X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); pid24 = L24.pitcher_id.to_numpy()
def parts(df):
    dd = O.prep(df); pri = B["tables"]["prior"][O.cells(dd, B.get("use_isf", False))]
    Xf = O.feats(dd, B["tables"], pri, B["cols"], a_p=B.get("a_p"), beta=B.get("beta")); Xn = Xf.to_numpy(np.float64)
    q = np.mean([np.asarray(mm.predict(Xn)).reshape(len(dd), 3) for mm in B["mains"]], axis=0)
    EB = Xf[[f"pit_c{j}" for j in range(3)]].to_numpy(float); EB = np.clip(EB, 1e-9, None); EB /= EB.sum(1, keepdims=True)
    t = O.target_grade(dd, B["tables"], pri, B, EB); cmp = O.B_CMP["m"].predict(O._cmp_feats(dd).to_numpy(np.float32)) @ G
    return q, t, cmp
q, t, cmp = parts(X24)
Hs = H[B29].reset_index(drop=True); ch = O.B_CMP["m"].predict(O._cmp_feats(O.prep(Hs)).to_numpy(np.float32)) @ G
hp = pd.DataFrame({"p": Hs.pitcher_id.to_numpy(), "c": ch, "y": Hs.season.to_numpy()})
allb = hp.groupby("p").c.agg(["mean", "size"]); last = hp[hp.y == 2023].groupby("p").c.agg(["mean", "size"])
pid = pd.Series(pid24)
defs = {"all": allb["mean"], "last": last["mean"], "last50": last["mean"].where(last["size"] >= 50).dropna().combine_first(allb["mean"])}
act = pd.Series(pid24).value_counts(); act = act[act >= 50].index
def ev(tag, tt): return score(y24, O.anchor(q, tt, 1.0, mode="proj"), pid24, tag)
b = ev("base", t)
for nm, tab in defs.items():
    cb = pid.map(tab); has = cb.notna().to_numpy(); cb = cb.fillna(0.0).to_numpy()
    dr = (cmp - cb) * has; pm = pd.Series(dr).groupby(pid24).mean().reindex(act)
    P(f"[{nm}] 이력 커버 행 {has.mean()*100:.1f}% · 투수 단위 drift 평균항 sd(활동투수) {pm.std():.3f} · |평균| {pm.abs().mean():.3f}")
    for k in (0.5, 0.75, 1.0):
        r = ev(f"  drift[{nm}] κ={k}", t + k * dr); P(f"      Δ Pitch {r[1]-b[1]:+.4f} Player {r[2]-b[2]:+.4f} · 투수항 sd {k*pm.std():.3f}")
