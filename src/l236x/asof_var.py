# -*- coding: utf-8 -*-
"""as-of(이력≤2023 → 2024) 점검: proxy w · 타자 편차 g · drift κ 조합 — 홀드아웃 번들(O57형, poff 없음) 위에서"""
import os, json, importlib.util, numpy as np, pandas as pd
from common import *
HO = X + "/hold24"; sp = importlib.util.spec_from_file_location("o57s", HO + "/script.py"); O = importlib.util.module_from_spec(sp); sp.loader.exec_module(O)
B = O.make_bundle(HO); m57 = json.load(open(DN + "/tmp_O57/model/meta.json")); B["cmp_g"] = m57["cmp_g"]; B["cmp_mu"] = m57["cmp_mu"]
d = load_d(); dl = d[d.lab]; H = d[d.season <= 2023]; HL = dl[dl.season <= 2023]; L24 = dl[dl.season == 2024]
X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); pid24 = L24.pitcher_id.to_numpy(); dd = O.prep(X24)
pri = B["tables"]["prior"][O.cells(dd, False)]; Xf = O.feats(dd, B["tables"], pri, B["cols"], a_p=B.get("a_p"), beta=B.get("beta"))
q = np.mean([np.asarray(mm.predict(Xf.to_numpy(np.float64))).reshape(len(dd), 3) for mm in B["mains"]], axis=0)
EB = Xf[[f"pit_c{j}" for j in range(3)]].to_numpy(float); EB = np.clip(EB, 1e-9, None); EB /= EB.sum(1, keepdims=True)
cmp24 = O.B_CMP["m"].predict(O._cmp_feats(dd).to_numpy(np.float32)) @ G
def pgr_table(W):
    L = HL; mu = L.pt.mean(); NK = int(d.pkey.max()) + 1
    c = np.bincount(L.pkey.to_numpy(), minlength=NK).astype(float); s = np.bincount(L.pkey.to_numpy(), weights=L.pt.to_numpy(), minlength=NK); PX = (s + 30. * mu) / (c + 30.)
    fill = np.where(H.lab.to_numpy(), np.nan_to_num(H.pt.to_numpy()), PX[H.pkey.to_numpy()]); sea = H.season.to_numpy()
    fill = fill - pd.Series(fill).groupby(sea).transform("mean").to_numpy() + float(fill.mean()); w = np.where(H.lab.to_numpy(), 1.0, W)
    F = pd.DataFrame({"p": H.pitcher_id.to_numpy(), "v": fill * w, "w": w}).groupby("p").agg(sum=("v", "sum"), size=("w", "sum")); return F[F["size"] > 0]
T0 = dict(B["tables"])
def target(W=1.0, gb=0.0, kd=0.0):
    T = dict(T0)
    if W != 1.0:
        F = pgr_table(W); T["pgr_idx"] = pd.Index(F.index.astype(str)); T["pgr_sum"] = F["sum"].to_numpy(float); T["pgr_n"] = F["size"].to_numpy(float)
    t = O.target_grade(dd, T, pri, B, EB)
    if gb:
        ch = O.B_CMP["m"].predict(O._cmp_feats(O.prep(HL[B29].reset_index(drop=True))).to_numpy(np.float32)) @ G
        r = HL.pt.to_numpy() - ch; r = r - pd.Series(r).groupby(HL.season.to_numpy()).transform("mean").to_numpy()
        g = pd.DataFrame({"b": HL.batter_id.to_numpy(), "r": r}).groupby("b").r.agg(["sum", "size"]); bd = g["sum"] / (g["size"] + 400.0)
        t = t + gb * pd.Series(X24.batter_id.to_numpy()).map(bd).fillna(0.0).to_numpy()
    if kd:
        ch2 = O.B_CMP["m"].predict(O._cmp_feats(O.prep(H[B29].reset_index(drop=True))).to_numpy(np.float32)) @ G
        g2 = pd.DataFrame({"p": H.pitcher_id.to_numpy(), "c": ch2}).groupby("p").c.agg(["sum", "size"]); cb = (g2["sum"] + 200 * ch2.mean()) / (g2["size"] + 200)
        cbp = pd.Series(pid24).map(cb); t = t + kd * cbp.notna().to_numpy() * (cmp24 - cbp.fillna(0).to_numpy())
    return t
ev = lambda tag, t: score(y24, O.anchor(q, t, 1.0, mode="proj"), pid24, tag)
b = ev("base (proxy w=1)", target())
for W in (0.0, 0.5): r = ev(f"proxy w={W}", target(W=W)); P(f"      Δ Pitch {r[1]-b[1]:+.4f} Player {r[2]-b[2]:+.4f}")
for gb in (-0.25, -0.5, -1.0, +0.5): r = ev(f"bat g={gb}", target(gb=gb)); P(f"      Δ Pitch {r[1]-b[1]:+.4f} Player {r[2]-b[2]:+.4f}")
r = ev("drift κ=0.5 + bat −0.5", target(gb=-0.5, kd=0.5)); P(f"      Δ Pitch {r[1]-b[1]:+.4f} Player {r[2]-b[2]:+.4f}")
r = ev("drift κ=0.5 + bat −0.5 + proxy 0", target(W=0.0, gb=-0.5, kd=0.5)); P(f"      Δ Pitch {r[1]-b[1]:+.4f} Player {r[2]-b[2]:+.4f}")
