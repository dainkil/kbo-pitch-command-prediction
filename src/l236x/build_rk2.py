# -*- coding: utf-8 -*-
"""리뷰 6항 반영: RK(pitcher-season lambdarank, 이전 시즌만 → 다음 시즌 순위) 를 **현재 목표값 성분**(pgr EB α2400·GEO·SIT·SH·PLT·DIR·DSHs·log n·F비중)에 다시 residualize 한 뒤 같은 sd 0.25 로 재투입.
   python3 build_rk2.py SRCDIR NAME"""
import os, sys, json, shutil, numpy as np, pandas as pd
from common import *
import signals as SG
src, name = sys.argv[1], sys.argv[2]; out = DN + f"/src/submit_{name}"
shutil.rmtree(out, ignore_errors=True); shutil.copytree(DN + f"/src/submit_{src}", out, ignore=shutil.ignore_patterns("__pycache__", "output", "data"))
d = load_d(); meta = json.load(open(out + "/model/meta.json", encoding="utf-8")); S = SG.build(d, meta)
n, fsh, L164p, act, Z, cw, geo = S["n"], S["fsh"], S["L164p"], S["act"], S["Z"], S["cw"], S["geo"]
T = dict(np.load(out + "/model/tables.npz", allow_pickle=False)); pg = pd.Series(T["pgr_sum"], index=T["pgr_idx"].astype(str)); pn = pd.Series(T["pgr_n"], index=T["pgr_idx"].astype(str))
pgm = float((T["prior"] @ G).mean()); t_p = ((pg + 2400.0 * pgm) / (pn + 2400.0)).reindex(n.index).fillna(pgm)     # 현재 anchor 이력 목표(투수 수준 근사)
W = np.sqrt(n.to_numpy()); A0 = np.c_[np.ones(len(n)), L164p.to_numpy(), np.log1p(n.to_numpy()), fsh.reindex(n.index).to_numpy()]
orth0 = lambda z: z - A0 @ np.linalg.lstsq(A0 * W[:, None], z.to_numpy() * W, rcond=None)[0]
comp = {"GEO": 0.3 * geo}; others = {}
for k, sdv in (("SIT", .25), ("SH", .20), ("PLT", .20), ("DIR", .15), ("DSHs", .12)):
    z = orth0(Z[k].reindex(n.index).fillna(0.0)); others[k] = z; comp[k] = (sdv / float(z.reindex(act).std())) * cw(z)
A1 = np.c_[A0, t_p.to_numpy(), geo.to_numpy()] + 0.0; A1 = np.c_[A1, np.column_stack([others[k].to_numpy() for k in others])]
rk = Z["RK"].reindex(n.index).fillna(0.0); rk2 = rk - A1 @ np.linalg.lstsq(A1 * W[:, None], rk.to_numpy() * W, rcond=None)[0]
rk0 = orth0(rk); P(f"RK 재직교화: 기존 잔차와 상관 {np.corrcoef(rk0.reindex(act), rk2.reindex(act))[0,1]:+.4f} · 현재 목표 t_p 와 상관 기존 {np.corrcoef(rk0.reindex(act), t_p.reindex(act))[0,1]:+.3f} → 새 {np.corrcoef(rk2.reindex(act), t_p.reindex(act))[0,1]:+.3f}")
comp["RK"] = (0.25 / float(rk2.reindex(act).std())) * cw(rk2); o = sum(comp.values())
po0 = pd.Series(T["poff_val"], index=T["poff_idx"].astype(str)); P(f"poff 대비: 상관 {np.corrcoef(po0.reindex(act), o.reindex(act))[0,1]:+.4f} · 총 sd {float(o.reindex(act).std()):.3f} · 평균|Δ| {float((po0.reindex(act)-o.reindex(act)).abs().mean()):.3f}")
T["poff_idx"] = np.asarray(o.index.astype(str), dtype=object).astype("U32"); T["poff_val"] = o.to_numpy(np.float64); np.savez_compressed(out + "/model/tables.npz", **T)
meta["poff_kind"] = "GEO0.3+RK2(residualized on current target):0.25,SIT:0.25,SH:0.20,PLT:0.20,DIR:0.15,DSHs:0.12"; json.dump(meta, open(out + "/model/meta.json", "w", encoding="utf-8"), ensure_ascii=False); P("saved", out)
