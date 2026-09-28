# -*- coding: utf-8 -*-
"""SRC 디렉토리(TS형)의 대리등급 표를 라벨행 + W×프록시 로 재구성 → NAME 디렉토리 (zip 없음, 목표값 검산만)"""
import os, sys, json, shutil, importlib.util, numpy as np, pandas as pd
from common import *
src, name, W = sys.argv[1], sys.argv[2], float(sys.argv[3]); out = DN + f"/src/submit_{name}"
shutil.rmtree(out, ignore_errors=True); shutil.copytree(DN + f"/src/submit_{src}", out, ignore=shutil.ignore_patterns("__pycache__", "output", "data"))
d = load_d(); dl = d[d.lab]; L = dl; mu = L.pt.mean(); NK = int(d.pkey.max()) + 1
c = np.bincount(L.pkey.to_numpy(), minlength=NK).astype(float); s = np.bincount(L.pkey.to_numpy(), weights=L.pt.to_numpy(), minlength=NK); PX = (s + 30. * mu) / (c + 30.)
fill = np.where(d.lab.to_numpy(), np.nan_to_num(d.pt.to_numpy()), PX[d.pkey.to_numpy()]); sea = d.season.to_numpy()
fill = fill - pd.Series(fill).groupby(sea).transform("mean").to_numpy() + float(fill.mean()); w = np.where(d.lab.to_numpy(), 1.0, W)
F = pd.DataFrame({"p": d.pitcher_id.to_numpy(), "v": fill * w, "w": w}).groupby("p").agg(sum=("v", "sum"), size=("w", "sum")); F = F[F["size"] > 0]
T = dict(np.load(out + "/model/tables.npz", allow_pickle=False)); old = pd.Series(T["pgr_sum"] / T["pgr_n"], index=T["pgr_idx"].astype(str))
T["pgr_idx"] = np.asarray(F.index.astype(str), dtype=object).astype("U32"); T["pgr_sum"] = F["sum"].to_numpy(np.float64); T["pgr_n"] = F["size"].to_numpy(np.float64)
np.savez_compressed(out + "/model/tables.npz", **T); m = json.load(open(out + "/model/meta.json", encoding="utf-8")); m["pgr_w_unlab"] = W; json.dump(m, open(out + "/model/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
new = (F["sum"] / F["size"]).reindex(old.index); P(f"pgr w={W}: 투수 {len(F)} · 평균 n {F['size'].mean():.0f} · 투수평균 변화 sd {float((new-old).std()):.3f} · 상관 {np.corrcoef(old.fillna(old.mean()), new.fillna(new.mean()))[0,1]:.4f}")
def load_mod(nm, path):
    sp = importlib.util.spec_from_file_location(nm, path); mm = importlib.util.module_from_spec(sp); sp.loader.exec_module(mm); return mm
O = load_mod("n_", out + "/script.py"); B = O.make_bundle(out); O0 = load_mod("s_", DN + f"/src/submit_{src}/script.py"); B0 = O0.make_bundle(DN + f"/src/submit_{src}")
L24 = dl[dl.season == 2024]; X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); pid24 = L24.pitcher_id.to_numpy()
s0 = score(y24, O0.predict(X24, B0), pid24, f"{src} (2024 인샘플)"); s1 = score(y24, O.predict(X24, B), pid24, name)
h0, h1 = s0[3], s1[3]; P(f"  Δ Pitch {s1[1]-s0[1]:+.4f} Player {s1[2]-s0[2]:+.4f} · 투수 mean t 상관 {np.corrcoef(h0, h1)[0,1]:+.4f} · 순위이동 {float((h0.rank()-h1.rank()).abs().mean()):.1f}/{len(h0)}")
