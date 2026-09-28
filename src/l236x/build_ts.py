# -*- coding: utf-8 -*-
"""TS_P600G10(r12 조건부 표) 파쿠리 + 변형 — 표 값만 교체(키 pid|cnt|sh 동일 → script 무수정):
   python3 build_ts.py NAME VARIANT [--sd 0.42] [--lamS 900 --lamH 500] [--lam1 300 --lam2 600] [--ymin 2023] [--rlam 600] [--nozip]
   VARIANT: base   원본 재현(λ600)                          dir   등급별(Shadow λS·Heart λH) 조건부 → 100·ZS+30·ZH
            hier   2단 부분풀링: 투수×카운트(λ1) → ×손(λ2)   rec   최근 시즌(≥ymin)만으로 표
            dirhier  dir + hier 결합 · 표 sd 를 --sd 로 재스케일(정점 0.42)"""
import os, sys, json, shutil, zipfile, importlib.util, numpy as np, pandas as pd
from common import *
name, var = sys.argv[1], sys.argv[2]; a = sys.argv; opt = lambda k, dft: float(a[a.index(k) + 1]) if k in a else dft
SD = opt("--sd", 0.42); lamS, lamH = opt("--lamS", 900), opt("--lamH", 500); lam1, lam2 = opt("--lam1", 300), opt("--lam2", 600); ymin = int(opt("--ymin", 2019)); rlam = opt("--rlam", 600); nozip = "--nozip" in a
BASE_TS = JM + "/submit_TS_P600G10"; out = DN + f"/src/submit_{name}"; zp = DN + f"/out/submit_{name}.zip"; assert not os.path.exists(zp), zp
d = load_d(); dl = d[d.lab].copy(); dl["sh"] = (dl.pitcher_hand == dl.batter_hand).astype(int)
cen = lambda v: v - v.groupby(dl.season).transform("mean") + float(v.mean())
dl["ptc"] = cen(dl.pt); dl["sc"] = cen((dl.y == 0).astype(float)); dl["hc"] = cen((dl.y == 1).astype(float))
def ztab(v, lam, rows, key, lvl_key=None, lvl_val=None):
    """Z[key] = n/(n+λ)·((셀 평균 − 상위 평균) − (리그 셀 평균 − 리그 상위 평균)). 상위 = 투수(기본) 또는 lvl_val(투수×카운트)"""
    x = rows; g = x[v].groupby(key).agg(["mean", "size"]); parts = pd.Series(g.index, index=g.index).str.split("|")
    pid = parts.str[0]; cellk = parts.str[1:].str.join("|")
    if lvl_val is None: up = pid.map(x.groupby("pitcher_id")[v].mean()); lup = float(x[v].mean())
    else: up = pd.Series(g.index, index=g.index).map(lvl_val).fillna(pid.map(x.groupby("pitcher_id")[v].mean())); lup = lvl_key
    lc = x[v].groupby(cellk.reindex(key.index) if False else (key.str.split("|").str[1:].str.join("|"))).mean()
    return g["size"] / (g["size"] + lam) * ((g["mean"] - up) - (cellk.map(lc) - (lup if np.isscalar(lup) else cellk.map(lup))))
rows = dl[dl.season >= ymin]; K3 = rows.pitcher_id.astype(str) + "|" + rows.cnt.astype(int).astype(str) + "|" + rows.sh.astype(str)
K2 = rows.pitcher_id.astype(str) + "|" + rows.cnt.astype(int).astype(str)
if var == "base": Z = ztab("ptc", 600, rows, K3)
elif var == "rec": Z = ztab("ptc", rlam, rows, K3)
elif var == "dir": Z = 100 * ztab("sc", lamS, rows, K3) + 30 * ztab("hc", lamH, rows, K3)
elif var in ("hier", "dirhier"):
    def hier(v, l1, l2):
        Z2 = ztab(v, l1, rows, K2)                                            # 투수×카운트 편차 (리그 카운트 효과 제거)
        g2 = rows[v].groupby(K2).mean(); lc2 = rows[v].groupby(rows.cnt.astype(int).astype(str)).mean()
        g3 = rows[v].groupby(K3).agg(["mean", "size"]); p3 = pd.Series(g3.index, index=g3.index).str.split("|")
        k2 = p3.str[0] + "|" + p3.str[1]; lc3 = rows[v].groupby(rows.cnt.astype(int).astype(str) + "|" + rows.sh.astype(str)).mean(); c3 = p3.str[1] + "|" + p3.str[2]
        Z3 = g3["size"] / (g3["size"] + l2) * ((g3["mean"] - k2.map(g2)) - (c3.map(lc3) - p3.str[1].map(lc2)))   # ×손 편차 (투수×카운트 기준 · 리그 손 효과 제거)
        return k2.map(Z2).fillna(0.0) + Z3
    Z = hier("ptc", lam1, lam2) if var == "hier" else 100 * hier("sc", lam1 * 1.5, lam2 * 1.5) + 30 * hier("hc", lam1 * 0.83, lam2 * 0.83)
elif var == "lowrank":
    # ★ 요인분석: 투수×(카운트12×손2=24칸) 편차행렬을 가중 저랭크(k) 로 재구성 — 칸별 수축 대신 칸 간 정보를 공유(소이력 투수의 빈 칸을 요인으로 채움)
    k = int(opt("--rank", 2)); lamp = opt("--lamp", 600)
    g = rows["ptc"].groupby(K3).agg(["mean", "size"]); parts = pd.Series(g.index, index=g.index).str.split("|"); pid = parts.str[0]; cellk = parts.str[1] + "|" + parts.str[2]
    pm = rows.groupby("pitcher_id")["ptc"].mean(); lc = rows["ptc"].groupby(rows.cnt.astype(int).astype(str) + "|" + rows.sh.astype(str)).mean(); lmu = float(rows["ptc"].mean())
    dev = (g["mean"] - pid.map(pm)) - (cellk.map(lc) - lmu)
    M = pd.DataFrame({"p": pid.to_numpy(), "c": cellk.to_numpy(), "v": dev.to_numpy(), "w": g["size"].to_numpy()})
    V = M.pivot(index="p", columns="c", values="v").fillna(0.0); W = M.pivot(index="p", columns="c", values="w").fillna(0.0)
    Wn = W / (W + 50.0); Xh = (Wn * V).to_numpy()                      # 가중 SVD (EM 3회): 관측 가중 n/(n+50)
    for _ in range(3):
        U, s_, Vt = np.linalg.svd(Xh, full_matrices=False); R = (U[:, :k] * s_[:k]) @ Vt[:k]; Xh = Wn.to_numpy() * V.to_numpy() + (1 - Wn.to_numpy()) * R
    U, s_, Vt = np.linalg.svd(Xh, full_matrices=False); R = pd.DataFrame((U[:, :k] * s_[:k]) @ Vt[:k], index=V.index, columns=V.columns)
    npit = rows.groupby("pitcher_id").size().reindex(V.index).fillna(0.0); R = R.mul(npit / (npit + lamp), axis=0)   # 투수 전체 n 으로 한 번 더 수축
    Z = R.stack(); Z.index = [f"{p}|{c}" for p, c in Z.index]; Z = pd.Series(Z.to_numpy(), index=Z.index)
    P(f"  lowrank k={k}: 특이값 {np.round(s_[:5], 2)} · 설명비 {np.round((s_[:k]**2).sum()/(s_**2).sum(), 3)} · 칸 {len(Z)}")
else: raise SystemExit(var)
Z = Z.dropna(); sc0 = float(Z.std()); Z = Z * (SD / sc0)
R0 = pd.Series(np.load(BASE_TS + "/model/tables.npz", allow_pickle=False)["r12_val"], index=np.load(BASE_TS + "/model/tables.npz", allow_pickle=False)["r12_idx"])
j = R0.to_frame("t").join(Z.rename("z"), how="outer").fillna(0.0)
P(f"{name}[{var}]: 셀 {len(Z)} · 원 sd {sc0:.4f} → 재스케일 {SD} · 원표(TS) 와 상관 {np.corrcoef(j.t, j.z)[0,1]:+.4f} · 평균|Δ| {(j.t-j.z).abs().mean():.3f}")
shutil.rmtree(out, ignore_errors=True); shutil.copytree(BASE_TS, out, ignore=shutil.ignore_patterns("__pycache__", "output", "data"))
T = dict(np.load(out + "/model/tables.npz", allow_pickle=False)); T["r12_idx"] = np.asarray(Z.index.astype(str), dtype=object).astype("U48"); T["r12_val"] = Z.to_numpy(np.float64)
np.savez_compressed(out + "/model/tables.npz", **T); m = json.load(open(out + "/model/meta.json", encoding="utf-8")); m["r12_g"] = 1.0; m["r12_kind"] = f"{var} sd{SD} lamS{lamS} lamH{lamH} lam1{lam1} lam2{lam2} ymin{ymin} rlam{rlam}"
json.dump(m, open(out + "/model/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
def load_mod(nm, path):
    sp = importlib.util.spec_from_file_location(nm, path); mm = importlib.util.module_from_spec(sp); sp.loader.exec_module(mm); return mm
O = load_mod("tsnew", out + "/script.py"); B = O.make_bundle(out); O0 = load_mod("ts0", BASE_TS + "/script.py"); B0 = O0.make_bundle(BASE_TS)
tr = d[B29]; rng = np.random.default_rng(0); Tst = tr.loc[rng.choice(len(tr), 30000, replace=False)].reset_index(drop=True); Pp = O.predict(Tst, B); wv = 0.0
for a_, b_ in [(0, 7000), (7000, 15000), (15000, 30000)]: wv = max(wv, np.abs(O.predict(Tst.iloc[a_:b_].copy(), B) - Pp[a_:b_]).max())
ix = rng.permutation(len(Tst))[:9000]; wv = max(wv, np.abs(O.predict(Tst.iloc[ix].copy(), B) - Pp[ix]).max())
for i in rng.choice(len(Tst), 30, replace=False): wv = max(wv, np.abs(O.predict(Tst.iloc[[i]].copy(), B)[0] - Pp[i]).max())
wv = max(wv, np.abs(O.predict(Tst.iloc[::-1].copy(), B)[::-1] - Pp).max()); P(f"[게이트] 행 독립 최대차 {wv:.3e} · 확률합 {np.abs(Pp.sum(1)-1).max():.2e} · 최소 {Pp.min():.2e}  {'✓' if wv < 1e-10 else '★실패'}"); assert wv < 1e-10
L24 = dl[dl.season == 2024]; X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); pid24 = L24.pitcher_id.to_numpy()
s0 = score(y24, O0.predict(X24, B0), pid24, "TS_P600G10 (2024 인샘플)"); s1 = score(y24, O.predict(X24, B), pid24, name)
h0, h1 = s0[3], s1[3]; P(f"  Δ Pitch {s1[1]-s0[1]:+.4f} Player {s1[2]-s0[2]:+.4f} · 투수 mean t 상관 {np.corrcoef(h0, h1)[0,1]:+.4f} · 순위이동 {float((h0.rank()-h1.rank()).abs().mean()):.1f}/{len(h0)}")
if nozip: sys.exit(0)
z = zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED)
for f in ("script.py", "requirements.txt"): z.write(f"{out}/{f}", f)
for root, _, files in os.walk(f"{out}/model"):
    if "__pycache__" in root: continue
    for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), out))
z.close(); P(f"  zip → {zp} {os.path.getsize(zp)/1e6:.1f}MB")
with open(DN + "/out/submissions.md", "a", encoding="utf-8") as f:
    f.write(f"| submit_{name}.zip | **TS_P600G10 파쿠리** r12 표 교체: {m['r12_kind']} (TS 표와 상관 {np.corrcoef(j.t, j.z)[0,1]:+.3f}) | 2024 인샘플 TS 대비 Pitch {s1[1]-s0[1]:+.4f} Player {s1[2]-s0[2]:+.4f} · 순위이동 {float((h0.rank()-h1.rank()).abs().mean()):.1f}/{len(h0)} | |\n")
