# -*- coding: utf-8 -*-
"""기존 하이브리드 빌드(예: submit_d29a) 위에 표/메타만 패치해 새 zip: python3 build_var.py BASE NAME [--proxy W] [--bat G] [--kappa K] [--nozip]
   --proxy W : 대리등급 표를 라벨행 + W×미라벨(proxy) 로 재구성 (W=0 → 라벨행만 · 현행 W=1 · PGR2 수정은 사라짐)
   --bat G   : 타자 편차 행 항 t += G·bdev[batter_id]  (bdev = 시즌중심 (pt − cmp) 의 타자 평균, α=400, 학습 라벨행)
   --kappa K : drift 세기 변경"""
import os, sys, json, shutil, zipfile, importlib.util, numpy as np, pandas as pd
from common import *
base, name = sys.argv[1], sys.argv[2]; a = sys.argv
opt = lambda k: float(a[a.index(k) + 1]) if k in a else None
Wp, Gb, Kd = opt("--proxy"), opt("--bat"), opt("--kappa"); nozip = "--nozip" in a
out = DN + f"/src/submit_{name}"; zp = DN + f"/out/submit_{name}.zip"; assert not os.path.exists(zp), zp
shutil.rmtree(out, ignore_errors=True); shutil.copytree(DN + f"/src/submit_{base}", out, ignore=shutil.ignore_patterns("__pycache__", "output", "data"))
jr = out + "/model/jm_root"; T = dict(np.load(jr + "/model/tables.npz", allow_pickle=False)); m = json.load(open(jr + "/model/meta.json", encoding="utf-8"))
def load_mod(nm, path):
    sp = importlib.util.spec_from_file_location(nm, path); mm = importlib.util.module_from_spec(sp); sp.loader.exec_module(mm); return mm
O = load_mod("jmp", jr + "/script.py"); O.make_bundle(jr); d = load_d(); dl = d[d.lab]; desc = []
if Wp is not None:
    L = dl; mu = L.pt.mean(); NK = int(d.pkey.max()) + 1
    c = np.bincount(L.pkey.to_numpy(), minlength=NK).astype(float); s = np.bincount(L.pkey.to_numpy(), weights=L.pt.to_numpy(), minlength=NK); PX = (s + 30. * mu) / (c + 30.)
    fill = np.where(d.lab.to_numpy(), np.nan_to_num(d.pt.to_numpy()), PX[d.pkey.to_numpy()]); sea = d.season.to_numpy()
    fill = fill - pd.Series(fill).groupby(sea).transform("mean").to_numpy() + float(fill.mean())      # PGR_CTR=1 (O57 과 동일 검증됨)
    w = np.where(d.lab.to_numpy(), 1.0, Wp); F = pd.DataFrame({"p": d.pitcher_id.to_numpy(), "v": fill * w, "w": w}).groupby("p").agg(sum=("v", "sum"), size=("w", "sum")); F = F[F["size"] > 0]
    T["pgr_idx"] = np.asarray(F.index.astype(str), dtype=object).astype("U32"); T["pgr_sum"] = F["sum"].to_numpy(np.float64); T["pgr_n"] = F["size"].to_numpy(np.float64)
    m["pgr_w_unlab"] = Wp; desc.append(f"proxy w={Wp}"); P(f"대리등급 표 재구성 w={Wp}: 투수 {len(F)} · 평균 n {F['size'].mean():.0f}")
if Gb is not None:
    cmp_all = O.B_CMP["m"].predict(O._cmp_feats(O.prep(dl[B29].reset_index(drop=True))).to_numpy(np.float32)) @ G
    r = dl.pt.to_numpy() - cmp_all; r = r - pd.Series(r).groupby(dl.season.to_numpy()).transform("mean").to_numpy()
    g = pd.DataFrame({"b": dl.batter_id.to_numpy(), "r": r}).groupby("b").r.agg(["sum", "size"]); bd = g["sum"] / (g["size"] + 400.0)
    T["bdev_idx"] = np.asarray(bd.index.astype(str), dtype=object).astype("U32"); T["bdev_val"] = bd.to_numpy(np.float64)
    m["bat_g"] = Gb; m["bdev_kind"] = "season-centered (pt − cmp) batter mean, α=400"; desc.append(f"bat g={Gb}")
    P(f"타자 편차 표: {len(bd)}명 · sd {bd.std():.3f} · 범위 {bd.min():+.2f}~{bd.max():+.2f}")
if Kd is not None: m["cdr_g"] = Kd; desc.append(f"drift κ={Kd}")
np.savez_compressed(jr + "/model/tables.npz", **T); json.dump(m, open(jr + "/model/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
dm = json.load(open(out + "/model/meta.json", encoding="utf-8")); dm["jm_source"] += " · " + " · ".join(desc); json.dump(dm, open(out + "/model/meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
# 게이트 + 2024 검산 (기준 = base 빌드)
D = load_mod("dainhy", out + "/script.py"); pl = __import__("pathlib").Path; TT = D.load_tables(pl(out + "/model")); pred = lambda df: D.predict_all(df.reset_index(drop=True), pl(out + "/model"), TT)
D0 = load_mod("dainhy0", DN + f"/src/submit_{base}/script.py"); T0 = D0.load_tables(pl(DN + f"/src/submit_{base}/model")); pred0 = lambda df: D0.predict_all(df.reset_index(drop=True), pl(DN + f"/src/submit_{base}/model"), T0)
tr = d[B29]; rng = np.random.default_rng(0); Tst = tr.loc[rng.choice(len(tr), 30000, replace=False)].reset_index(drop=True); Pp = pred(Tst); wv = 0.0
for a_, b_ in [(0, 7000), (7000, 15000), (15000, 30000)]: wv = max(wv, np.abs(pred(Tst.iloc[a_:b_]) - Pp[a_:b_]).max())
ix = rng.permutation(len(Tst))[:9000]; wv = max(wv, np.abs(pred(Tst.iloc[ix]) - Pp[ix]).max())
for i in rng.choice(len(Tst), 20, replace=False): wv = max(wv, np.abs(pred(Tst.iloc[[i]])[0] - Pp[i]).max())
wv = max(wv, np.abs(pred(Tst.iloc[::-1])[::-1] - Pp).max()); P(f"[게이트] 행 독립 최대차 {wv:.3e} · 확률합 {np.abs(Pp.sum(1)-1).max():.2e} · 최소 {Pp.min():.2e}  {'✓' if wv < 1e-10 else '★실패'}"); assert wv < 1e-10
L24 = dl[dl.season == 2024]; X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); pid24 = L24.pitcher_id.to_numpy()
s0 = score(y24, pred0(X24), pid24, f"{base} (2024 인샘플)"); s1 = score(y24, pred(X24), pid24, f"{name}")
h0, h1 = s0[3], s1[3]; P(f"  Δ Pitch {s1[1]-s0[1]:+.4f} Player {s1[2]-s0[2]:+.4f} · 투수 mean t 상관 {np.corrcoef(h0, h1)[0,1]:+.4f} · 순위이동 {float((h0.rank()-h1.rank()).abs().mean()):.1f}/{len(h0)}")
if nozip: sys.exit(0)
z = zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED)
for f in ("script.py", "requirements.txt"): z.write(f"{out}/{f}", f)
for root, _, files in os.walk(f"{out}/model"):
    if "__pycache__" in root: continue
    for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), out))
z.close(); P(f"  zip → {zp} {os.path.getsize(zp)/1e6:.1f}MB")
with open(DN + "/out/submissions.md", "a", encoding="utf-8") as f:
    f.write(f"| submit_{name}.zip | {base} + {' · '.join(desc)} | 2024 인샘플 {base} 대비 Pitch {s1[1]-s0[1]:+.4f} Player {s1[2]-s0[2]:+.4f} · 순위이동 {float((h0.rank()-h1.rank()).abs().mean()):.1f}/{len(h0)} | |\n")
