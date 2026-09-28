# -*- coding: utf-8 -*-
"""스택: (pgr 패치된 빌드) + r12 표 = 여러 변형 평균(잡음 감소·하방 축소) → sd 0.42 재스케일.  python3 build_stack.py SRC NAME "d34dir,d34hier,TS" [--sd 0.42]"""
import os, sys, json, shutil, zipfile, importlib.util, numpy as np, pandas as pd
from common import *
src, name, comps = sys.argv[1], sys.argv[2], sys.argv[3].split(","); a = sys.argv; SD = float(a[a.index("--sd") + 1]) if "--sd" in a else 0.42; nozip = "--nozip" in a
BASE_TS = JM + "/submit_TS_P600G10"; out = DN + f"/src/submit_{name}"; zp = DN + f"/out/submit_{name}.zip"; assert not os.path.exists(zp), zp
def r12(dirp):
    A = np.load(dirp + "/model/tables.npz", allow_pickle=False); return pd.Series(A["r12_val"], index=A["r12_idx"].astype(str))
tabs = [r12(BASE_TS if c == "TS" else DN + f"/src/submit_{c}") for c in comps]; M = pd.concat(tabs, axis=1).fillna(0.0); Z = M.mean(1); Z = Z * (SD / Z.std())
P(f"r12 평균 {comps}: 성분 상관\n{M.corr().round(3).to_string()}\n  평균표 sd → {SD} · TS 표와 상관 {np.corrcoef(M.iloc[:,comps.index('TS')] if 'TS' in comps else tabs[0].reindex(M.index).fillna(0), Z)[0,1]:+.4f}")
shutil.rmtree(out, ignore_errors=True); shutil.copytree(DN + f"/src/submit_{src}", out, ignore=shutil.ignore_patterns("__pycache__", "output", "data"))
T = dict(np.load(out + "/model/tables.npz", allow_pickle=False)); T["r12_idx"] = np.asarray(Z.index.astype(str), dtype=object).astype("U48"); T["r12_val"] = Z.to_numpy(np.float64); np.savez_compressed(out + "/model/tables.npz", **T)
m = json.load(open(out + "/model/meta.json", encoding="utf-8")); m["r12_kind"] = f"avg({'+'.join(comps)}) sd{SD}"; json.dump(m, open(out + "/model/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
def load_mod(nm, path):
    sp = importlib.util.spec_from_file_location(nm, path); mm = importlib.util.module_from_spec(sp); sp.loader.exec_module(mm); return mm
d = load_d(); dl = d[d.lab]; O = load_mod("stk", out + "/script.py"); B = O.make_bundle(out); O0 = load_mod("ts0", BASE_TS + "/script.py"); B0 = O0.make_bundle(BASE_TS)
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
    f.write(f"| submit_{name}.zip | **스택** {src} + r12={m['r12_kind']} | 2024 인샘플 TS 대비 Pitch {s1[1]-s0[1]:+.4f} Player {s1[2]-s0[2]:+.4f} · 순위이동 {float((h0.rank()-h1.rank()).abs().mean()):.1f}/{len(h0)} | |\n")
