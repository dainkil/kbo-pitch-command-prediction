# -*- coding: utf-8 -*-
"""python3 build_hyb2.py NAME JMDIR W_DAIN Q_T [--nozip]  — Player 비트 불변(TS 목표) · q 평균/온도로 Pitch 만"""
import os, sys, json, shutil, zipfile, importlib.util, numpy as np, pandas as pd
from common import *
name, jmdir, W, QT = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]); nozip = "--nozip" in sys.argv
out = DN + f"/src/submit_{name}"; zp = DN + f"/out/submit_{name}.zip"; assert not os.path.exists(zp), zp
shutil.rmtree(out, ignore_errors=True); os.makedirs(out + "/model")
for f in ("lgb_seed0.txt", "lgb_seed1.txt", "lgb_seed2.txt", "lgb_seed3.txt", "lgb_seed4.txt", "meta.json"): shutil.copy(DN + "/src/submit_d20/model/" + f, out + "/model/" + f)
shutil.copytree(DN + "/src/submit_d20/model/tables", out + "/model/tables"); shutil.copy(DN + "/src/script_hybrid/requirements.txt", out + "/requirements.txt")
shutil.copy(DN + "/src/script_hybrid/script.py", out + "/dain_feats.py"); shutil.copy(X + "/hyb2_script.py", out + "/script.py")
os.makedirs(out + "/model/jm_root"); shutil.copytree(jmdir + "/model", out + "/model/jm_root/model"); shutil.copy(jmdir + "/script.py", out + "/model/jm_root/script.py")
json.dump({"w_dain": W, "q_T": QT, "jm_source": jmdir}, open(out + "/model/hyb.json", "w"))
def load_mod(nm, path):
    sp = importlib.util.spec_from_file_location(nm, path); mm = importlib.util.module_from_spec(sp); sp.loader.exec_module(mm); return mm
sys.path.insert(0, out); D = load_mod("hyb2", out + "/script.py"); pl = __import__("pathlib").Path; pred = lambda df: D.predict_all(df.reset_index(drop=True), pl(out + "/model"))
d = load_d(); dl = d[d.lab]; tr = d[B29]; rng = np.random.default_rng(0); Tst = tr.loc[rng.choice(len(tr), 30000, replace=False)].reset_index(drop=True); Pp = pred(Tst); wv = 0.0
for a_, b_ in [(0, 7000), (7000, 15000), (15000, 30000)]: wv = max(wv, np.abs(pred(Tst.iloc[a_:b_]) - Pp[a_:b_]).max())
ix = rng.permutation(len(Tst))[:9000]; wv = max(wv, np.abs(pred(Tst.iloc[ix]) - Pp[ix]).max())
for i in rng.choice(len(Tst), 20, replace=False): wv = max(wv, np.abs(pred(Tst.iloc[[i]])[0] - Pp[i]).max())
wv = max(wv, np.abs(pred(Tst.iloc[::-1])[::-1] - Pp).max()); P(f"[게이트] 행 독립 최대차 {wv:.3e} · 확률합 {np.abs(Pp.sum(1)-1).max():.2e} · 최소 {Pp.min():.2e}  {'✓' if wv < 1e-10 else '★실패'}"); assert wv < 1e-10
O0 = load_mod("jm0", jmdir + "/script.py"); B0 = O0.make_bundle(jmdir)
L24 = dl[dl.season == 2024]; X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); pid24 = L24.pitcher_id.to_numpy()
P0 = O0.predict(X24, B0); P1 = pred(X24); s0 = score(y24, P0, pid24, "TS (2024 인샘플)"); s1 = score(y24, P1, pid24, name)
sc0 = pd.Series(P0 @ G).groupby(pid24).mean(); sc1 = pd.Series(P1 @ G).groupby(pid24).mean()
P(f"  Δ Pitch {s1[1]-s0[1]:+.4f} Player {s1[2]-s0[2]:+.6f} · 투수 평균점수 최대차 {float((sc0-sc1).abs().max()):.2e} (Player 불변 검증)")
if nozip: sys.exit(0)
z = zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED)
for f in ("script.py", "dain_feats.py", "requirements.txt"): z.write(f"{out}/{f}", f)
for root, _, files in os.walk(f"{out}/model"):
    if "__pycache__" in root: continue
    for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), out))
z.close(); P(f"  zip → {zp} {os.path.getsize(zp)/1e6:.1f}MB")
with open(DN + "/out/submissions.md", "a", encoding="utf-8") as f:
    f.write(f"| submit_{name}.zip | **Player 불변 하이브리드**: t = {os.path.basename(jmdir)} 그대로 · q = {W:.2f}·dain5시드 + {1-W:.2f}·jm5시드, T={QT} · proj anchor | 2024 인샘플 Pitch {s1[1]-s0[1]:+.4f} · 투수평균 최대차 {float((sc0-sc1).abs().max()):.1e} | |\n")
