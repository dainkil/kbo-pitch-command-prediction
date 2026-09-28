# -*- coding: utf-8 -*-
"""d29: drift-anchor 하이브리드 = O57 목표 t + κ·[has]·(cmp_row − cmpbar_p) (행 단위 상황 보정, 이력 평균에 고정) · 등급 배분 q = dain 5시드.
   사용: python3 build_d29.py NAME KAPPA [--nozip]"""
import os, sys, json, shutil, zipfile, importlib.util, numpy as np, pandas as pd
from common import *
name, kap = sys.argv[1], float(sys.argv[2]); nozip = "--nozip" in sys.argv; K0 = 200.0
out = DN + f"/src/submit_{name}"; zp = DN + f"/out/submit_{name}.zip"; assert not os.path.exists(zp), zp
def load_mod(nm, path):
    sp = importlib.util.spec_from_file_location(nm, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
O = load_mod("o57s", DN + "/tmp_O57/script.py"); B57 = O.make_bundle(DN + "/tmp_O57")
d = load_d(); dl = d[d.lab]; L24 = dl[dl.season == 2024]
# 1) cmpbar_p 표: 전체 학습행(라벨+미라벨, 2019~2024)의 상황 기대점수 평균 · κ0=200 수축
Ha = d[B29].reset_index(drop=True); ch = O.B_CMP["m"].predict(O._cmp_feats(O.prep(Ha)).to_numpy(np.float32)) @ G
g = pd.DataFrame({"p": Ha.pitcher_id.to_numpy(), "c": ch}).groupby("p").c.agg(["sum", "size"]); mu = float(ch.mean())
cb = (g["sum"] + K0 * mu) / (g["size"] + K0)
P(f"cmpbar 표: 투수 {len(cb)} · 리그평균 {mu:.3f} · sd {cb.std():.3f} · 행 cmp sd {ch.std():.3f}")
# 2) jm_root = O57 + drift 패치
jr = out + "/model/jm_root"; shutil.rmtree(out, ignore_errors=True)
shutil.copytree(DN + "/src/submit_d20", out, ignore=shutil.ignore_patterns("__pycache__", "jm_root", "output", "data"))
shutil.copy(DN + "/src/script_hybrid/script.py", out + "/script.py"); shutil.copy(DN + "/src/script_hybrid/requirements.txt", out + "/requirements.txt")
os.makedirs(jr); shutil.copytree(DN + "/tmp_O57/model", jr + "/model"); shutil.copy(DN + "/tmp_O57/script.py", jr + "/script.py")
s = open(jr + "/script.py", encoding="utf-8").read()
old1 = '    if "poff_val" in A.files:'
new1 = '    if "cdr_val" in A.files:      # ★ drift-anchor: 투수별 이력 상황 기대점수 평균(학습 상수 · pitcher_id 조회 → 행 독립)\n        T["cdr_idx"] = pd.Index(A["cdr_idx"].astype(object)); T["cdr_val"] = A["cdr_val"].astype(float)\n' + old1
assert s.count(old1) == 1; s = s.replace(old1, new1)
old2 = '        return anchor(P, target_grade(d, B["tables"], pri, B, EB), w, mode=B.get("anchor_mode", "mult"))'
new2 = '''        t = target_grade(d, B["tables"], pri, B, EB)
        g_ = float(B.get("cdr_g", 0.0) or 0.0)
        if g_ and "cdr_val" in B["tables"] and B_CMP.get("m") is not None:
            # ★ drift-anchor: t += κ·(자기 행의 상황 기대점수 − 그 투수의 이력 상황 평균). 행 안에서 상황 보정을 복원(Pitch)하되
            #   투수 평균은 (2025 구성 − 이력 구성)만큼만 움직인다. 자기 행 29열 + 학습 표만 사용 → 행 독립. 이력 없는 투수는 0.
            cv = B_CMP["m"].predict(_cmp_feats(d).to_numpy(dtype=np.float32)) @ np.array([100.0, 30.0, 0.0])
            loc = B["tables"]["cdr_idx"].get_indexer(pd.Index(d["pitcher_id"].to_numpy())); ok = loc >= 0
            cb = np.where(ok, B["tables"]["cdr_val"][np.where(ok, loc, 0)], 0.0)
            t = t + g_ * ok.astype(float) * (cv - cb)
        return anchor(P, t, w, mode=B.get("anchor_mode", "mult"))'''
assert s.count(old2) == 1; s = s.replace(old2, new2); open(jr + "/script.py", "w", encoding="utf-8").write(s)
T = dict(np.load(jr + "/model/tables.npz", allow_pickle=False)); T["cdr_idx"] = np.asarray(cb.index.astype(str), dtype=object).astype("U32"); T["cdr_val"] = cb.to_numpy(np.float64)
np.savez_compressed(jr + "/model/tables.npz", **T)
m = json.load(open(jr + "/model/meta.json", encoding="utf-8")); m["cdr_g"] = kap; m["cdr_k0"] = K0; m["cdr_kind"] = "cmp(all rows 2019-24) shrink200 · has only"
json.dump(m, open(jr + "/model/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
# 3) dain script: SF(1,0,-1) → proj 최소비용 방향 (점수 1점당 (170,-40,-130)/3/5266.67 ×100)
ds = open(out + "/script.py", encoding="utf-8").read(); old3 = "SF = np.array([1.0, 0.0, -1.0])"
assert ds.count(old3) == 1; ds = ds.replace(old3, "SF = np.array([170.0, -40.0, -130.0]) / 3.0 / (15800.0 / 3.0) * 100.0   # proj 최소비용 방향 (합 0 · 점수 1점 이동)")
dm = json.load(open(out + "/model/meta.json", encoding="utf-8")); dm["jm_source"] = f"seongmin submit_O57 (LB 70.2797/70.0930/70.7154) + drift-anchor κ={kap}: 목표 t 는 jm_root, 등급 배분 q 는 dain 5시드"; json.dump(dm, open(out + "/model/meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
open(out + "/script.py", "w", encoding="utf-8").write(ds)
# 4) 게이트 + 2024 검산
D = load_mod("dainhy", out + "/script.py"); TT = D.load_tables(__import__("pathlib").Path(out + "/model"))
pred = lambda df: D.predict_all(df.reset_index(drop=True), __import__("pathlib").Path(out + "/model"), TT)
tr = d[B29 + ["y", "lab"]]; rng = np.random.default_rng(0); Tst = tr.loc[rng.choice(len(tr), 30000, replace=False), B29].reset_index(drop=True)
Pp = pred(Tst); w = 0.0
for a, b in [(0, 7000), (7000, 15000), (15000, 30000)]: w = max(w, np.abs(pred(Tst.iloc[a:b]) - Pp[a:b]).max())
ix = rng.permutation(len(Tst))[:9000]; w = max(w, np.abs(pred(Tst.iloc[ix]) - Pp[ix]).max())
for i in rng.choice(len(Tst), 20, replace=False): w = max(w, np.abs(pred(Tst.iloc[[i]])[0] - Pp[i]).max())
w = max(w, np.abs(pred(Tst.iloc[::-1])[::-1] - Pp).max())
P(f"[게이트] 행 독립 최대차 {w:.3e} · 확률합 {np.abs(Pp.sum(1)-1).max():.2e} · 최소 {Pp.min():.2e}  {'✓' if w < 1e-10 else '★실패'}"); assert w < 1e-10
X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); pid24 = L24.pitcher_id.to_numpy()
P0 = O.predict(X24, B57); s0 = score(y24, P0, pid24, "O57 (2024 라벨행·인샘플)")
Bj = load_mod("jmpatched", jr + "/script.py"); BJ = Bj.make_bundle(jr); P1 = Bj.predict(X24, BJ); s1 = score(y24, P1, pid24, f"O57 + drift κ={kap} (jm q)")
P2 = pred(X24); s2 = score(y24, P2, pid24, f"{name}: + dain q (proj)")
P(f"  Δ vs O57: drift(jm q) Pitch {s1[1]-s0[1]:+.4f} Player {s1[2]-s0[2]:+.4f} · +dain q Pitch {s2[1]-s1[1]:+.4f} Player {s2[2]-s1[2]:+.4f} · 합계 Pitch {s2[1]-s0[1]:+.4f} Player {s2[2]-s0[2]:+.4f}")
h0, h2 = s0[3], s2[3]; P(f"  투수 mean t 상관 {np.corrcoef(h0, h2)[0,1]:+.4f} · 평균 순위이동 {float((h0.rank()-h2.rank()).abs().mean()):.1f}/{len(h0)}")
if nozip: sys.exit(0)
z = zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED)
for f in ("script.py", "requirements.txt"): z.write(f"{out}/{f}", f)
for root, _, files in os.walk(f"{out}/model"):
    if "__pycache__" in root: continue
    for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), out))
z.close(); P(f"  zip → {zp} {os.path.getsize(zp)/1e6:.1f}MB")
with open(DN + "/out/submissions.md", "a", encoding="utf-8") as f:
    f.write(f"| submit_{name}.zip | **drift-anchor 하이브리드**: O57 목표 + κ={kap}·[이력有]·(cmp행 − 이력 cmp 평균, 수축200) · q = dain 5시드(proj) | 2024 인샘플 O57 대비 Pitch {s2[1]-s0[1]:+.4f} Player {s2[2]-s0[2]:+.4f} · as-of 홀드아웃 κ0.5 Pitch +0.087 Player +0.47 / κ1.0 +0.151 +0.34 | |\n")
