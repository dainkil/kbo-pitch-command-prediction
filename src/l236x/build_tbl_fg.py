# -*- coding: utf-8 -*-
"""TS_P600G10 위에서 **anchor 이력 표(pgr)만** 델타 패치 — F 행 재측정 (데이터 증강/전처리 계열).
   python3 build_tbl.py NAME VARIANT [--c 1.0] [--nozip]
   soft  : 2019~22 F 비반대투구 행 → pt' = (1 − c·p̂(rev|위치·구종·카운트·손·이닝)) · pt   (p̂ 는 2023+ F 행으로 학습, dain d14 방식)
   gt23  : 2019~22 F 행 → 기하등급(100/30/0) × (1 − r̄_F,2023+)      (태그 체제 무관 재채점)
   gtall : 모든 F 행 → 기하등급 × (1 − r̄_F,2023+)
   PGR2·프록시 채움·시즌중심화는 그대로 두고 라벨 F 행의 Δpt 만 정확히 반영: Δsum_p = Σ_p (Δpt − Δc_season + Δglobal)"""
import os, sys, json, shutil, zipfile, importlib.util, numpy as np, pandas as pd, lightgbm as lgb
from common import *
name, var = sys.argv[1], sys.argv[2]; a = sys.argv; C = float(a[a.index("--c") + 1]) if "--c" in a else 1.0; nozip = "--nozip" in a
BASE_TS = JM + "/submit_FIN_G/model/jm_root"; out = DN + f"/src/submit_{name}"; zp = DN + f"/out/submit_{name}.zip"; assert not os.path.exists(zp), zp
d = load_d(); lab = d.lab.to_numpy(); isF = (d.isF == 1).to_numpy(); pre = (d.season <= 2022).to_numpy(); rev = d.rev.to_numpy() == 1
x = d.plate_loc_side.abs(); h = d.plate_loc_height; heart = (x <= 0.263) & (h >= 0.537) & (h <= 0.963); shadow = (x <= 0.373) & (h >= 0.427) & (h <= 1.073) & ~heart
geo = np.where(heart, 30.0, np.where(shadow, 100.0, 0.0)); rF23 = float(d[lab & isF & (d.season >= 2023)].rev.mean()); pt = np.nan_to_num(d.pt.to_numpy())
dpt = np.zeros(len(d))
if var == "soft":
    F23 = d[lab & isF & (d.season >= 2023)]; tgt = lab & isF & pre & ~rev
    feats = lambda q: pd.DataFrame({"side": q.plate_loc_side, "h": q.plate_loc_height, "abs": q.plate_loc_side.abs(), "heart": ((q.plate_loc_side.abs() <= 0.263) & (q.plate_loc_height >= 0.537) & (q.plate_loc_height <= 0.963)).astype(int), "box": ((q.plate_loc_side.abs() <= 0.373) & (q.plate_loc_height >= 0.427) & (q.plate_loc_height <= 1.073)).astype(int), "pk": pd.to_numeric(q.PITCH_KIND, errors="coerce"), "b": q.balls_before, "st": q.strikes_before, "ph": q.pitcher_hand, "bh": q.batter_hand, "inn": q.inning}).astype(float)
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, verbose=-1, num_threads=8), lgb.Dataset(feats(F23), label=F23.rev.to_numpy()), 300)
    ph = m.predict(feats(d[tgt])); dpt[tgt] = -C * ph * pt[tgt]
    P(f"soft: p̂ 학습 2023+ F {len(F23):,}행 (rev {F23.rev.mean():.3f}) · 대상 2019~22 F 비반대 {int(tgt.sum()):,}행 · p̂ 평균 {ph.mean():.4f} · pt 평균 {pt[tgt].mean():.2f} → {(pt[tgt]+dpt[tgt]).mean():.2f}")
elif var in ("gt23", "gtall"):
    tgt = lab & isF & (pre if var == "gt23" else np.ones(len(d), bool)); dpt[tgt] = C * (geo[tgt] * (1 - rF23) - pt[tgt])
    P(f"{var}: r̄_F,2023+ {rF23:.4f} · 대상 {int(tgt.sum()):,}행 · pt 평균 {pt[tgt].mean():.2f} → {(pt[tgt]+dpt[tgt]).mean():.2f}")
else: raise SystemExit(var)
sea = d.season.to_numpy(); Ns = pd.Series(sea).value_counts(); dcs = pd.Series(dpt).groupby(sea).sum() / Ns; dglob = dpt.sum() / len(d)
adj = dpt - pd.Series(sea).map(dcs).to_numpy() + dglob; dsum = pd.Series(adj).groupby(d.pitcher_id.to_numpy()).sum()
P(f"  시즌 중심 이동 {dcs.round(3).to_dict()} · 전역 {dglob:+.3f} · 투수 Δsum 비0 {int((dsum.abs()>1e-9).sum())}명")
shutil.rmtree(out, ignore_errors=True); shutil.copytree(BASE_TS, out, ignore=shutil.ignore_patterns("__pycache__", "output", "data"))
T = dict(np.load(out + "/model/tables.npz", allow_pickle=False)); idx = pd.Index(T["pgr_idx"].astype(str)); add = dsum.reindex(idx).fillna(0.0).to_numpy()
old = T["pgr_sum"] / T["pgr_n"]; T["pgr_sum"] = T["pgr_sum"] + add; new = T["pgr_sum"] / T["pgr_n"]
P(f"  pgr 투수평균 변화: sd {np.std(new-old):.3f} · 최대 {np.abs(new-old).max():.2f} · 상관(old,new) {np.corrcoef(old,new)[0,1]:.4f}")
np.savez_compressed(out + "/model/tables.npz", **T); m = json.load(open(out + "/model/meta.json", encoding="utf-8")); m["pgr_patch"] = f"{var} c={C}"; json.dump(m, open(out + "/model/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
def load_mod(nm, path):
    sp = importlib.util.spec_from_file_location(nm, path); mm = importlib.util.module_from_spec(sp); sp.loader.exec_module(mm); return mm
O = load_mod("tsnew", out + "/script.py"); B = O.make_bundle(out); O0 = load_mod("ts0", BASE_TS + "/script.py"); B0 = O0.make_bundle(BASE_TS)
tr = d[B29]; rng = np.random.default_rng(0); Tst = tr.loc[rng.choice(len(tr), 30000, replace=False)].reset_index(drop=True); Pp = O.predict(Tst, B); wv = 0.0
for a_, b_ in [(0, 7000), (7000, 15000), (15000, 30000)]: wv = max(wv, np.abs(O.predict(Tst.iloc[a_:b_].copy(), B) - Pp[a_:b_]).max())
ix = rng.permutation(len(Tst))[:9000]; wv = max(wv, np.abs(O.predict(Tst.iloc[ix].copy(), B) - Pp[ix]).max())
for i in rng.choice(len(Tst), 30, replace=False): wv = max(wv, np.abs(O.predict(Tst.iloc[[i]].copy(), B)[0] - Pp[i]).max())
wv = max(wv, np.abs(O.predict(Tst.iloc[::-1].copy(), B)[::-1] - Pp).max()); P(f"[게이트] 행 독립 최대차 {wv:.3e} · 확률합 {np.abs(Pp.sum(1)-1).max():.2e} · 최소 {Pp.min():.2e}  {'✓' if wv < 1e-10 else '★실패'}"); assert wv < 1e-10
dl = d[d.lab]; L24 = dl[dl.season == 2024]; X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); pid24 = L24.pitcher_id.to_numpy()
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
    f.write(f"| submit_{name}.zip | **TS_P600G10 + anchor 이력표 F행 재측정** {m['pgr_patch']} (pgr 평균 변화 sd {np.std(new-old):.3f}) | 2024 인샘플 TS 대비 Pitch {s1[1]-s0[1]:+.4f} Player {s1[2]-s0[2]:+.4f} · 순위이동 {float((h0.rank()-h1.rank()).abs().mean()):.1f}/{len(h0)} | |\n")
