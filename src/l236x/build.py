# -*- coding: utf-8 -*-
"""빌드: python3 build.py NAME 'RK:0.25,...' ['{"meta 갱신 json"}'] [--bc] [--nozip]
   base(L236 zip) 복사 → poff 표·meta 주입 → 게이트(행 독립·형식·L236 대비 목표값 비교) → out/submit_NAME.zip → out/submissions.md"""
import os, sys, json, shutil, zipfile, time, numpy as np, pandas as pd
from common import *
import signals as SG
name, spec = sys.argv[1], sys.argv[2]; mup = json.loads(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].startswith("{") else {}
bc = "--bc" in sys.argv; nozip = "--nozip" in sys.argv
out = DN + f"/src/submit_{name}"; zp = DN + f"/out/submit_{name}.zip"
assert not os.path.exists(zp), f"{zp} 이미 있음 — 번호 확인"
meta0 = base_meta(); d = load_d(); S = SG.build(d, meta0)
o, comp = SG.make_poff(S, spec, bc=bc)
po0 = pd.Series(base_tables()["poff_val"], index=base_tables()["poff_idx"]); ad = po0.reindex(S["act"]); bd = o.reindex(S["act"])
P(f"  L236 poff 대비: 상관 {np.corrcoef(ad, bd)[0,1]:+.4f} · 평균 |Δ| {float((ad-bd).abs().mean()):.3f}")
shutil.rmtree(out, ignore_errors=True); shutil.copytree(BASE, out, ignore=shutil.ignore_patterns("__pycache__", "output", "data"))
T = dict(np.load(f"{out}/model/tables.npz", allow_pickle=False)); T["poff_idx"] = np.asarray(o.index.astype(str), dtype=object).astype("U32"); T["poff_val"] = o.to_numpy(np.float64)
np.savez_compressed(f"{out}/model/tables.npz", **T)
m = dict(meta0); m.update(poff_g=1.0, poff_default=0.0, poff_kind="GEO0.3+" + spec + ("|bc" if bc else "")); m.update(mup)
if "--ncut" in sys.argv:   # 카운트 항을 uterms 로 옮기고 이력 ncut 초과 투수에서 끈다 (eval24: ncut1000 2024 인샘플 Pitch +0.024 · Player +0.32)
    nc = float(sys.argv[sys.argv.index("--ncut") + 1]); U = dict(keys=meta0["use_keys"], b=meta0["use_b"], mu=meta0["use_mu"], sd=meta0["use_sd"], w=True, aw=meta0["use_aw"], floor=0.0, g=meta0["use_g"], ncut=nc)
    m["use_g"] = 0.0; m["uterms"] = [U]; mup = dict(mup, ncut=nc)
json.dump(m, open(f"{out}/model/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
# ---- 게이트 ----
B = SC.make_bundle(out); B0 = SC.make_bundle(BASE)
tr = d[B29 + ["y", "lab", "pt"]]; L24 = tr[(tr.season == 2024) & tr.lab]
rng = np.random.default_rng(0); Tst = tr.loc[rng.choice(len(tr), 30000, replace=False), B29].reset_index(drop=True)
Pp = SC.predict(Tst, B); w = 0.0
for a, b in [(0, 7000), (7000, 15000), (15000, 30000)]: w = max(w, np.abs(SC.predict(Tst.iloc[a:b].copy(), B) - Pp[a:b]).max())
ix = rng.permutation(len(Tst))[:9000]; w = max(w, np.abs(SC.predict(Tst.iloc[ix].copy(), B) - Pp[ix]).max())
for i in rng.choice(len(Tst), 30, replace=False): w = max(w, np.abs(SC.predict(Tst.iloc[[i]].copy(), B)[0] - Pp[i]).max())
w = max(w, np.abs(SC.predict(Tst.iloc[::-1].copy(), B)[::-1] - Pp).max())
P(f"[게이트] 행 독립 최대차 {w:.3e} · 확률합 {np.abs(Pp.sum(1)-1).max():.2e} · 최소 {Pp.min():.2e}  {'✓' if w < 1e-10 else '★실패'}")
assert w < 1e-10
X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); pid24 = L24.pitcher_id.to_numpy()
t0 = time.time(); P1 = SC.predict(X24, B); P(f"  2024 라벨행 {len(X24):,} 추론 {time.time()-t0:.1f}s")
P0 = SC.predict(X24, B0)
s0 = score(y24, P0, pid24, "L236 (2024 라벨행·인샘플)"); s1 = score(y24, P1, pid24, f"{name}")
h0, h1 = s0[3], s1[3]; P(f"  투수 mean t 상관 {np.corrcoef(h0, h1)[0,1]:+.5f} · 평균 순위이동 {float((h0.rank()-h1.rank()).abs().mean()):.1f}/{len(h0)} · ΔPitch {s1[1]-s0[1]:+.4f} ΔPlayer {s1[2]-s0[2]:+.4f}")
if nozip: P("  (--nozip)"); sys.exit(0)
z = zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED)
for f in ("script.py", "requirements.txt"): z.write(f"{out}/{f}", f)
for root, _, files in os.walk(f"{out}/model"):
    for f in files: z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), out))
z.close(); P(f"  zip → {zp} {os.path.getsize(zp)/1e6:.1f}MB · " + " ".join(sorted(zipfile.ZipFile(zp).namelist())))
with open(DN + "/out/submissions.md", "a", encoding="utf-8") as f:
    f.write(f"| submit_{name}.zip | L236 + poff[{spec}{'|bc' if bc else ''}] meta{json.dumps(mup, ensure_ascii=False) if mup else ''} | 2024 인샘플 L236 대비 ΔPitch {s1[1]-s0[1]:+.4f} ΔPlayer {s1[2]-s0[2]:+.4f} · 순위이동 {float((h0.rank()-h1.rank()).abs().mean()):.1f}/{len(h0)} | |\n")
