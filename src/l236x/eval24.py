# -*- coding: utf-8 -*-
"""Pitch 트랙 후보 선별 — L236 번들을 메모리에서 변형(meta 키 패치)해 2024 라벨행에서 Pitch/Player 비교 (q 인샘플 → 차이만 신뢰)."""
import json, numpy as np, pandas as pd
from common import *
d = load_d(); dl = d[d.lab]; L24 = dl[dl.season == 2024]
X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); pid24 = L24.pitcher_id.to_numpy()
meta = base_meta()
# cw_val: 투수×시즌 고정효과 제거 후 12카운트별 점수차 (학습 데이터 상수)
r = dl.pt - dl.groupby(["pitcher_id", "season"]).pt.transform("mean"); cwv = r.groupby(dl.cnt).mean().reindex(range(12)).fillna(0.0)
frq = dl.cnt.value_counts(normalize=True).reindex(range(12)).fillna(0.0); cwv = cwv - float((cwv * frq).sum())
P("cw_val (b*3+s):", cwv.round(3).to_dict())
U = dict(keys=meta["use_keys"], b=meta["use_b"], mu=meta["use_mu"], sd=meta["use_sd"], w=True, aw=meta["use_aw"], floor=0.0, g=meta["use_g"])
def run(tag, patch):
    B = SC.make_bundle(BASE); B.update(patch); Pp = SC.predict(X24, B); return score(y24, Pp, pid24, tag)
base = run("L236 base", {})
eq = run("uterms 동치판(ncut 없음)", {"use_g": 0.0, "uterms": [dict(U)]})
P(f"   동치 확인 ΔPitch {eq[1]-base[1]:+.6f} ΔPlayer {eq[2]-base[2]:+.6f}")
res = {}
for tag, patch in [
    ("ronly", {"use_g": 0.0, "uterms": [dict(U, ronly=True)]}),
    ("ncut3000", {"use_g": 0.0, "uterms": [dict(U, ncut=3000)]}),
    ("ncut2000", {"use_g": 0.0, "uterms": [dict(U, ncut=2000)]}),
    ("ncut3000+ronly", {"use_g": 0.0, "uterms": [dict(U, ncut=3000, ronly=True)]}),
    ("cw0.5", {"cw_g": 0.5, "cw_val": cwv.tolist()}),
    ("cw1.0", {"cw_g": 1.0, "cw_val": cwv.tolist()}),
    ("ncut3000+cw0.5", {"use_g": 0.0, "uterms": [dict(U, ncut=3000)], "cw_g": 0.5, "cw_val": cwv.tolist()}),
    ("use_g0 (카운트 항 제거·참고)", {"use_g": 0.0}),
]:
    s = run(tag, patch); res[tag] = (s[1] - base[1], s[2] - base[2])
P("\n요약 (ΔPitch, ΔPlayer vs L236, 2024 인샘플):"); [P(f"   {k:28s} {v[0]:+.4f}  {v[1]:+.4f}") for k, v in res.items()]
json.dump({"cw_val": cwv.tolist()}, open(X + "/cw_val.json", "w"))
