import json, numpy as np, pandas as pd
from common import *
d = load_d(); dl = d[d.lab]; L24 = dl[dl.season == 2024]
X24 = L24[B29].reset_index(drop=True); y24 = L24.y.to_numpy().astype(int); pid24 = L24.pitcher_id.to_numpy(); meta = base_meta()
U = dict(keys=meta["use_keys"], b=meta["use_b"], mu=meta["use_mu"], sd=meta["use_sd"], w=True, aw=meta["use_aw"], floor=0.0, g=meta["use_g"])
def run(tag, patch):
    B = SC.make_bundle(BASE); B.update(patch); return score(y24, SC.predict(X24, B), pid24, tag)
base = run("L236 base", {})
for nc in (800, 1000, 1500, 2000, 2500):
    s = run(f"ncut{nc}", {"use_g": 0.0, "uterms": [dict(U, ncut=nc)]}); P(f"     Δ Pitch {s[1]-base[1]:+.4f} Player {s[2]-base[2]:+.4f}")
