"""최종 후보 3폴드 평가 — EB / GBDT(noF) / anchor(λ) / anchor + 주입(신호셋 × ε). Total·Pitch·τ 셋 다 본다."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
d_all = load(labeled_only=False); tm = load_tm(); d = d_all[d_all.y.notna()]; rows = []
def rec(Y, m, r): rows.append(dict(Y=Y, m=m, tot=r[0], pitch=r[1], tau=r[2]))
for Y in (2022, 2023, 2024):
    lab, ev = fold(d, Y); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy(); head(f"평가 시즌 {Y}")
    T = build_tables(lab, d_all[d_all.season < Y], tm[tm.season < Y], Y); E = eb_tables(lab, alpha=2000); Pe = eb_predict(ev, E)
    sd_e = eb_level_sd(lab, E); print(f"  EB 투수간 점수 sd(학습) = {sd_e:.3f}")
    z = np.load(OUT + f"p_val_noF_{Y}.npz", allow_pickle=True); assert (z["pid"] == pid).all(); Pg = z["P"]
    rec(Y, "EB", score(Pe, pid, y, "EB α=2000")); rec(Y, "GBDT_noF", score(Pg, pid, y, "GBDT noF"))
    for lam in (0.5, 0.75, 1.0):
        rec(Y, f"anchor{lam}", score(anchor(Pg, Pe, lam), pid, y, f"anchor λ={lam}"))
    Pa = anchor(Pg, Pe, 1.0); sub("anchor λ=1 + 주입")
    for nm, sigs in SIGSETS.items():
        zt = inject_z(T["pitcher"], sigs, sd_e); zr = pd.Series(pid).map(zt).fillna(0.0).to_numpy()
        for eps in (0.05, 0.1, 0.2):
            rec(Y, f"anchor+{nm}@{eps}", score(inject(Pa, zr, eps * sd_e), pid, y, f"  +{nm} ε={eps}"))
R = pd.DataFrame(rows).set_index(["m", "Y"]).unstack("Y"); base = R.loc["EB"]
R["gate_vs_EB"] = [("✓" if ((R.loc[m, "tot"] > base["tot"]).all() and (R.loc[m, "tau"] >= base["tau"] - 1e-9).all()) else "✗") for m in R.index]
for k in ("tot", "pitch", "tau"): R[(k, "mean")] = R[k].mean(1)
head("3폴드 요약 (게이트: EB 대비 Total 3/3 ↑ 이면서 τ 3/3 ≥)"); print(R.round(4).to_string())
