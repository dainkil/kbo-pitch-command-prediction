"""E3 배포용 — TabPFN v2 투수 단위 예측을 '표' 로 만든다 (추론 서버에 TabPFN 불필요). 사용: pfn_signal.py <U> [<U> ...]  (U=2025: 제출용, 2024: 배포동일 검산용)
출력 out/pfn_z_<U>.csv: pitcher_id, z  — z = neutralize(z(PFN) − z(EB 수준)), 학습 = (투수, s<U) 피처(season<s) → s 평균 pt."""
import sys, os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
from tabpfn import TabPFNRegressor
FEATS = ["p_sh", "p_he", "p_fa", "p_n", "p1_sh", "p1_he", "p1_fa", "p1_n", "pr_fa", "p_skill", "pgtR_fa", "pgtR_sh", "pgtR_rev", "p_rev", "p_out", "p_h_sd", "p_side_sd", "p_abs_side",
         "mr_far50", "mr_chase", "mr_far30", "mr_outpos", "sd_sy", "sd_sx", "tm_mix_fastball", "tm_mix_offspeed", "tm_mix_breaking", "tm_rel_speed", "tm_spin_rate",
         "tm_induced_vert_break", "tm_horz_break", "tm_extension", "tm_rel_height", "tm_sd_rel_height", "tm_sd_rel_side", "tm_n"]
d_all = load(labeled_only=False); tm = load_tm(); d = d_all[d_all.y.notna()]
def pitcher_rows(lab_hist, allrows_hist, tm_hist, s, lab_target=None):
    T = build_tables(lab_hist, allrows_hist, tm_hist, s)["pitcher"]; X = T.reindex(columns=FEATS).astype(float)
    if lab_target is None: return X, T
    g = lab_target.groupby("pitcher_id").pt.agg(["size", "mean"]); g = g[g["size"] >= 50]; idx = g.index.intersection(X.index)
    return X.loc[idx], g.loc[idx, "mean"]
for U in [int(a) for a in sys.argv[1:]] or [2025]:
    lab = d[d.season < U]; Xs, ys = [], []
    for s in range(2020, U):
        X, t = pitcher_rows(lab[lab.season < s], d_all[d_all.season < s], tm[tm.season < s], s, lab[lab.season == s]); Xs.append(X); ys.append(t)
    Xtr = pd.concat(Xs); ytr = pd.concat(ys).to_numpy(); Xev, Tp = pitcher_rows(lab, d_all[d_all.season < U], tm[tm.season < U], U)
    m = TabPFNRegressor(device="cuda", random_state=0); m.fit(Xtr.to_numpy(), ytr); p = pd.Series(m.predict(Xev.to_numpy()), index=Xev.index)
    pz = ((p - ytr.mean()) / ytr.std()).reindex(Tp.index); pz = pz.fillna(pz.median())
    E = eb_tables(lab, 2000); eb_lvl = pd.Series(eb_predict(lab, E) @ PT).groupby(lab.pitcher_id.to_numpy()).mean(); ez = ((eb_lvl - eb_lvl.mean()) / eb_lvl.std()).reindex(Tp.index).fillna(0)
    z = neutralize(pz - ez, Tp); z.rename("z").to_csv(OUT + f"pfn_z_{U}.csv")
    print(f"U={U}: 학습 투수시즌 {len(Xtr)} · 표 투수 {len(z)} · z sd {z.std():.3f} → {OUT}pfn_z_{U}.csv", flush=True)
