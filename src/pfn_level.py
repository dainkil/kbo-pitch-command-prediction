"""E3 — TabPFN v2 투수×시즌 회귀를 EB 수준과 z-평균. 학습행 = (투수, s<Y) 피처(season<s 표) → 타깃 s 평균 pt; 평가 = season<Y 표 → Y.
게이트: 단독 τ ≥ .27 이고 EB⊕PFN z-평균이 3/3 > EB. Total 은 anchor(v3 GBDT seed0) 로."""
import sys, os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
from tabpfn import TabPFNRegressor
T0 = time.time(); tk = lambda: f"[{time.time()-T0:4.0f}s]"
d_all = load(labeled_only=False); tm = load_tm(); d = d_all[d_all.y.notna()]
def tau(a, b): return kendalltau(np.asarray(a).round(10), np.asarray(b).round(10), variant="b").statistic
FEATS = ["p_sh", "p_he", "p_fa", "p_n", "p1_sh", "p1_he", "p1_fa", "p1_n", "pr_fa", "p_skill", "pgtR_fa", "pgtR_sh", "pgtR_rev", "p_rev", "p_out", "p_h_sd", "p_side_sd", "p_abs_side",
         "mr_far50", "mr_chase", "mr_far30", "mr_outpos", "sd_sy", "sd_sx", "tm_mix_fastball", "tm_mix_offspeed", "tm_mix_breaking", "tm_rel_speed", "tm_spin_rate",
         "tm_induced_vert_break", "tm_horz_break", "tm_extension", "tm_rel_height", "tm_sd_rel_height", "tm_sd_rel_side", "tm_n"]
def pitcher_rows(lab_hist, allrows_hist, tm_hist, s, lab_target=None):
    """season<s 표 → 투수 피처. lab_target 주면 그 시즌 50투구+ 투수의 평균 pt 를 타깃으로 붙임."""
    T = build_tables(lab_hist, allrows_hist, tm_hist, s)["pitcher"]; X = T.reindex(columns=FEATS).astype(float)
    if lab_target is None: return X
    g = lab_target.groupby("pitcher_id").pt.agg(["size", "mean"]); g = g[g["size"] >= 50]; idx = g.index.intersection(X.index)
    return X.loc[idx], g.loc[idx, "mean"], g.loc[idx, "size"]
rows = []
for Y in (2022, 2023, 2024):
    lab, ev = fold(d, Y); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy()
    Xs, ys, ws = [], [], []
    for s in range(2020, Y):
        X, t, n = pitcher_rows(lab[lab.season < s], d_all[d_all.season < s], tm[tm.season < s], s, lab[lab.season == s]); Xs.append(X); ys.append(t); ws.append(n)
    Xtr = pd.concat(Xs); ytr = pd.concat(ys).to_numpy(); Xev = pitcher_rows(lab, d_all[d_all.season < Y], tm[tm.season < Y], Y)
    E = eb_tables(lab, 2000); Pe = eb_predict(ev, E); z = np.load(OUT + f"p_val_v3_{Y}.npz", allow_pickle=True); assert (z["pid"] == pid).all(); Pg = z["P"]
    g = pd.DataFrame({"pid": pid, "act": ev.pt.to_numpy(), "eb": Pe @ PT}).groupby("pid").agg(n=("act", "size"), act=("act", "mean"), eb=("eb", "mean")); g = g[g.n >= 50]
    print(f"{tk()} [{Y}] 학습 투수시즌 {len(Xtr)} · 평가 50+ {len(g)} (이력有 {g.index.isin(Xev.index).sum()})", flush=True)
    head(f"{Y}  EB τ {tau(g.act, g.eb):+.4f}")
    preds = {}
    Tp = build_tables(lab, d_all[d_all.season < Y], tm[tm.season < Y], Y)["pitcher"]
    for nm, kw in (("pfn", dict()),):
        m = TabPFNRegressor(device="cuda", random_state=0, **kw); m.fit(Xtr.to_numpy(), ytr)
        p = pd.Series(m.predict(Xev.to_numpy()), index=Xev.index); preds[nm] = p
        j = g.join(p.rename("pfn"), how="inner"); print(f"  {nm:9s} 단독 τ(이력有 {len(j)}) {tau(j.act, j.pfn):+.4f}   vs EB 같은 집합 {tau(j.act, j.eb):+.4f}", flush=True)
    # 결합: z(EB) + w·z(PFN) (PFN 없는 신인 = 0) → 행 확률로 환산: EB 행 점수에 (w·sd_e·z) 주입 (S↔F 축)
    sd_e = eb_level_sd(lab, E)
    r0 = score(anchor(Pg, Pe, 1.0), pid, y, "기준 anchor(EB)"); rows.append(dict(Y=Y, m="EB", tot=r0[0], tau=r0[2]))
    for nm, p in preds.items():
        ref = Xtr.index; pz = (p - ytr.mean()) / ytr.std()           # 학습 타깃 분포로 표준화(행 독립)
        pz_full = pz.reindex(Tp.index).fillna(pz.median()); pzn = neutralize(pz_full, Tp)   # 중립화: (과거F비중×이력) 셀 평균 0
        # 이력 있는 투수 안에서 EB 와의 잔차만 주입하는 변형: z(PFN) − z(EB수준) (EB 가 이미 아는 부분 제거)
        eb_lvl = pd.Series(eb_predict(lab, E) @ PT).groupby(lab.pitcher_id.to_numpy()).mean(); ez = ((eb_lvl - eb_lvl.mean()) / eb_lvl.std()).reindex(Tp.index).fillna(0)
        resid = neutralize((pz_full - ez), Tp)
        for lab_, zz in (("원본", pz_full), ("중립", pzn), ("중립잔차", resid)):
            zr = pd.Series(pid).map(zz).fillna(0.0).to_numpy()
            for w in (0.3, 0.5, 1.0):
                r = score(inject(anchor(Pg, Pe, 1.0), zr, w * sd_e), pid, y, f"  EB ⊕ {nm} {lab_} w={w}"); rows.append(dict(Y=Y, m=f"{nm}_{lab_}_w{w}", tot=r[0], tau=r[2]))
R = pd.DataFrame(rows).set_index(["m", "Y"]).unstack("Y"); base = R.loc["EB"]
R["gate"] = [("✓" if ((R.loc[m, "tau"] > base["tau"]).all() and (R.loc[m, "tot"] > base["tot"]).all()) else "✗") for m in R.index]
for k in ("tot", "tau"): R[(k, "mean")] = R[k].mean(1)
head("3폴드 요약"); print(R.round(4).sort_values(("tau", "mean"), ascending=False).to_string()); print(f"{tk()} done")
