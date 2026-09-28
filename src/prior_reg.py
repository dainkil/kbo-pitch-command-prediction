"""E2 — EB prior 를 배치셀(60) 에서 '투수 회귀 prior'(Trackman 프로필 + 과거 기용 히스토그램 + 이력 크기) 로. 3폴드 τ / Total(anchor v3).
회귀 학습행 = (투수, 시즌 s<Y): 피처는 season<s 이력, 타깃 = s 의 등급비율(50투구+). 평가 = season<Y 피처 → Y."""
import sys, os, time
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
T0 = time.time(); tk = lambda: f"[{time.time()-T0:4.0f}s]"
d_all = load(labeled_only=False); tm = load_tm(); d = d_all[d_all.y.notna()]
def tau(a, b): return kendalltau(np.asarray(a).round(10), np.asarray(b).round(10), variant="b").statistic
LI_B, INN_B, SD_B = (0.6, 1.4), (4, 7, 9), (-3, -1, 2, 4)
def deploy_hist(lab):
    """투수별 과거 기용 히스토그램(li3·inn4·sd5 주변분포 12개) + 평균 li + 이닝 + hand."""
    li = np.digitize(lab.li.to_numpy(float), LI_B); inn = np.digitize(lab.inning.to_numpy(float), INN_B); sd = np.digitize(lab.score_diff_pitcher_team.to_numpy(float), SD_B)
    f = pd.DataFrame({"p": lab.pitcher_id.to_numpy()})
    for i in range(3): f[f"li{i}"] = (li == i).astype(float)
    for i in range(4): f[f"inn{i}"] = (inn == i).astype(float)
    for i in range(5): f[f"sd{i}"] = (sd == i).astype(float)
    f["li_mean"] = lab.li.to_numpy(float); f["inn_mean"] = lab.inning.to_numpy(float); f["hand"] = (lab.pitcher_hand.to_numpy(float) == 1).astype(float)
    return f.groupby("p").mean()
def feats(lab, tmY):
    """season<s 이력 → 투수별 피처 (Trackman 프로필 + 기용 히스토그램 + log n)."""
    prof = tm_profile(tmY); dh = deploy_hist(lab); n = lab.groupby("pitcher_id").size().rename("n_lab")
    X = dh.join(prof, how="outer").join(n, how="outer"); X["log_n"] = np.log1p(X.n_lab.fillna(0)); X["log_tm_n"] = np.log1p(X.tm_n.fillna(0))
    return X.drop(columns=["n_lab"])
def targets(lab_s, min_n=50):
    oh = np.eye(3)[lab_s.y.to_numpy().astype(int)]; g = pd.DataFrame(oh, index=lab_s.pitcher_id.to_numpy(), columns=CLASSES).groupby(level=0).agg(["mean", "size"])
    ok = g[(CLASSES[0], "size")] >= min_n; return g.loc[ok, [(c, "mean") for c in CLASSES]].droplevel(1, axis=1)
def eb_rows(ev, E, prior_p=None, mix=1.0, alpha=2000.0):
    """L2 EB + 투수 회귀 prior: prior_row = mix·prior_p(투수) + (1−mix)·prior_cell(행), prior_p 없으면 cell."""
    pr = E["prior"][cells(ev)]
    loc = E["pidx"].get_indexer(pd.Index(ev.pitcher_id.to_numpy())); have = loc >= 0; safe = np.where(have, loc, 0)
    n = np.where(have, E["pn"][safe], 0.0)[:, None]; rate = np.where(have[:, None], E["prate"][safe], 0.0)
    if prior_p is not None:
        pp = prior_p.reindex(ev.pitcher_id.to_numpy()); has = pp.notna().all(axis=1).to_numpy(); ppv = pp.fillna(0).to_numpy()
        pr = np.where(has[:, None], mix * ppv + (1 - mix) * pr, pr)
    p = (rate * n + alpha * pr) / (n + alpha); p = np.clip(p, 1e-6, None); return p / p.sum(1, keepdims=True)
rows = []
for Y in (2022, 2023, 2024):
    lab, ev = fold(d, Y); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy()
    # 회귀 학습행: s = 2020..Y-1
    Xs, Ys = [], []
    for s in range(2020, Y):
        Xf = feats(lab[lab.season < s], tm[tm.season < s]); Tg = targets(lab[lab.season == s]); idx = Tg.index.intersection(Xf.index)
        Xs.append(Xf.loc[idx].assign(_s=s)); Ys.append(Tg.loc[idx])
    Xtr = pd.concat(Xs); Ytr = pd.concat(Ys); Xev = feats(lab, tm[tm.season < Y]); cols = [c for c in Xtr.columns if c != "_s"]
    med = Xtr[cols].median(); mu = Xtr[cols].fillna(med).mean(); sd = Xtr[cols].fillna(med).std().replace(0, 1)
    def prep(X): Z = X[cols].fillna(med); return ((Z - mu) / sd).to_numpy()
    print(f"{tk()} [{Y}] 회귀 학습 {len(Xtr)} 투수시즌 · 피처 {len(cols)} · 평가 투수(이력有) {len(Xev)}", flush=True)
    E = eb_tables(lab, 2000); z = np.load(OUT + f"p_val_v3_{Y}.npz", allow_pickle=True); assert (z["pid"] == pid).all(); Pg = z["P"]
    head(f"{Y}"); r0 = score(anchor(Pg, eb_rows(ev, E), 1.0), pid, y, "기준: 배치셀 prior (=y2)"); rows.append(dict(Y=Y, m="cell", tot=r0[0], pitch=r0[1], tau=r0[2]))
    priors = {}
    for a in (3.0, 30.0, 300.0):
        m = Ridge(alpha=a).fit(prep(Xtr), Ytr.to_numpy()); P = np.clip(m.predict(prep(Xev)), 1e-3, None); P /= P.sum(1, keepdims=True); priors[f"ridge{a:g}"] = pd.DataFrame(P, index=Xev.index, columns=CLASSES)
    Pl = np.column_stack([lgb.train(dict(objective="regression", num_leaves=7, min_data_in_leaf=30, learning_rate=0.03, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=12, seed=0),
                                    lgb.Dataset(Xtr[cols], Ytr[c]), 300).predict(Xev[cols]) for c in CLASSES]); Pl = np.clip(Pl, 1e-3, None); Pl /= Pl.sum(1, keepdims=True)
    priors["lgb"] = pd.DataFrame(Pl, index=Xev.index, columns=CLASSES)
    # prior 자체의 τ (투수 수준, 이력 있는 평가 투수만)
    act = ev.groupby("pitcher_id").pt.agg(["size", "mean"]); act = act[act["size"] >= 50]
    sub("prior 단독 τ (50+ 투수 중 이력 있는 투수)")
    for nm, pp in priors.items():
        j = act.join(pp, how="inner"); print(f"    {nm:10s} n={len(j)} τ {tau(j['mean'], j[CLASSES].to_numpy() @ PT):+.4f}")
    cellp = pd.Series(E["prior"][cells(ev)] @ PT).groupby(pid).mean(); j = act.join(cellp.rename("c"), how="inner"); print(f"    {'cell(행평균)':10s} n={len(j)} τ {tau(j['mean'], j.c):+.4f}")
    sub("EB 에 prior 교체/혼합 → anchor Total (v3 GBDT seed0)")
    for nm, pp in priors.items():
        for mix in (0.5, 1.0):
            for a in (500.0, 2000.0, 4000.0):
                r = score(anchor(Pg, eb_rows(ev, E, pp, mix, a), 1.0), pid, y, f"  {nm} mix={mix} α={a:g}"); rows.append(dict(Y=Y, m=f"{nm}_m{mix}_a{a:g}", tot=r[0], pitch=r[1], tau=r[2]))
R = pd.DataFrame(rows).set_index(["m", "Y"]).unstack("Y"); base = R.loc["cell"]
R["gate"] = [("✓" if ((R.loc[m, "tau"] > base["tau"]).all() and (R.loc[m, "tot"] > base["tot"]).all()) else "✗") for m in R.index]
for k in ("tot", "tau"): R[(k, "mean")] = R[k].mean(1)
head("3폴드 요약 (게이트: τ·Total 3/3 > 배치셀 prior)"); print(R[[("tau", 2022), ("tau", 2023), ("tau", 2024), ("tau", "mean"), ("tot", "mean"), ("gate", "")]].round(4).sort_values(("tau", "mean"), ascending=False).to_string())
print(f"{tk()} done")
