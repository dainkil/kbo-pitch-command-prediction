"""투수 단위 모델 — 투수×시즌 행: 이력 피처(season<Y 표) → 그 시즌 실제 평균 제구점수. τ 를 직접 겨냥한다.
2024 폴드에서 (a) 신호 자체의 τ, (b) v2 LGB 확률에 ε 주입했을 때 Total. 주입은 행별 pitcher_id 표 조회 → 규칙 4 충족."""
import sys, os, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
T0 = time.time(); tk = lambda: f"[{time.time()-T0:4.0f}s]"
d_all = load(labeled_only=False); tm = load_tm(); lab_all = d_all[d_all.y.notna()]
rows = []
for Y in range(2020, 2025):
    T = build_tables(lab_all[lab_all.season < Y], d_all[d_all.season < Y], tm[tm.season < Y], Y)
    g = lab_all[lab_all.season == Y].groupby("pitcher_id").pt.agg(["size", "mean"]); g = g[g["size"] >= 50]
    F = T["pitcher"].reindex(g.index).astype(float); F["Y"] = Y; F["target"] = g["mean"]; F["w"] = g["size"]; rows.append(F)
R = pd.concat(rows); R["p_n"] = R.p_n.fillna(0); R["nohist"] = (R.p_n == 0).astype(float)
X = [c for c in R.columns if c not in ("Y", "target", "w")]
tr, ev = R[R.Y < 2024], R[R.Y == 2024]; print(f"{tk()} 투수×시즌 학습 {len(tr)} · 2024 평가 {len(ev)} · 피처 {len(X)}")
def tau(pred): return kendalltau(ev.target.round(10), pd.Series(pred, index=ev.index).round(10), variant="b").statistic
head("(a) 신호 자체의 τ (2024 투수 321명)")
for c in ("p_skill", "p_fa", "p_he", "pr_fa", "p1_fa", "pgtR_fa", "p_rev", "p_out", "tm_mix_fastball"):
    v = ev[c].fillna(ev[c].median()); s = -1 if c in ("p_fa", "pr_fa", "p1_fa", "pgtR_fa", "p_rev", "p_out", "tm_mix_fastball") else 1
    print(f"    {c:16s} τ {tau(s*v):+.4f}")
preds = {}
for nm, prm in (("lgb_reg_small", dict(num_leaves=7, min_data_in_leaf=30, learning_rate=0.03, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, n=500)),
                ("lgb_reg_tiny", dict(num_leaves=4, min_data_in_leaf=50, learning_rate=0.03, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=30, n=600))):
    n = prm.pop("n"); Ps = []
    for s in range(5):
        m = lgb.train({**prm, "objective": "regression", "verbose": -1, "num_threads": 8, "seed": s}, lgb.Dataset(tr[X], tr.target, weight=np.sqrt(tr.w)), n)
        Ps.append(m.predict(ev[X]))
    preds[nm] = np.mean(Ps, 0); print(f"    {nm:16s} τ {tau(preds[nm]):+.4f}")
    imp = pd.Series(m.feature_importance("gain"), index=X); print("      top:", (imp/imp.sum()*100).round(1).sort_values(ascending=False).head(8).to_dict())
# Ridge
from sklearn.linear_model import Ridge
med = tr[X].median(); Xt = tr[X].fillna(med); Xe = ev[X].fillna(med); mu, sd = Xt.mean(), Xt.std().replace(0, 1)
for a in (10, 100, 1000):
    r = Ridge(alpha=a).fit((Xt-mu)/sd, tr.target, sample_weight=np.sqrt(tr.w)); preds[f"ridge{a}"] = r.predict((Xe-mu)/sd); print(f"    ridge α={a:<5d}     τ {tau(preds[f'ridge{a}']):+.4f}")
z = np.load(OUT + "p_val_2024.npz", allow_pickle=True); P, y, pid = z["P"], z["y"], z["pid"]
own = pd.Series(P @ PT).groupby(pid).mean().reindex(ev.index); print(f"    v2 LGB 자체(참고)  τ {tau(own):+.4f}")
head("(b) v2 확률에 ε 주입 → Total (2024)")
score(P, pid, y, "기준 v2 앙상블")
for nm in ("lgb_reg_small", "lgb_reg_tiny", "ridge100"):
    r = pd.Series(preds[nm], index=ev.index); rz = ((r - r.mean())/r.std()); rr = pd.Series(pid).map(rz).fillna(0).to_numpy()
    print(f"  신호 {nm}")
    for eps in (0.005, 0.01, 0.02, 0.03, 0.05):
        Pe = np.clip(P + eps*rr[:, None]*np.array([1, 0, -1]), 1e-4, None); Pe /= Pe.sum(1, keepdims=True); score(Pe, pid, y, f"    ε={eps}")
    # 블렌드: 자체 순위 + 외부 순위 (z 평균) 가 더 나은가
    oz = ((own - own.mean())/own.std()); print(f"    blend τ (자체z+외부z)/2: {tau((oz + rz)/2):+.4f}   자체 {tau(oz):+.4f}  외부 {tau(rz):+.4f}")
print(f"{tk()} done")
