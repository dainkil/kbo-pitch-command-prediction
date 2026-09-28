"""하이퍼파라미터·피처 제거 변형 (v2 피처, seed 0, 2024 폴드) — τ 에 무엇이 좋은가."""
import sys, os, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
T0 = time.time(); tk = lambda: f"[{time.time()-T0:4.0f}s]"
d_all = load(labeled_only=False); tm = load_tm()
Xtr, ytr, Xev, ev, T = build_xy(d_all, tm, [2020, 2021, 2022, 2023], 2024); feats = list(Xtr.columns); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy()
rng = np.random.default_rng(0); msk = rng.random(len(Xtr)) < 0.9
def run(label, prm=None, drop=()):
    cols = [c for c in feats if not any(c.startswith(p) for p in drop)]; cat = [cols.index("base_state")]
    dtr = lgb.Dataset(Xtr[msk][cols], ytr[msk], categorical_feature=cat); dva = lgb.Dataset(Xtr[~msk][cols], ytr[~msk], reference=dtr)
    m = lgb.train({**PARAMS, **(prm or {})}, dtr, 4000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    P = m.predict(Xev[cols].to_numpy(dtype=np.float32), num_iteration=m.best_iteration); score(P, pid, y, f"{tk()} {label} it={m.best_iteration}"); return P
head("하이퍼파라미터")
P0 = run("기준 v2 (63잎, minleaf200, ff0.8)")
run("minleaf 1000", dict(min_data_in_leaf=1000))
run("31잎 ff0.5", dict(num_leaves=31, feature_fraction=0.5))
run("lr 0.02", dict(learning_rate=0.02))
run("l2=10 · minleaf 500", dict(lambda_l2=10, min_data_in_leaf=500))
run("extra_trees", dict(extra_trees=True))
run("127잎 minleaf 100", dict(num_leaves=127, min_data_in_leaf=100))
head("피처 제거")
run("−타자(b_*)", drop=("b_",))
run("−Trackman(tm_*)", drop=("tm_",))
run("−위치이력(p_rev,p_out,p_side,p_h,p_abs,pgt_rev)", drop=("p_rev", "p_out", "p_side", "p_h_", "p_abs", "pgt_rev"))
run("−최근성(pr_*)", drop=("pr_",))
run("−직전시즌(p1_*)", drop=("p1_",))
print(f"{tk()} done")
