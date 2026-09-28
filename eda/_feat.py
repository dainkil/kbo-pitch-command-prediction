"""피처 빌더 — 확장창(expanding window) 인코딩: 시즌 Y 행의 이력 피처는 season<Y 에서만 계산 (2025 추론과 같은 구조, 누수 없음)."""
from _common import *
import lightgbm as lgb

CTX = ["game_month", "game_dayofweek", "inning", "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
       "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
       "home_win_expectancy", "away_win_expectancy", "li", "pitcher_hand", "batter_hand", "top_bottom", "game_type", "base_state"]
CAT = ["top_bottom", "game_type", "base_state"]

def _enc(past, cur, key, alpha, pri, tag, w=None):
    P, n = entity_probs(past, cur, key, alpha, pri, w)
    f = pd.DataFrame(P, columns=[f"{tag}_{c[:2].lower()}" for c in CLASSES], index=cur.index); f[f"{tag}_n"] = n
    return f

def _resid_skill(past, cur, alpha=200):
    """종민 06b/07 의 '상황보정 잔차 EB' 실력 (τ 최고). cnt×game_type 평균을 걷어낸 pt 잔차의 투수 EB(0 수축)."""
    ctab = past.groupby(["cnt", "game_type"]).pt.mean().rename("c").reset_index()
    p = past[["pitcher_id", "cnt", "game_type", "pt"]].merge(ctab, on=["cnt", "game_type"], how="left")
    r = (p.pt - p.c); g = pd.DataFrame({"pid": p.pitcher_id.to_numpy(), "r": r.to_numpy()}).groupby("pid").r.agg(["sum", "size"])
    sk = (g["sum"]/(g["size"] + alpha))
    return cur.pitcher_id.map(sk).fillna(0.0).to_numpy()

def build(d, seasons_train, Y_eval, alpha_p=200, alpha_b=300, recency_h=None, extra=None):
    """반환: (X_train, y_train, X_eval, ev_df). extra(past, cur)->DataFrame 로 피처 추가 가능."""
    parts = {}
    for Y in sorted(set(seasons_train) | {Y_eval}):
        past, cur = fold(d, Y); pri = prior(past) if len(past) else np.array([1/3]*3)
        x = cur[CTX].copy()
        for c in CAT: x[c] = x[c].astype("category")
        if len(past):
            x = pd.concat([x, _enc(past, cur, "pitcher_id", alpha_p, pri, "p"), _enc(past, cur, "batter_id", alpha_b, pri, "b")], axis=1)
            if recency_h:
                w = 0.5**((Y - past.season).to_numpy(float)/recency_h)
                x = pd.concat([x, _enc(past, cur, "pitcher_id", alpha_p, pri, "pr", w)], axis=1)
            prev = past[past.season == Y-1]
            x = pd.concat([x, _enc(prev, cur, "pitcher_id", alpha_p, pri, "p1")], axis=1)
            x["p_skill"] = _resid_skill(past, cur)
            if extra is not None: x = pd.concat([x, extra(past, cur)], axis=1)
        parts[Y] = x
    cats = {c: parts[Y_eval][c].cat.categories for c in CAT}
    Xtr = pd.concat([parts[Y] for Y in seasons_train]); ytr = d.loc[Xtr.index, "y"].to_numpy()
    Xev = parts[Y_eval]
    for c in CAT:
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats[c]); Xev[c] = pd.Categorical(Xev[c].astype(str), categories=cats[c])
    return Xtr, ytr, Xev, d.loc[Xev.index]

PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=40, verbose=-1, seed=0)

def fit_lgb(Xtr, ytr, params=None, rounds=3000, es=50, seed=0):
    params = {**PARAMS, **(params or {})}
    rng = np.random.default_rng(seed); msk = rng.random(len(Xtr)) < 0.9
    dtr = lgb.Dataset(Xtr[msk], ytr[msk]); dva = lgb.Dataset(Xtr[~msk], ytr[~msk])
    m = lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(es, verbose=False)])
    return m

def importance(m, cols, top=15):
    imp = pd.Series(m.feature_importance("gain"), index=cols); imp = (imp/imp.sum()*100).sort_values(ascending=False)
    return imp.round(1).head(top).to_dict()
