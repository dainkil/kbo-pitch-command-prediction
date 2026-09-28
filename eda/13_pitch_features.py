"""EDA 13 — Brier(70%) 를 움직이는 사전 정보는 무엇인가. 각 피처(조합)의 표 모형으로 2024 를 예측해 ΔPitch 를 잰다.
투수 정보 0 인 '상황' 축만 본다 (종민 05/06 은 이 축이 τ 에 역상관임을 보였다 — 여기서는 Brier 관점).
마지막에 LightGBM(상황 열 전부, ID 없음) 으로 '상황 정보의 천장' 을 잰다."""
from _common import *
import lightgbm as lgb
d = load()
SINGLE = ["cnt", "balls_before", "strikes_before", "game_type", "inn", "outs_before", "base_state", "num_runners_on", "li3", "sd5", "hw5",
          "matchup", "pitcher_hand", "batter_hand", "top_bottom", "game_month", "game_dayofweek", "pitcher_team_id", "batter_team_id"]
PAIRS = [["cnt", "game_type"], ["cnt", "matchup"], ["cnt", "inn"], ["cnt", "base_state"], ["cnt", "outs_before"], ["cnt", "li3"], ["cnt", "sd5"],
         ["cnt", "pitcher_hand"], ["game_type", "inn"], ["cnt", "game_month"], ["cnt", "game_type", "matchup"], ["cnt", "game_type", "inn"],
         ["cnt", "game_type", "matchup", "inn", "base_state"]]
res = []
for Y in (2022, 2023, 2024):
    past, ev = fold(d, Y); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy(); pri = prior(past); n = len(ev)
    base = pitch_score(const(pri, n), y)
    for keys in [[k] for k in SINGLE] + PAIRS:
        P = table_probs(past, ev, keys, 50, pri); tot, ps, tau = score(P, pid, y, quiet=True)
        res.append(dict(Y=Y, feat="×".join(keys), dPitch=ps-base, tau=tau, Total=tot))
R = pd.DataFrame(res).groupby("feat").agg(dPitch=("dPitch", "mean"), dPitch_min=("dPitch", "min"), tau=("tau", "mean"), Total=("Total", "mean")).sort_values("dPitch", ascending=False)
head("상황 피처별 ΔPitch (상수 prior 대비, 3폴드 평균 · min) — Brier 관점의 피처 중요도.  τ 는 그 표만으로 투수를 줄세웠을 때")
print(R.round(4).to_string())

head("LightGBM — 상황 열 전부 (ID·이력 없음): 상황 정보만의 Pitch 천장  (2024 폴드)")
past, ev = fold(d, 2024); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy(); pri = prior(past)
CTX = ["season", "game_month", "game_dayofweek", "inning", "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
       "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
       "home_win_expectancy", "away_win_expectancy", "li", "pitcher_hand", "batter_hand", "top_bottom", "game_type", "base_state"]
def X(df):
    x = df[CTX].copy()
    for c in ("top_bottom", "game_type", "base_state"): x[c] = x[c].astype("category")
    return x
ps_ = X(past); ev_ = X(ev)
for c in ("top_bottom", "game_type", "base_state"): ev_[c] = pd.Categorical(ev_[c], categories=ps_[c].cat.categories)
params = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=40, verbose=-1, seed=0)
rng = np.random.default_rng(0); msk = rng.random(len(ps_)) < 0.9
dtr = lgb.Dataset(ps_[msk], past.y.to_numpy()[msk]); dva = lgb.Dataset(ps_[~msk], past.y.to_numpy()[~msk])
m = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
P = m.predict(ev_, num_iteration=m.best_iteration)
print(f"  best_iter {m.best_iteration}"); base = pitch_score(const(pri, len(ev)), y)
tot, ps, tau = score(P, pid, y, "LGB 상황 전부 (season 포함)")
print(f"  ΔPitch vs prior {ps-base:+.4f}")
imp = pd.Series(m.feature_importance("gain"), index=CTX).sort_values(ascending=False); print("  gain 상위:", (imp/imp.sum()*100).round(1).head(12).to_dict())
sub("season 열 제거 (2025 는 학습 범위 밖 → 외삽 위험 확인)")
CTX2 = [c for c in CTX if c != "season"]
dtr = lgb.Dataset(ps_[msk][CTX2], past.y.to_numpy()[msk]); dva = lgb.Dataset(ps_[~msk][CTX2], past.y.to_numpy()[~msk])
m2 = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
P2 = m2.predict(ev_[CTX2], num_iteration=m2.best_iteration); score(P2, pid, y, "LGB 상황 전부 (season 제외)")
print(f"\n{tick()} done")
