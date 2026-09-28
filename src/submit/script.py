# -*- coding: utf-8 -*-
"""추론 스크립트 — 평가 서버에서 자동 실행. data/test.csv 의 각 행을 독립적으로 예측한다.
이력 피처는 학습 데이터(2019~2024)로 미리 만든 model/tables/*.csv 를 행 단위로 join 하여 만든다 (평가 행 간 집계 없음).
pandas 2.0.3 / numpy 1.26.4 / lightgbm 4.7.0 호환."""
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ID = "row_id"
PROB_COLS = ["prob_shadow", "prob_heart", "prob_failure"]
CTX = ["game_month", "game_dayofweek", "inning", "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
       "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
       "home_win_expectancy", "away_win_expectancy", "li", "pitcher_hand", "batter_hand", "top_bottom", "game_type", "base_state"]
BASE_STATES = ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]
EB = ["sh", "he", "fa"]
# ── L2 형 EB 성분 (종민 submit_L2 설계): 투수 과거 등급비율을 배치셀 prior 로 수축. 고정 경계 → test 분포 무관.
LI_BINS = (0.6, 1.4); INN_BINS = (4, 7, 9); SD_BINS = (-3, -1, 2, 4); NCELL = 3 * 4 * 5


def cells(df):
    li = pd.to_numeric(df["li"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    inn = pd.to_numeric(df["inning"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    sd = pd.to_numeric(df["score_diff_pitcher_team"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return (np.digitize(li, LI_BINS) * 4 + np.digitize(inn, INN_BINS)) * 5 + np.digitize(sd, SD_BINS)


def eb_predict(rows, E, T=None, S=None):
    """p_row = (n·rate_투수 + α·prior_row) / (n + α). 신인(n=0)은 순수 prior.
    prior_row = 배치셀 prior (형태). S(skill prior) 가 있으면 그 수준(제구점수)을 q_skill(row) 로 갈아끼운다 — S↔F 축 이동 (apply_adj 와 동일).
    이력 큰 투수는 α/(n+α) 가 작아 prior 영향이 자동 소멸 → 소이력·신인 투수의 순서만 바뀐다."""
    pr = E["prior"][cells(rows)]
    if S is not None and T is not None:
        mu = S["booster"].predict(skill_X(rows, T, S["features"]).to_numpy(dtype=np.float32))
        cs = 100.0 * pr[:, 0] + 30.0 * pr[:, 1]
        d = np.clip(mu if S.get("resid") else mu - cs, -S["cap"], S["cap"]) * S["w"]   # resid: μ 가 셀 prior 대비 잔차(점) 로 학습됨
    loc = E["pidx"].get_indexer(pd.Index(rows["pitcher_id"].astype(str).to_numpy())); have = loc >= 0; safe = np.where(have, loc, 0)
    n = np.where(have, E["pn"][safe], 0.0).reshape(-1, 1); rate = np.where(have.reshape(-1, 1), E["prate"][safe], 0.0)
    if S is not None and T is not None:
        if S.get("gate", 0): d = d * np.minimum(1.0, n[:, 0] / float(S["gate"]))   # 이력 n<gate 인 투수(신인)는 셀 prior 를 그대로 (신인은 셀 prior 가 더 낫다: 2024 τ .355)
        pr = _norm(pr + (d / 100.0)[:, None] * SF)
    return _norm((rate * n + E["alpha"] * pr) / (n + E["alpha"]))


# ── d10: 학습된 행-문맥 prior (skill prior) — "이 행의 문맥·상대·구종 구성이 말하는 투수 수준" (beta-binomial regression prior 의 GBDT 판)
SKILL_ROW_A = ["game_type", "li", "inning", "score_diff_pitcher_team", "top_bottom", "home_win_expectancy", "game_month", "pitcher_hand", "batter_hand"]
SKILL_ROW_B = SKILL_ROW_A + ["balls_before", "strikes_before", "outs_before", "num_runners_on", "run_total_before"]   # 기용 B군 (행 수준 관계 ≈0, 투수 수준 증거)
SKILL_P = ["p_n_log", "p_n_all_log", "p_fshare", "pk_mix_1", "pk_mix_2", "pk_mix_4", "pk_mix_5", "pk_mix_7", "pk_mix_9", "pk_mix_10",
           "tm_mix_fastball", "tm_mix_breaking", "tm_mix_offspeed", "tm_fb_rel_speed", "tm_rel_speed", "tm_n_log"]
SKILL_B = ["b_fshare", "b_n_all_log", "b_sh", "b_he", "b_fa", "b_n_log"]
REGIME_FEATS = ["game_type", "post23", "p_fshare"]   # F 라벨 체제(2023 반전)에 노출되는 피처 — variant C 는 제외
def skill_feats(variant):
    f = (SKILL_ROW_B if variant == "B" else SKILL_ROW_A) + ["post23"] + SKILL_P + SKILL_B
    return [c for c in f if c not in REGIME_FEATS] if variant == "C" else f


def skill_X(rows, T, feats):
    """행 자기 열 + pitcher_id/batter_id 표 조회만 (test 행 간 집계 없음). 결측은 NaN 그대로(LightGBM 분기)."""
    x = pd.DataFrame(index=rows.index)
    for c in feats:
        if c in ("post23", "p_n_log", "p_n_all_log", "tm_n_log", "b_n_all_log", "b_n_log") or c.startswith(("p_", "pk_", "tm_", "b_")): continue
        if c == "top_bottom":  x[c] = rows[c].map({"T": 0, "B": 1}).astype(float)
        elif c == "game_type": x[c] = rows[c].map({"R": 0, "F": 1}).astype(float)
        else:                  x[c] = pd.to_numeric(rows[c], errors="coerce").astype(float)
    x["post23"] = (pd.to_numeric(rows["season"], errors="coerce").astype(float) >= 2023).astype(float)
    P = T["pitcher"]; B = T["batter"]
    jp = pd.DataFrame({"pitcher_id": rows["pitcher_id"].astype(str).to_numpy()}).merge(P, left_on="pitcher_id", right_index=True, how="left"); jp.index = rows.index
    jb = pd.DataFrame({"batter_id": rows["batter_id"].astype(str).to_numpy()}).merge(B, left_on="batter_id", right_index=True, how="left"); jb.index = rows.index
    x["p_n_log"] = np.log1p(jp["p_n"].fillna(0.0).astype(float)); x["p_n_all_log"] = np.log1p(jp["p_n_all"].fillna(0.0).astype(float))
    x["tm_n_log"] = np.log1p(jp["tm_n"].fillna(0.0).astype(float))
    x["b_n_all_log"] = np.log1p(jb["b_n_all"].fillna(0.0).astype(float)); x["b_n_log"] = np.log1p(jb["b_n"].fillna(0.0).astype(float))
    for c in feats:
        if c in x.columns: continue
        x[c] = (jb[c] if c.startswith("b_") else jp[c]).astype(float)
    return x[feats]


def load_skill(mdir):
    f = mdir / "skill_meta.json"
    if not f.exists(): return None
    S = json.load(open(f, encoding="utf-8"))
    if not S.get("w", 1.0): return None
    S["booster"] = lgb.Booster(model_file=str(mdir / S["model"])); S["cap"] = float(S.get("cap", 15.0)); S["w"] = float(S.get("w", 1.0)); S["gate"] = float(S.get("gate", 0.0))
    return S


def _norm(P):
    """행 정규화. 합을 (c0+c1)+c2 고정 순서로 계산해 배치 크기(벡터화 경로)와 무관하게 비트 동일한 결과를 낸다."""
    P = np.clip(P, 1e-6, None); s = (P[:, 0] + P[:, 1]) + P[:, 2]
    return P / s[:, None]


SF = np.array([1.0, 0.0, -1.0])   # Shadow↔Failure 축: 점수 100/단위 — Brier 비용이 가장 싼 이동 방향


def anchor(Pg, Pe, lam=1.0):
    """행별 제구점수(100·pS+30·pH)를 EB 의 값에 고정하고 등급 배분 형태만 GBDT 에서 가져온다 → 투수 순위(τ)는 EB 와 동일, Brier 만 개선."""
    delta = ((100.0 * Pe[:, 0] + 30.0 * Pe[:, 1]) - (100.0 * Pg[:, 0] + 30.0 * Pg[:, 1])) / 100.0   # 원소 연산(BLAS 미사용) → 배치 크기 무관 비트 동일
    return _norm(Pg + lam * delta[:, None] * SF)


def inject(P, z_rows, eps_pts):
    """투수별 순위 신호 z(학습표 조회, 신인=0) 를 eps_pts 점/σ 만큼 S↔F 축으로 주입."""
    return _norm(P + (eps_pts / 100.0) * z_rows[:, None] * SF)


def load_adj(mdir):
    """평가면 측 환경 보정 상수 (종민 L25 재현, LB 69.431): F 레벨 γ·lvl_diff·[F행] + 팀 γ·team_dev[투수팀]. adj.json 없으면 None."""
    f = mdir / "adj.json"
    if not f.exists(): return None
    A = json.load(open(f, encoding="utf-8"))
    A["team_index"] = pd.Index(np.asarray(A.get("team_idx", []), dtype=object)); A["team_dev_arr"] = np.asarray(A.get("team_dev", []), dtype=float)
    return A


def apply_adj(P, rows, A, E=None):
    """행별 목표 제구점수를 Δt 점만큼 S↔F 축으로 이동. 학습 상수 + 자기 행의 game_type·pitcher_team_id (+ 투수 이력 수 표) 만 사용 → 행 독립."""
    dt = np.zeros(len(rows), dtype=float)
    hd = A.get("hist_delta")   # d17: 이력 구간 수준 보정 (마진 투수 선택 효과, R 행 LOSO 회귀 계수) — n 은 EB 표(학습 라벨 행 수) 조회
    if hd and E is not None:
        loc = E["pidx"].get_indexer(pd.Index(rows["pitcher_id"].astype(str).to_numpy())); n = np.where(loc >= 0, E["pn"][np.where(loc >= 0, loc, 0)], 0.0)
        dt = dt + float(hd.get("d0", 0)) * (n == 0) + float(hd.get("d1", 0)) * ((n > 0) & (n <= 300)) + float(hd.get("d2", 0)) * ((n > 300) & (n <= 1000))
    lg = float(A.get("lvl_g", 0.0) or 0.0)
    if lg: dt = dt + lg * float(A["lvl_diff"]) * (rows["game_type"].astype(str).to_numpy() == "F").astype(float)
    tg = float(A.get("lvl_team_g", 0.0) or 0.0)
    if tg and len(A["team_dev_arr"]):
        loc = A["team_index"].get_indexer(pd.Index(rows["pitcher_team_id"].astype(str).to_numpy()))
        dt = dt + tg * np.where(loc >= 0, A["team_dev_arr"][np.where(loc >= 0, loc, 0)], 0.0)
    return _norm(P + (dt / 100.0)[:, None] * SF)


def load_eb(mdir):
    A = np.load(mdir / "eb_tables.npz", allow_pickle=False); meta = json.load(open(mdir / "eb_meta.json", encoding="utf-8"))
    return {"prior": A["prior"].astype(float), "prate": A["prate"].astype(float), "pn": A["pn"].astype(float),
            "pidx": pd.Index(A["pid"].astype(object)), "alpha": float(meta["alpha"])}


def apply_tables(rows, T):
    """rows: 사전 정보 DataFrame (train/test 29열). T: build_tables() 결과 (pitcher/batter 표 + meta). 반환: 피처 DataFrame (meta['features'] 순서)."""
    meta = T["meta"]; pri = np.asarray(meta["prior"], dtype=float)
    x = pd.DataFrame(index=rows.index)
    for c in CTX:
        if c == "top_bottom":   x[c] = rows[c].map({"T": 0, "B": 1}).astype(float)
        elif c == "game_type":  x[c] = rows[c].map({"R": 0, "F": 1}).astype(float)
        elif c == "base_state": x[c] = rows[c].map({s: i for i, s in enumerate(BASE_STATES)}).astype(float)
        else:                   x[c] = pd.to_numeric(rows[c], errors="coerce").astype(float)
    x["season"] = pd.to_numeric(rows["season"], errors="coerce").astype(float)

    P = T["pitcher"]; B = T["batter"]
    jp = pd.DataFrame({"pitcher_id": rows["pitcher_id"].astype(str).to_numpy()}).merge(P, left_on="pitcher_id", right_index=True, how="left")
    jb = pd.DataFrame({"batter_id": rows["batter_id"].astype(str).to_numpy()}).merge(B, left_on="batter_id", right_index=True, how="left")
    jp.index = rows.index; jb.index = rows.index
    # 투수 EB 3종 (전체 / 직전시즌 / 최근성) + 타자 EB: 미등장 → prior, n=0
    for tag, j in (("p", jp), ("p1", jp), ("pr", jp), ("b", jb)):
        for i, c in enumerate(EB): x[f"{tag}_{c}"] = j[f"{tag}_{c}"].fillna(pri[i]).astype(float)
        x[f"{tag}_n"] = j[f"{tag}_n"].fillna(0.0).astype(float)
    x["p_skill"] = jp["p_skill"].fillna(0.0).astype(float)
    # 투수×game_type EB: 그 행의 game_type 에 해당하는 표를 선택
    isF = (x["game_type"] == 1).to_numpy()
    for i, c in enumerate(EB):
        v = np.where(isF, jp[f"pgtF_{c}"].to_numpy(dtype=float), jp[f"pgtR_{c}"].to_numpy(dtype=float))
        x[f"pgt_{c}"] = np.where(np.isnan(v), pri[i], v)
    x["p_fshare"] = jp["p_fshare"].astype(float)          # 결측 허용(NaN) — 이력 없음
    x["p_n_all"] = jp["p_n_all"].fillna(0.0).astype(float)
    for c in meta["tm_cols"]: x[c] = jp[c].astype(float)   # Trackman 프로필: 결측 허용
    for c in meta.get("extra_cols", []): x[c] = jp[c].astype(float)   # v2: 위치·반대투구 이력, 결측 허용
    if "pgt_rev" in meta["features"]:
        x["pgt_rev"] = np.where(isF, jp["pgtF_rev"].to_numpy(dtype=float), jp["pgtR_rev"].to_numpy(dtype=float))
    return x[meta["features"]]


def load_tables(mdir):
    T = {"meta": json.load(open(mdir / "meta.json", encoding="utf-8"))}
    T["pitcher"] = pd.read_csv(mdir / "tables" / "pitcher.csv", dtype={"pitcher_id": str}).set_index("pitcher_id")
    T["batter"] = pd.read_csv(mdir / "tables" / "batter.csv", dtype={"batter_id": str}).set_index("batter_id")
    return T


def predict_ensemble(mdir, X, meta):
    P = np.zeros((len(X), 3)); k = 0
    for f in meta["models"]:
        m = lgb.Booster(model_file=str(mdir / f)); P += m.predict(X.to_numpy(dtype=np.float32)); k += 1
    return _norm(P / k)


def predict_all(test, mdir, T):
    """GBDT 확률 → (anchor|blend) EB 결합 → 순위 신호 주입. 모든 단계가 행 독립."""
    meta = T["meta"]; X = apply_tables(test, T)
    P = predict_ensemble(mdir, X, meta)      # 열 순서: Shadow, Heart, Failure (학습 시 y=0,1,2)
    mode = meta.get("mode", "gbdt")
    if mode in ("anchor", "blend"):
        Pe = eb_predict(test, load_eb(mdir), T, load_skill(mdir))   # d10+: skill prior (skill_meta.json 있을 때만)
        ck = float(meta.get("center_k", 0.0) or 0.0)
        if ck and "s_hat" in T["pitcher"].columns:
            # 투수-중심 anchor: GBDT 행 점수의 '투수 평균(학습행 표) 대비 편차' 를 κ 만큼 목표에 되살린다 → 행 수준 정보 보존, 투수 평균은 EB 근처. 표 없는 투수(신인)는 0.
            sh = pd.DataFrame({"pitcher_id": test["pitcher_id"].astype(str).to_numpy()}).merge(T["pitcher"][["s_hat"]], left_on="pitcher_id", right_index=True, how="left")["s_hat"].to_numpy(dtype=float)
            dev = np.where(np.isnan(sh), 0.0, (100.0 * P[:, 0] + 30.0 * P[:, 1]) - sh)
            Pe = _norm(Pe + (ck * dev / 100.0)[:, None] * SF)
        if mode == "anchor": P = anchor(P, Pe, float(meta.get("lam", 1.0)))
        else: w = float(meta["eb_weight"]); P = _norm((1.0 - w) * P + w * Pe)
    if meta.get("inject_eps", 0) and "inj_z" in T["pitcher"].columns:
        z = pd.DataFrame({"pitcher_id": test["pitcher_id"].astype(str).to_numpy()}).merge(T["pitcher"][["inj_z"]], left_on="pitcher_id", right_index=True, how="left")["inj_z"].fillna(0.0).to_numpy(dtype=float)
        P = inject(P, z, float(meta["inject_eps"]))
    A = load_adj(mdir)                       # d9+: 환경 보정 (adj.json 있을 때만)
    if A is not None: P = apply_adj(P, test, A, load_eb(mdir) if A.get("hist_delta") else None)
    return P


def main():
    t0 = time.time(); root = Path("./"); mdir = root / "model"
    test = pd.read_csv(root / "data" / "test.csv", dtype={ID: str})
    sample = pd.read_csv(root / "data" / "sample_submission.csv", dtype={ID: str})
    if not test[ID].reset_index(drop=True).equals(sample[ID].reset_index(drop=True)):
        raise ValueError("test.csv 와 sample_submission.csv 의 row_id 순서가 다릅니다.")
    T = load_tables(mdir)
    P = predict_all(test, mdir, T)
    if not np.isfinite(P).all(): raise ValueError("비정상 확률")
    if not np.allclose(P.sum(1), 1.0, atol=1e-5): raise ValueError("확률 합 != 1")
    sub = sample.copy()
    for i, c in enumerate(PROB_COLS): sub[c] = P[:, i]
    (root / "output").mkdir(exist_ok=True)
    sub.to_csv(root / "output" / "submission.csv", index=False)
    print(f"saved output/submission.csv ({len(sub):,} rows) in {time.time()-t0:.1f}s · mean probs {P.mean(0).round(4)}")


if __name__ == "__main__":
    main()
