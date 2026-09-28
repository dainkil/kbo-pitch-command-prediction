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


def eb_predict(rows, E):
    """p_row = (n·rate_투수 + α·prior[cell_row]) / (n + α). 신인(n=0)은 순수 배치 prior."""
    pr = E["prior"][cells(rows)]
    loc = E["pidx"].get_indexer(pd.Index(rows["pitcher_id"].astype(str).to_numpy())); have = loc >= 0; safe = np.where(have, loc, 0)
    n = np.where(have, E["pn"][safe], 0.0).reshape(-1, 1); rate = np.where(have.reshape(-1, 1), E["prate"][safe], 0.0)
    return _norm((rate * n + E["alpha"] * pr) / (n + E["alpha"]))


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
    A["hist_index"] = pd.Index(np.asarray(A.get("hist_idx", []), dtype=object)); A["hist_z_arr"] = np.asarray(A.get("hist_z", []), dtype=float)
    return A


def apply_adj(P, rows, A):
    """행별 목표 제구점수를 Δt 점만큼 S↔F 축으로 이동. 학습 상수 + 자기 행의 game_type·pitcher_team_id 만 사용 → 행 독립."""
    dt = np.zeros(len(rows), dtype=float); isF = (rows["game_type"].astype(str).to_numpy() == "F").astype(float)
    lg = float(A.get("lvl_g", 0.0) or 0.0)
    if lg: dt = dt + lg * float(A["lvl_diff"]) * isF
    ln = float(A.get("lvln_g", 0.0) or 0.0)   # F 보정 × 이력크기 z (표 조회, 신인 = hist_z_none)
    if ln and len(A["hist_z_arr"]):
        loc = A["hist_index"].get_indexer(pd.Index(rows["pitcher_id"].astype(str).to_numpy()))
        z = np.where(loc >= 0, A["hist_z_arr"][np.where(loc >= 0, loc, 0)], float(A.get("hist_z_none", 0.0)))
        dt = dt + ln * float(A["lvl_diff"]) * isF * z
    hg = float(A.get("hand_g", 0.0) or 0.0)   # 플래툰(투수손×타자손) 환경 편차 — 행 자기 열 + 상수 4개 (종민 L26/L87 재현)
    if hg and A.get("hand_dev"):
        ph = pd.to_numeric(rows["pitcher_hand"], errors="coerce").fillna(0).astype(int).astype(str).to_numpy()
        bh = pd.to_numeric(rows["batter_hand"], errors="coerce").fillna(0).astype(int).astype(str).to_numpy()
        hd = {str(k): float(v) for k, v in A["hand_dev"].items()}
        dt = dt + hg * np.array([hd.get(p + "_" + b, 0.0) for p, b in zip(ph, bh)], dtype=float)
    tg = float(A.get("lvl_team_g", 0.0) or 0.0)
    if tg and len(A["team_dev_arr"]):
        loc = A["team_index"].get_indexer(pd.Index(rows["pitcher_team_id"].astype(str).to_numpy()))
        dt = dt + tg * np.where(loc >= 0, A["team_dev_arr"][np.where(loc >= 0, loc, 0)], 0.0)
    return _norm(P + (dt / 100.0)[:, None] * SF)


def apply_usage(P, rows, T, mdir):
    """기용 B군(투구 직전 볼·스트라이크·주자·득점·아웃) 구성 → 투수 수준 잔차 회귀 계수(β, 2022~24 투수-시즌 적합)를 행별로 가산.
    지표가 투수 평균을 내주므로 투수별 2025 기용 구성이 순위 신호가 된다. w = α/(n_p+α) 로 이력 큰 투수는 약하게 (신인 1)."""
    U = json.load(open(mdir / "usage.json", encoding="utf-8"))
    if U.get("mode") == "S":   # 구조 항: 카운트 12상태 더미(자기 행) + 초구 스트라이크율 이력 z(투수 표 조회, 미등록 0)
        b = pd.to_numeric(rows["balls_before"], errors="coerce").fillna(0).to_numpy(); st = pd.to_numeric(rows["strikes_before"], errors="coerce").fillna(0).to_numpy()
        fidx = pd.Index(np.asarray(U["fs_idx"], dtype=object)); floc = fidx.get_indexer(pd.Index(rows["pitcher_id"].astype(str).to_numpy())); fz = np.asarray(U["fs_z"], dtype=float)
        cols = []
        for k, mu, sd in zip(U["keys"], U["mu"], U["sd"]):
            if k.startswith("cnt_b"): cols.append((((b == int(k[5])) & (st == int(k[7]))).astype(float) - mu) / sd)
            elif k == "fs_z": cols.append(np.where(floc >= 0, fz[np.where(floc >= 0, floc, 0)], 0.0))
            else: cols.append((pd.to_numeric(rows[k], errors="coerce").fillna(mu).to_numpy(dtype=float) - mu) / sd)
        Z = np.column_stack(cols)
    else:
        Z = np.column_stack([(pd.to_numeric(rows[k], errors="coerce").fillna(mu).to_numpy(dtype=float) - mu) / sd for k, mu, sd in zip(U["keys"], U["mu"], U["sd"])])
    if U.get("mode") == "C":   # 과거 기용 z 평균(표) 과 블렌드: x = (1−λ)·z_row + λ·z̄_hist(pitcher) · 미등장 투수는 z̄=0
        idx = pd.Index(np.asarray(U["hist_idx"], dtype=object)); loc = idx.get_indexer(pd.Index(rows["pitcher_id"].astype(str).to_numpy()))
        HZ = np.asarray(U["hist_z"], dtype=float); ZH = np.where((loc >= 0)[:, None], HZ[np.where(loc >= 0, loc, 0)], 0.0)
        lam = float(U["lam"]); Z = (1.0 - lam) * Z + lam * ZH
    dt = Z @ np.asarray(U["beta"], dtype=float)
    if U.get("weighted", False):
        n = pd.DataFrame({"pitcher_id": rows["pitcher_id"].astype(str).to_numpy()}).merge(T["pitcher"][["p_n"]], left_on="pitcher_id", right_index=True, how="left")["p_n"].fillna(0.0).to_numpy(dtype=float)
        dt = dt * (float(U["alpha"]) / (n + float(U["alpha"])))
    dt = dt * float(U.get("scale", 1.0))
    return _norm(P + (dt / 100.0)[:, None] * SF)


def apply_pch(P, rows, mdir):
    """투수×카운트12×같은손 조건부 편차 표(Δ, 학습 데이터에서 λ 부분풀링) — 자기 행의 pitcher_id·balls·strikes·두 손 으로 조회만. 미등록 셀 0."""
    U = json.load(open(mdir / "pch.json", encoding="utf-8")); idx = pd.Index(np.asarray(U["keys"], dtype=object)); vals = np.asarray(U["vals"], dtype=float); s = float(U.get("scale", 1.0))
    b = pd.to_numeric(rows["balls_before"], errors="coerce").fillna(0).astype(int).to_numpy(); st = pd.to_numeric(rows["strikes_before"], errors="coerce").fillna(0).astype(int).to_numpy()
    ph = pd.to_numeric(rows["pitcher_hand"], errors="coerce").fillna(0).astype(int).to_numpy(); bh = pd.to_numeric(rows["batter_hand"], errors="coerce").fillna(0).astype(int).to_numpy()
    ch = (b * 3 + st) * 2 + (ph == bh).astype(int); key = pd.Index([f"{p}|{c}" for p, c in zip(rows["pitcher_id"].astype(str).to_numpy(), ch)])
    loc = idx.get_indexer(key); dt = np.where(loc >= 0, vals[np.where(loc >= 0, loc, 0)], 0.0) * s
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
    if (mdir / "jm_root" / "script.py").exists():   # ★ 하이브리드: 목표 t 는 동봉한 종민 L87 번들로, 등급 배분은 dain GBDT 로
        import importlib.util
        spec = importlib.util.spec_from_file_location("jm_script", str(mdir / "jm_root" / "script.py")); jm = importlib.util.module_from_spec(spec); spec.loader.exec_module(jm)
        B = jm.make_bundle(mdir / "jm_root"); Pj = np.empty((len(test), 3), dtype=float)
        for s0 in range(0, len(test), 200000):   # 배치는 메모리 단위 · 행별 독립
            e0 = min(s0 + 200000, len(test)); Pj[s0:e0] = jm.predict(test.iloc[s0:e0].reset_index(drop=True), B)
        tj = 100.0 * Pj[:, 0] + 30.0 * Pj[:, 1]; sg = 100.0 * P[:, 0] + 30.0 * P[:, 1]
        P = _norm(P + ((tj - sg) / 100.0)[:, None] * SF)   # 행 제구점수 = L87 목표 · Shadow:Heart:Failure 형태 = dain
        if (mdir / "usage.json").exists(): P = apply_usage(P, test, T, mdir)   # d21+: 기용 B군 항
        if (mdir / "pch.json").exists(): P = apply_pch(P, test, mdir)   # d41+: 투수×카운트×같은손 조건부 편차
        return P
    mode = meta.get("mode", "gbdt")
    if mode in ("anchor", "blend"):
        Pe = eb_predict(test, load_eb(mdir))
        if mode == "anchor": P = anchor(P, Pe, float(meta.get("lam", 1.0)))
        else: w = float(meta["eb_weight"]); P = _norm((1.0 - w) * P + w * Pe)
    if meta.get("inject_eps", 0) and "inj_z" in T["pitcher"].columns:
        z = pd.DataFrame({"pitcher_id": test["pitcher_id"].astype(str).to_numpy()}).merge(T["pitcher"][["inj_z"]], left_on="pitcher_id", right_index=True, how="left")["inj_z"].fillna(0.0).to_numpy(dtype=float)
        P = inject(P, z, float(meta["inject_eps"]))
    A = load_adj(mdir)                       # d9+: 환경 보정 (adj.json 있을 때만)
    if A is not None: P = apply_adj(P, test, A)
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
