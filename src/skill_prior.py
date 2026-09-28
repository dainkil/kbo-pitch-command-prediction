"""d10: 학습된 행-문맥 prior (skill prior) — anchor 목표 EB 의 배치셀 prior 수준을 q_skill(row) 로 갈아끼운다.
타깃 = 그 행이 속한 투수-시즌(≥50 라벨구)의 LOO 평균 제구점수. 가중 1/n_ps (τ 는 투수 동일가중).
피처 = 행 자기 열(A군 기용 [+B군]) + post23 + 투수 표(이력크기·F비중·구종구성·트랙맨 구성/구속) + 타자 표(레벨·EB).
투수 이력 등급(p_sh 등)은 넣지 않는다 — prior 는 이력과 독립이어야 EB 가 이중계산하지 않는다.
사용:  python3 skill_prior.py --variant A --upto 2025 --out ../out/skill_A_2025          (배포용: 2020..2024 학습)
       python3 skill_prior.py --variant A --upto 2024 --out ../out/skill_A_2024 --eval   (2024 홀드아웃 진단: 2020..2023 학습)"""
import sys, os, json, time, argparse
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import *
from script import skill_X, skill_feats, SF, eb_predict, cells
from scipy.stats import kendalltau

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="A", choices=["A", "B", "C"]); ap.add_argument("--upto", type=int, default=2025); ap.add_argument("--out", required=True)
ap.add_argument("--rounds", type=int, default=300); ap.add_argument("--leaves", type=int, default=15); ap.add_argument("--minleaf", type=int, default=3000)
ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--eval", action="store_true", help="upto 시즌을 홀드아웃으로 진단 (표는 season<upto)")
ap.add_argument("--eb_alpha", type=float, default=1300.0); ap.add_argument("--first", type=int, default=2020)
ap.add_argument("--rfilter", action="store_true", help="학습·평가 행을 (R) | (season>=2023) 으로 제한 — 2025 평가면에 존재하는 행 유형만 (Phase2 상위팀 규칙)")
ap.add_argument("--gate", type=float, default=0.0, help="이력 n<gate 투수는 skill 이동을 n/gate 로 축소 (신인=0 → 셀 prior 유지)")
ap.add_argument("--resid", action="store_true", help="타깃 = LOO 평균 − 배치셀 prior 점수(행) : 셀 prior 를 형태·수준 모두 유지하고 잔차만 학습 (신인은 셀 prior 그대로)")
A = ap.parse_args(); T0 = time.time(); tk = lambda: f"[{time.time()-T0:5.0f}s]"
feats = skill_feats(A.variant); print(f"variant {A.variant} · 피처 {len(feats)}: {feats}", flush=True)

d_all = load(labeled_only=False); tm = load_tm(); lab_all = d_all[d_all.y.notna()]
print(f"{tk()} loaded {d_all.shape}", flush=True)

def season_rows(Y):
    """시즌 Y 의 라벨 행을 season<Y 표로 피처화. 반환 (X, target, weight, pid, pt, p_n)."""
    T = build_tables(lab_all[lab_all.season < Y], d_all[d_all.season < Y], tm[tm.season < Y], Y)
    cur = lab_all[lab_all.season == Y]
    if A.rfilter: cur = cur[(cur.game_type == "R") | (cur.season >= 2023)]
    X = skill_X(cur, T, feats)
    g = cur.groupby("pitcher_id").pt.agg(["sum", "size"]); s = cur.pitcher_id.map(g["sum"]).to_numpy(); n = cur.pitcher_id.map(g["size"]).to_numpy(float)
    ok = n >= 50
    tgt = (s - cur.pt.to_numpy()) / np.maximum(n - 1, 1)          # LOO 평균 (자기 라벨 제외)
    if A.resid: tgt = tgt - eb_tables(lab_all[lab_all.season < Y], alpha=A.eb_alpha)["prior"][cells(cur)] @ PT   # 셀 prior 점수 대비 잔차
    pn = pd.Series(cur.pitcher_id.astype(str).to_numpy()).map(T["pitcher"]["p_n"]).fillna(0.0).to_numpy()
    return X[ok], tgt[ok], (1.0 / n[ok]), cur.pitcher_id.to_numpy()[ok], cur.pt.to_numpy()[ok], pn[ok], T, cur

parts = [season_rows(Y) for Y in range(A.first, A.upto)]
Xtr = pd.concat([p[0] for p in parts]); ytr = np.concatenate([p[1] for p in parts]); wtr = np.concatenate([p[2] for p in parts])
print(f"{tk()} 학습 {list(range(A.first, A.upto))} · 행 {len(Xtr):,} · 타깃 mean {ytr.mean():.2f} sd {ytr.std():.2f}", flush=True)
params = dict(objective="regression", learning_rate=A.lr, num_leaves=A.leaves, min_data_in_leaf=A.minleaf, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=12, verbose=-1, seed=0)
m = lgb.train(params, lgb.Dataset(Xtr, ytr, weight=wtr), A.rounds)
imp = pd.Series(m.feature_importance("gain"), index=feats); imp = (imp / imp.sum() * 100).sort_values(ascending=False)
print(f"{tk()} gain%: {imp.round(1).to_dict()}", flush=True)
os.makedirs(A.out, exist_ok=True); m.save_model(os.path.join(A.out, "skill_prior.txt"))
mu_tr = m.predict(Xtr.to_numpy(dtype=np.float32))
meta = {"variant": A.variant, "model": "skill_prior.txt", "features": feats, "w": 1.0, "cap": 15.0, "rounds": A.rounds, "resid": bool(A.resid), "rfilter": bool(A.rfilter), "gate": float(A.gate),
        "train_seasons": list(range(A.first, A.upto)), "params": {k: v for k, v in params.items() if k != "num_threads"},
        "mu_train_mean": float(mu_tr.mean()), "mu_train_sd": float(mu_tr.std())}
json.dump(meta, open(os.path.join(A.out, "skill_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{tk()} saved → {A.out} · μ(학습행) mean {mu_tr.mean():.2f} sd {mu_tr.std():.2f}", flush=True)

if A.eval:
    # ── 홀드아웃 진단: upto 시즌 (표 season<upto). 투수(≥50구) 평균 μ 의 τ 를 이력 구간별로 — 배치셀 prior · EB(셀) · EB(skill) 과 비교
    Y = A.upto; Xe, te, we, pid, pt, pn, T, cur = season_rows(Y)
    labU = lab_all[lab_all.season < Y]; E = eb_tables(labU, alpha=A.eb_alpha)
    S = {"booster": m, "features": feats, "cap": 15.0, "w": 1.0, "resid": bool(A.resid), "gate": float(A.gate)}
    cellp = E["prior"][cells(cur)] @ PT
    okm = (cur.pitcher_id.map(cur.groupby("pitcher_id").size()) >= 50).to_numpy()
    mu = m.predict(Xe.to_numpy(dtype=np.float32)); mu = mu + cellp[okm] if A.resid else mu   # 표시용 μ = 셀 prior + 잔차
    eb0 = eb_predict(cur, E) @ PT; eb1 = eb_predict(cur, E, T, S) @ PT
    df = pd.DataFrame({"pid": cur.pitcher_id.to_numpy(), "act": cur.pt.to_numpy(), "cell": cellp, "mu": np.nan, "eb0": eb0, "eb1": eb1, "pn": pd.Series(cur.pitcher_id.astype(str).to_numpy()).map(T["pitcher"]["p_n"]).fillna(0.0).to_numpy()})
    df.loc[okm, "mu"] = mu
    g = df[okm].groupby("pid").agg(n=("act", "size"), act=("act", "mean"), cell=("cell", "mean"), mu=("mu", "mean"), eb0=("eb0", "mean"), eb1=("eb1", "mean"), pn=("pn", "first"))
    def tau(a, b): r = kendalltau(np.round(a, 10), np.round(b, 10), variant="b").statistic; return 0.0 if np.isnan(r) else r
    print(f"\n★ {Y} 홀드아웃 · 투수 {len(g)}명 (≥50구) · 표 season<{Y} · EB α={A.eb_alpha:.0f}")
    print(f"{'구간':<14}{'명':>5}  {'셀prior τ':>10}  {'q_skill τ':>10}  {'EB(셀) τ':>10}  {'EB(skill) τ':>12}  {'ΔEB':>8}")
    buckets = [("전체", g.pn >= 0), ("이력 0", g.pn == 0), ("1~300", (g.pn > 0) & (g.pn <= 300)), ("301~1000", (g.pn > 300) & (g.pn <= 1000)), ("1001~3000", (g.pn > 1000) & (g.pn <= 3000)), ("3001+", g.pn > 3000), ("≤300 (소이력)", g.pn <= 300)]
    for nm, mk in buckets:
        q = g[mk]
        if len(q) < 5: continue
        t_c, t_m, t_0, t_1 = tau(q.act, q.cell), tau(q.act, q.mu), tau(q.act, q.eb0), tau(q.act, q.eb1)
        print(f"{nm:<14}{len(q):>5}  {t_c:>+10.4f}  {t_m:>+10.4f}  {t_0:>+10.4f}  {t_1:>+12.4f}  {t_1-t_0:>+8.4f}", flush=True)
    # 행 수준: μ 와 실제 등급점수의 상관 (prior 의 행 MSE 비용 감시) · Brier 대용
    r_cell = np.corrcoef(df.cell[okm], df.act[okm])[0, 1]; r_mu = np.corrcoef(mu, df.act[okm])[0, 1]
    print(f"행 수준 corr(실제 pt): 셀 prior {r_cell:+.4f} · q_skill {r_mu:+.4f} · 행 MSE 셀 {((df.cell[okm]-df.act[okm])**2).mean():.1f} q_skill {((mu-df.act[okm])**2).mean():.1f}")
    print(f"μ 분포(평가행): mean {mu.mean():.2f} sd {mu.std():.2f} · 투수평균 μ sd {g.mu.std():.3f} (셀 {g.cell.std():.3f})")
    print(f"{tk()} eval done", flush=True)
