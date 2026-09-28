# -*- coding: utf-8 -*-
"""기용 B군(투구 직전 카운트·주자·득점·아웃) → 투수 수준 잔차 회귀. 행 자기 열 × 학습 계수 → t 에 가산 (지표가 투수 평균을 내준다).
폴드: 시즌 Y 의 EB(라벨행 season<Y, α=2000, 배치셀 prior) 로 행별 점수 m_row 를 만들고, 투수(평가 50구+) 잔차 r_p = 실제평균 − m 평균 을
투수별 기용평균 u_pk(표준화) 에 ridge 회귀. 표본외: {2022,2023} 적합 → 2024 평가 / {2022,2024} → 2023.
변형 A: 전 투수 동일 계수 · 변형 B: 이력 가중 w=α/(n+α) (신인 1, 대이력 ↓).
사용: usage_fit.py [--alpha 2000] [--out /path/usage.json]"""
import sys, os, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from build_pre23f import eb_tables_w, cells
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eda")); from _common import load, PT
from scipy.stats import kendalltau
KEYS = ["balls_before", "strikes_before", "num_runners_on", "run_total_before", "outs_before"]
ap = argparse.ArgumentParser(); ap.add_argument("--alpha", type=float, default=2000.0); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--out", default=""); A = ap.parse_args()
d = load(labeled_only=False); L = d[d.y.notna()].copy(); L["sc"] = PT[L.y.to_numpy().astype(int)]
G = np.array([100.0, 30.0, 0.0])

def eb_rows(E, rows):
    pr = E["prior"][cells(rows)]; loc = pd.Index(E["pid"].astype(str)).get_indexer(pd.Index(rows.pitcher_id.astype(str).to_numpy())); have = loc >= 0; safe = np.where(have, loc, 0)
    n = np.where(have, E["pn"][safe], 0.0); rate = np.where(have[:, None], E["prate"][safe], 0.0)
    P = (rate * n[:, None] + E["alpha"] * pr) / (n + E["alpha"])[:, None]; return P @ G, n

def season_panel(Y):
    H = L[L.season < Y]; E = eb_tables_w(H, np.eye(3)[H.y.to_numpy().astype(int)], np.ones(len(H)), alpha=A.alpha)
    mu = H[KEYS].astype(float).mean().to_numpy(); sd = H[KEYS].astype(float).std().to_numpy()
    R = L[L.season == Y]; m, n = eb_rows(E, R); Z = (R[KEYS].astype(float).to_numpy() - mu) / sd
    df = pd.DataFrame({"pid": R.pitcher_id.to_numpy(), "m": m, "n": n, "sc": R.sc.to_numpy()}); Zd = pd.DataFrame(Z, columns=KEYS, index=df.index); df = pd.concat([df, Zd], axis=1)
    g = df.groupby("pid").agg(cnt=("sc", "size"), sc=("sc", "mean"), m=("m", "mean"), n=("n", "first"), **{k: (k, "mean") for k in KEYS}); g = g[g.cnt >= 50]
    return df, g, mu, sd

def fit(gs, weighted):
    X = np.vstack([(g[KEYS].to_numpy() * ((A.alpha / (g.n.to_numpy() + A.alpha))[:, None] if weighted else 1.0)) for g in gs]); r = np.concatenate([(g.sc - g.m).to_numpy() for g in gs])
    Xc = X - X.mean(0); rc = r - r.mean(); beta = np.linalg.solve(Xc.T @ Xc + A.lam * np.eye(len(KEYS)), Xc.T @ rc); return beta

def evaluate(df, g, beta, weighted, tag):
    w = (A.alpha / (df.n.to_numpy() + A.alpha)) if weighted else 1.0
    dt = (df[KEYS].to_numpy() @ beta) * w; df2 = df.assign(t=df.m + dt)
    gg = df2.groupby("pid").agg(cnt=("sc", "size"), sc=("sc", "mean"), m=("m", "mean"), t=("t", "mean"), n=("n", "first")); gg = gg[gg.cnt >= 50]
    t0 = kendalltau(gg.m, gg.sc, variant="b").statistic; t1 = kendalltau(gg.t, gg.sc, variant="b").statistic
    lo = gg[gg.n <= 300]; tl0 = kendalltau(lo.m, lo.sc, variant="b").statistic; tl1 = kendalltau(lo.t, lo.sc, variant="b").statistic
    mse0 = float(((df.m - df.sc) ** 2).mean()); mse1 = float(((df2.t - df.sc) ** 2).mean())
    print(f"  {tag}: 투수 {len(gg)} · τ {t0:+.4f} → {t1:+.4f} (Δ{t1-t0:+.4f}) · 소이력(≤300) {len(lo)} τ {tl0:+.4f} → {tl1:+.4f} · 행MSE {mse0:.1f} → {mse1:.1f} ({(mse1/mse0-1)*100:+.3f}%) · 항 sd {float(np.std(dt)):.3f}점 · 투수평균항 sd {float(np.std(gg.t-gg.m)):.3f}")
    return t1 - t0

pan = {Y: season_panel(Y) for Y in (2022, 2023, 2024)}
for weighted in (False, True):
    print(f"\n===== 변형 {'B(이력가중)' if weighted else 'A(동일계수)'}")
    for Y in (2023, 2024):
        tr = [pan[y][1] for y in (2022, 2023, 2024) if y != Y]; beta = fit(tr, weighted)
        print(f"  [{Y} 평가 · {[y for y in (2022,2023,2024) if y != Y]} 적합] β = {dict(zip(KEYS, np.round(beta, 3)))}")
        evaluate(pan[Y][0], pan[Y][1], beta, weighted, f"{Y}")
    beta_all = fit([pan[y][1] for y in (2022, 2023, 2024)], weighted); print(f"  [전체 적합] β = {dict(zip(KEYS, np.round(beta_all, 3)))}")
    if A.out:
        H = L; mu = H[KEYS].astype(float).mean().to_numpy(); sd = H[KEYS].astype(float).std().to_numpy()
        json.dump({"keys": KEYS, "beta": [float(b) for b in beta_all], "mu": [float(x) for x in mu], "sd": [float(x) for x in sd], "weighted": weighted, "alpha": A.alpha,
                   "note": "t += Σ β_k·z_k(row)" + (" · × α/(n_p+α)" if weighted else "") + " · z 표준화 = 전체 라벨행 · β 는 2022~24 투수 잔차 회귀"},
                  open(A.out.replace(".json", ("_B" if weighted else "_A") + ".json"), "w"), ensure_ascii=False); print("  saved", A.out.replace(".json", ("_B" if weighted else "_A") + ".json"))
