# -*- coding: utf-8 -*-
"""단발 제출용 투수 단위 '제구 이력' 오프셋: FS(초구 스트라이크율, κ100) + 과거 카운트 구성 z̄(balls·strikes, κ200) 만 (역할성 변수 runners/outs/run_total 제외 — ROLE 계열 LB 실패 전례).
현행 행 항 전부(EB+F+팀+플래툰+B5) 위의 투수-시즌 잔차에 ridge(λ2) 적합 → 표본외 Δτ 출력 → 2022~24 풀 적합 β 로 투수별 결합 오프셋 표 off_z 를 만들어
usage.json(mode S, keys=["fs_z"], beta=[scale]) 로 저장. 추론: t += scale·w(n)·off_z[pitcher] (행 왜곡 0 · 표 조회만).
사용: usage_fit5.py --out <usage.json> --scale 1.0"""
import numpy as np, pandas as pd, sys, json, argparse
sys.path.insert(0, "/home/ubuntu/workspace/산출물/dain/eda"); from _common import load, PT
sys.path.insert(0, "/home/ubuntu/workspace/산출물/dain/src"); from build_pre23f import eb_tables_w, cells
from scipy.stats import kendalltau
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--lam", type=float, default=2.0); A = ap.parse_args()
d = load(labeled_only=False); L = d[d.y.notna()].copy(); L["sc"] = PT[L.y.to_numpy().astype(int)]; L["isF"] = (L.game_type == "F").astype(float); G = np.array([100., 30., 0.]); AW = 2000.0
B5 = ["balls_before", "strikes_before", "num_runners_on", "run_total_before", "outs_before"]; U = json.load(open("/home/ubuntu/workspace/산출물/dain/out/usage_B.json")); HK = ["balls_before", "strikes_before"]
def eb_rows(E, rows):
    pr = E["prior"][cells(rows)]; loc = pd.Index(E["pid"].astype(str)).get_indexer(pd.Index(rows.pitcher_id.astype(str).to_numpy())); have = loc >= 0; safe = np.where(have, loc, 0)
    n = np.where(have, E["pn"][safe], 0.0); rate = np.where(have[:, None], E["prate"][safe], 0.0); return ((rate * n[:, None] + 2000 * pr) / (n + 2000)[:, None]) @ G, n
def fs_table(Hall, kappa=100.0):
    F0 = Hall[(Hall.balls_before == 0) & (Hall.strikes_before == 0)]; k = (F0.STRK_YN == "T").astype(float); g = pd.DataFrame({"k": k.to_numpy()}, index=F0.pitcher_id.to_numpy()).groupby(level=0).k.agg(["sum", "size"])
    gmu = float(k.mean()); r = (g["sum"] + kappa * gmu) / (g["size"] + kappa) - gmu; return r / r.std()
def h_table(Hall, kappa=200.0):
    mu = Hall[HK].astype(float).mean().to_numpy(); sd = Hall[HK].astype(float).std().to_numpy(); Z = (Hall[HK].astype(float).to_numpy() - mu) / sd
    g = pd.DataFrame(Z, columns=HK, index=Hall.pitcher_id.to_numpy()).groupby(level=0).agg(["sum", "size"]); n = g.loc[:, (HK[0], "size")].to_numpy(float)
    return pd.DataFrame({k: g.loc[:, (k, "sum")].to_numpy(float) / (n + kappa) for k in HK}, index=g.index)
KEYS = ["fs_z"] + [k + "_h" for k in HK]; pan = {}
for Y in (2022, 2023, 2024):
    H = L[L.season < Y]; Hall = d[d.season < Y]; E = eb_tables_w(H, np.eye(3)[H.y.to_numpy().astype(int)], np.ones(len(H)), alpha=2000.0)
    R = L[L.season == Y].copy(); m, n = eb_rows(E, R); w = AW / (n + AW)
    P = H[H.season == Y - 1]; mu0 = P.sc.mean(); tdev = P.groupby("pitcher_team_id").sc.mean() - mu0; hdev = P.groupby(["pitcher_hand", "batter_hand"]).sc.mean() - mu0
    Z = (R[B5].astype(float).to_numpy() - np.array(U["mu"])) / np.array(U["sd"]); cnt = 0.5 * w * (Z @ np.array(U["beta"]))
    base = m + 1.333 * R.isF.to_numpy() - 0.25 * R.pitcher_team_id.map(tdev).fillna(0).to_numpy() + 0.6 * pd.Series(list(zip(R.pitcher_hand, R.batter_hand))).map(hdev).fillna(0).to_numpy() + cnt
    fsz = R.pitcher_id.map(fs_table(Hall)).fillna(0.0).to_numpy(); HT = h_table(Hall); ZH = HT.reindex(R.pitcher_id.astype(str).to_numpy()).fillna(0.0).to_numpy()
    df = pd.DataFrame({"pid": R.pitcher_id.to_numpy(), "sc": R.sc.to_numpy(), "base": base, "n": n, "fs_z": fsz}); df = pd.concat([df, pd.DataFrame(ZH, columns=KEYS[1:], index=df.index)], axis=1)
    g = df.groupby("pid").agg(cnt=("sc", "size"), sc=("sc", "mean"), base=("base", "mean"), n=("n", "first"), **{k: (k, "first") for k in KEYS}); pan[Y] = (df, g[g.cnt >= 50])
def fit(gs):
    X = np.vstack([g[KEYS].to_numpy() * (AW / (g.n.to_numpy() + AW))[:, None] for g in gs]); r = np.concatenate([(g.sc - g.base).to_numpy() for g in gs])
    Xc = X - X.mean(0); rc = r - r.mean(); return np.linalg.solve(Xc.T @ Xc + A.lam * np.eye(len(KEYS)), Xc.T @ rc)
def ev(df, beta, scale):
    dt = (df[KEYS].to_numpy() @ beta) * (AW / (df.n.to_numpy() + AW)) * scale; t = df.base.to_numpy() + dt
    gg = pd.DataFrame({"pid": df.pid, "sc": df.sc, "b": df.base, "t": t, "n": df.n}).groupby("pid").agg(cnt=("sc", "size"), sc=("sc", "mean"), b=("b", "mean"), t=("t", "mean"), n=("n", "first")); gg = gg[gg.cnt >= 50]; lo = gg[gg.n <= 300]
    return kendalltau(gg.t, gg.sc, variant="b").statistic - kendalltau(gg.b, gg.sc, variant="b").statistic, kendalltau(lo.t, lo.sc, variant="b").statistic - kendalltau(lo.b, lo.sc, variant="b").statistic, float(np.std(gg.t - gg.b))
for scale in (0.7, 1.0, 1.5):
    res = {}
    for Y in (2023, 2024):
        tr = [pan[y][1] for y in (2022, 2023, 2024) if y != Y]; res[Y] = ev(pan[Y][0], fit(tr), scale)
    print(f"FS+balls_h+strikes_h scale {scale}: 2023 Δτ {res[2023][0]:+.4f} (소이력 {res[2023][1]:+.4f}, sd {res[2023][2]:.3f}) · 2024 Δτ {res[2024][0]:+.4f} (소이력 {res[2024][1]:+.4f}, sd {res[2024][2]:.3f})")
beta = fit([pan[y][1] for y in (2022, 2023, 2024)]); print("β(2022~24 풀 적합):", dict(zip(KEYS, np.round(beta, 3))))
Hall = d; fs = fs_table(Hall); HT = h_table(Hall); idx = fs.index.union(HT.index)
off = beta[0] * fs.reindex(idx).fillna(0.0) + sum(beta[i + 1] * HT[k].reindex(idx).fillna(0.0) for i, k in enumerate(HK))
out = {"mode": "S", "keys": ["fs_z"], "beta": [float(A.scale)], "mu": [0.0], "sd": [1.0], "weighted": True, "alpha": AW, "scale": 1.0, "fs_idx": [str(i) for i in idx], "fs_z": [float(v) for v in off.to_numpy()],
       "note": f"투수 오프셋 = {beta[0]:.3f}·FS(초구S율 z, κ100) + {beta[1]:.3f}·balls_h + {beta[2]:.3f}·strikes_h (과거 카운트 z̄, κ200) · 표 season<2025 전 행 · scale {A.scale} · w=α/(n+α) α=2000 · 투수 {len(idx)} · 오프셋 sd {off.std():.3f}"}
json.dump(out, open(A.out, "w"), ensure_ascii=False); print("saved", A.out, "·", out["note"])
