# -*- coding: utf-8 -*-
"""기용 항의 행 왜곡을 줄이는 두 변형 (표본외 폴드): 투수의 **과거 기용 평균** z̄_hist (표, 전 행 season<Y, κ 수축) 를
 (D) 별도 회귀항으로 추가:  dt = Σβ_k z_k(row) + Σβ'_k z̄_hist_k(pitcher)
 (C) 행값을 과거 평균으로 블렌드: dt = Σβ_k [(1−λ) z_k(row) + λ z̄_hist_k]   (λ=0.3/0.5/0.7)
비교 기준 B5 (aw=2000). 사용: usage_fit3.py [--out ../out/usage3.json --pick D|C0.5]"""
import sys, os, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from build_pre23f import eb_tables_w, cells
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eda")); from _common import load, PT
from scipy.stats import kendalltau
KEYS = ["balls_before", "strikes_before", "num_runners_on", "run_total_before", "outs_before"]; KAPPA = 200.0
ap = argparse.ArgumentParser(); ap.add_argument("--alpha", type=float, default=2000.0); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--out", default=""); ap.add_argument("--pick", default=""); A = ap.parse_args()
d = load(labeled_only=False); L = d[d.y.notna()].copy(); L["sc"] = PT[L.y.to_numpy().astype(int)]; G = np.array([100.0, 30.0, 0.0])
def eb_rows(E, rows):
    pr = E["prior"][cells(rows)]; loc = pd.Index(E["pid"].astype(str)).get_indexer(pd.Index(rows.pitcher_id.astype(str).to_numpy())); have = loc >= 0; safe = np.where(have, loc, 0)
    n = np.where(have, E["pn"][safe], 0.0); rate = np.where(have[:, None], E["prate"][safe], 0.0)
    return ((rate * n[:, None] + E["alpha"] * pr) / (n + E["alpha"])[:, None]) @ G, n
def hist_usage(Hall, mu, sd):
    """투수별 과거 기용 z 평균 (전 행, 라벨 무관) κ 수축 → DataFrame(index pid)"""
    Z = (Hall[KEYS].astype(float).to_numpy() - mu) / sd; g = pd.DataFrame(Z, columns=KEYS, index=Hall.pitcher_id.to_numpy()).groupby(level=0).agg(["sum", "size"])
    n = g.loc[:, (KEYS[0], "size")].to_numpy(float); out = pd.DataFrame({k: g.loc[:, (k, "sum")].to_numpy(float) / (n + KAPPA) for k in KEYS}, index=g.index); return out
pan = {}
for Y in (2022, 2023, 2024):
    H = L[L.season < Y]; Hall = d[d.season < Y]; E = eb_tables_w(H, np.eye(3)[H.y.to_numpy().astype(int)], np.ones(len(H)), alpha=A.alpha)
    mu = Hall[KEYS].astype(float).mean().to_numpy(); sd = Hall[KEYS].astype(float).std().to_numpy(); HU = hist_usage(Hall, mu, sd)
    R = L[L.season == Y]; m, n = eb_rows(E, R); Z = (R[KEYS].astype(float).to_numpy() - mu) / sd
    ZH = HU.reindex(R.pitcher_id.astype(str).to_numpy()).fillna(0.0).to_numpy()
    df = pd.DataFrame({"pid": R.pitcher_id.to_numpy(), "m": m, "n": n, "sc": R.sc.to_numpy()}); df = pd.concat([df, pd.DataFrame(Z, columns=KEYS, index=df.index), pd.DataFrame(ZH, columns=[k + "_h" for k in KEYS], index=df.index)], axis=1)
    KH = KEYS + [k + "_h" for k in KEYS]; g = df.groupby("pid").agg(cnt=("sc", "size"), sc=("sc", "mean"), m=("m", "mean"), n=("n", "first"), **{k: (k, "mean") for k in KH}); pan[Y] = (df, g[g.cnt >= 50])
def fit(gs, keys, aw):
    X = np.vstack([g[keys].to_numpy() * (aw / (g.n.to_numpy() + aw))[:, None] for g in gs]); r = np.concatenate([(g.sc - g.m).to_numpy() for g in gs])
    Xc = X - X.mean(0); rc = r - r.mean(); return np.linalg.solve(Xc.T @ Xc + A.lam * np.eye(len(keys)), Xc.T @ rc)
def ev(df, X, beta, aw, scale):
    dt = (X @ beta) * (aw / (df.n.to_numpy() + aw)) * scale; t = df.m.to_numpy() + dt
    gg = pd.DataFrame({"pid": df.pid, "sc": df.sc, "m": df.m, "t": t, "n": df.n}).groupby("pid").agg(cnt=("sc", "size"), sc=("sc", "mean"), m=("m", "mean"), t=("t", "mean"), n=("n", "first")); gg = gg[gg.cnt >= 50]; lo = gg[gg.n <= 300]
    return (kendalltau(gg.t, gg.sc, variant="b").statistic - kendalltau(gg.m, gg.sc, variant="b").statistic, kendalltau(lo.t, lo.sc, variant="b").statistic - kendalltau(lo.m, lo.sc, variant="b").statistic, float(((t - df.sc) ** 2).mean() / ((df.m - df.sc) ** 2).mean() - 1) * 100)
KH = KEYS + [k + "_h" for k in KEYS]; rows = []; aw = 2000.0
for name in ("B5", "D(row+hist)", "H(hist only)", "C0.3", "C0.5", "C0.7"):
    for scale in (0.5, 0.75, 1.0):
        res = {}
        for Y in (2023, 2024):
            tr = [pan[y][1] for y in (2022, 2023, 2024) if y != Y]; df = pan[Y][0]
            if name == "B5": keys = KEYS; beta = fit(tr, keys, aw); X = df[keys].to_numpy()
            elif name.startswith("D"): keys = KH; beta = fit(tr, keys, aw); X = df[keys].to_numpy()
            elif name.startswith("H"): keys = [k + "_h" for k in KEYS]; beta = fit(tr, keys, aw); X = df[keys].to_numpy()
            else:
                lam = float(name[1:]); trb = [g.assign(**{k: (1 - lam) * g[k] + lam * g[k + "_h"] for k in KEYS}) for g in tr]; beta = fit(trb, KEYS, aw)
                X = np.column_stack([(1 - lam) * df[k].to_numpy() + lam * df[k + "_h"].to_numpy() for k in KEYS])
            res[Y] = ev(df, X, beta, aw, scale)
        rows.append(dict(var=name, scale=scale, d23=res[2023][0], d24=res[2024][0], lo23=res[2023][1], lo24=res[2024][1], mse23=res[2023][2], mse24=res[2024][2], mean=(res[2023][0] + res[2024][0]) / 2))
R = pd.DataFrame(rows); R["eff"] = R["mean"] / ((R.mse23 + R.mse24) / 2); pd.set_option("display.width", 200); print(R.sort_values("mean", ascending=False).round(4).to_string(index=False))
if A.pick:
    Hall = d; mu = Hall[KEYS].astype(float).mean().to_numpy(); sd = Hall[KEYS].astype(float).std().to_numpy(); HU = hist_usage(Hall, mu, sd)
    gs = [pan[y][1] for y in (2022, 2023, 2024)]
    if A.pick == "D": keys = KH; beta = fit(gs, keys, aw); out = {"mode": "D", "keys": KEYS, "beta": [float(b) for b in beta[:5]], "beta_h": [float(b) for b in beta[5:]]}
    else: lam = float(A.pick[1:]); trb = [g.assign(**{k: (1 - lam) * g[k] + lam * g[k + "_h"] for k in KEYS}) for g in gs]; beta = fit(trb, KEYS, aw); out = {"mode": "C", "lam": lam, "keys": KEYS, "beta": [float(b) for b in beta]}
    out.update(mu=[float(x) for x in mu], sd=[float(x) for x in sd], weighted=True, alpha=aw, scale=1.0, hist_idx=[str(i) for i in HU.index], hist_z=[[float(v) for v in HU.loc[i, KEYS]] for i in HU.index], kappa=KAPPA,
               note=f"{A.pick} · 과거 기용 z 평균 표(전 행 season<2025, κ={KAPPA}) · β 2022~24 투수잔차 ridge")
    json.dump(out, open(A.out, "w"), ensure_ascii=False); print("saved", A.out, "beta", np.round(beta, 3))
