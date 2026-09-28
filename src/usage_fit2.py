# -*- coding: utf-8 -*-
"""usage_fit.py 확장: 피처셋 × 이력가중 α_w × scale 의 표본외 폴드 스캔 (τ · 소이력 τ · 행MSE). LB 실측: scale 1.0 → Player +0.54, 0.5 → +1.08 (오목).
피처셋  B5 = balls,strikes,runners,run_total,outs (선형) · C12 = 카운트 12셀 더미 + runners,run_total,outs · C12B = C12 + base_state 8 더미(runners 대체)
사용: usage_fit2.py --out ../out/usage2.json --pick <set>:<alpha_w>   (pick 이 있으면 전체 적합 계수를 저장)"""
import sys, os, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from build_pre23f import eb_tables_w, cells
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eda")); from _common import load, PT
from scipy.stats import kendalltau
BASE = ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]
SETS = {"B5": ["balls_before", "strikes_before", "num_runners_on", "run_total_before", "outs_before"],
        "C12": [f"cnt_b{b}s{s}" for b in range(4) for s in range(3)] + ["num_runners_on", "run_total_before", "outs_before"],
        "C12B": [f"cnt_b{b}s{s}" for b in range(4) for s in range(3)] + [f"bs_{x}" for x in BASE] + ["run_total_before", "outs_before"]}
def usage_cols(rows, keys):
    """행 자기 열만으로 키 → 값 행렬 (script 에도 동일 함수)."""
    b = pd.to_numeric(rows["balls_before"], errors="coerce").fillna(0).to_numpy(); s = pd.to_numeric(rows["strikes_before"], errors="coerce").fillna(0).to_numpy(); bs = rows["base_state"].astype(str).to_numpy()
    out = []
    for k in keys:
        if k.startswith("cnt_b"): out.append(((b == int(k[5])) & (s == int(k[7]))).astype(float))
        elif k.startswith("bs_"): out.append((bs == k[3:]).astype(float))
        else: out.append(pd.to_numeric(rows[k], errors="coerce").fillna(0).to_numpy(dtype=float))
    return np.column_stack(out)
ap = argparse.ArgumentParser(); ap.add_argument("--alpha", type=float, default=2000.0); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--out", default=""); ap.add_argument("--pick", default=""); A = ap.parse_args()
d = load(labeled_only=False); L = d[d.y.notna()].copy(); L["sc"] = PT[L.y.to_numpy().astype(int)]; G = np.array([100.0, 30.0, 0.0])
def eb_rows(E, rows):
    pr = E["prior"][cells(rows)]; loc = pd.Index(E["pid"].astype(str)).get_indexer(pd.Index(rows.pitcher_id.astype(str).to_numpy())); have = loc >= 0; safe = np.where(have, loc, 0)
    n = np.where(have, E["pn"][safe], 0.0); rate = np.where(have[:, None], E["prate"][safe], 0.0)
    return ((rate * n[:, None] + E["alpha"] * pr) / (n + E["alpha"])[:, None]) @ G, n
ALLK = sorted(set(sum(SETS.values(), []))); pan = {}
for Y in (2022, 2023, 2024):
    H = L[L.season < Y]; E = eb_tables_w(H, np.eye(3)[H.y.to_numpy().astype(int)], np.ones(len(H)), alpha=A.alpha)
    UH = usage_cols(H, ALLK); mu = UH.mean(0); sd = UH.std(0); sd[sd == 0] = 1
    R = L[L.season == Y]; m, n = eb_rows(E, R); Z = (usage_cols(R, ALLK) - mu) / sd
    df = pd.DataFrame({"pid": R.pitcher_id.to_numpy(), "m": m, "n": n, "sc": R.sc.to_numpy()}); df = pd.concat([df, pd.DataFrame(Z, columns=ALLK, index=df.index)], axis=1)
    g = df.groupby("pid").agg(cnt=("sc", "size"), sc=("sc", "mean"), m=("m", "mean"), n=("n", "first"), **{k: (k, "mean") for k in ALLK}); pan[Y] = (df, g[g.cnt >= 50], mu, sd)
def wfun(n, aw): return np.ones_like(n) if aw is None else aw / (n + aw)
def fit(gs, keys, aw):
    X = np.vstack([g[keys].to_numpy() * wfun(g.n.to_numpy(), aw)[:, None] for g in gs]); r = np.concatenate([(g.sc - g.m).to_numpy() for g in gs])
    Xc = X - X.mean(0); rc = r - r.mean(); return np.linalg.solve(Xc.T @ Xc + A.lam * np.eye(len(keys)), Xc.T @ rc)
def ev(df, keys, beta, aw, scale):
    dt = (df[keys].to_numpy() @ beta) * wfun(df.n.to_numpy(), aw) * scale; t = df.m.to_numpy() + dt
    gg = pd.DataFrame({"pid": df.pid, "sc": df.sc, "m": df.m, "t": t, "n": df.n}).groupby("pid").agg(cnt=("sc", "size"), sc=("sc", "mean"), m=("m", "mean"), t=("t", "mean"), n=("n", "first")); gg = gg[gg.cnt >= 50]
    lo = gg[gg.n <= 300]; return (kendalltau(gg.t, gg.sc, variant="b").statistic - kendalltau(gg.m, gg.sc, variant="b").statistic,
                                  kendalltau(lo.t, lo.sc, variant="b").statistic - kendalltau(lo.m, lo.sc, variant="b").statistic, float(((t - df.sc) ** 2).mean() / ((df.m - df.sc) ** 2).mean() - 1) * 100)
rows = []
for sname, keys in SETS.items():
    for aw in (500.0, 2000.0, 8000.0, None):
        for scale in (0.5, 0.75):
            res = {}
            for Y in (2023, 2024):
                tr = [pan[y][1] for y in (2022, 2023, 2024) if y != Y]; beta = fit(tr, keys, aw); res[Y] = ev(pan[Y][0], keys, beta, aw, scale)
            rows.append(dict(set=sname, aw=aw, scale=scale, d23=res[2023][0], d24=res[2024][0], lo23=res[2023][1], lo24=res[2024][1], mse23=res[2023][2], mse24=res[2024][2], mean=(res[2023][0] + res[2024][0]) / 2))
R = pd.DataFrame(rows).sort_values("mean", ascending=False); pd.set_option("display.width", 200); print(R.round(4).to_string(index=False))
if A.pick:
    sname, aw = A.pick.split(":"); aw = None if aw in ("None", "inf") else float(aw); keys = SETS[sname]
    beta = fit([pan[y][1] for y in (2022, 2023, 2024)], keys, aw); UH = usage_cols(L, ALLK); mu = UH.mean(0); sd = UH.std(0); sd[sd == 0] = 1; idx = [ALLK.index(k) for k in keys]
    json.dump({"keys": keys, "beta": [float(b) for b in beta], "mu": [float(mu[i]) for i in idx], "sd": [float(sd[i]) for i in idx], "weighted": aw is not None, "alpha": aw if aw is not None else 0.0, "scale": 1.0,
               "note": f"{sname} · α_w={aw} · β 2022~24 투수잔차 ridge λ={A.lam} · z 표준화 전체 라벨행"}, open(A.out, "w"), ensure_ascii=False)
    print("saved", A.out, dict(zip(keys, np.round(beta, 3))))
