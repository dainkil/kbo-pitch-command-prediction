# -*- coding: utf-8 -*-
"""구조 항 배포용 적합: C12(카운트 12상태 더미) + FS(초구 스트라이크율 이력표) 를 현행 행 항 전부(EB·F·팀·플래툰·B5) 위의 투수-시즌 잔차에 ridge 적합(2022~24 풀링).
출력 usage.json: keys(C12+fs_z) · beta(C12 는 s_c12 배, fs 는 s_fs 배 사전 곱) · mu/sd(전 행) · fs 표(전 행 season<2025 · 0-0 카운트 STRK_YN · κ100 · z) · weighted α=2000.
사용: usage_fit4.py --out <usage.json> [--s_c12 0.5] [--s_fs 1.0]"""
import numpy as np, pandas as pd, sys, json, argparse
sys.path.insert(0, "/home/ubuntu/workspace/산출물/dain/eda"); from _common import load, PT
sys.path.insert(0, "/home/ubuntu/workspace/산출물/dain/src"); from build_pre23f import eb_tables_w, cells
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--s_c12", type=float, default=0.5); ap.add_argument("--s_fs", type=float, default=1.0); ap.add_argument("--lam", type=float, default=2.0); A = ap.parse_args()
d = load(labeled_only=False); L = d[d.y.notna()].copy(); L["sc"] = PT[L.y.to_numpy().astype(int)]; L["isF"] = (L.game_type == "F").astype(float); G = np.array([100., 30., 0.]); AW = 2000.0
B5 = ["balls_before", "strikes_before", "num_runners_on", "run_total_before", "outs_before"]; U = json.load(open("/home/ubuntu/workspace/산출물/dain/out/usage_B.json"))
C12 = [f"cnt_b{b}s{s}" for b in range(4) for s in range(3)]
def c12(rows):
    b = rows.balls_before.to_numpy(); s = rows.strikes_before.to_numpy(); return np.column_stack([((b == bb) & (s == ss)).astype(float) for bb in range(4) for ss in range(3)])
def eb_rows(E, rows):
    pr = E["prior"][cells(rows)]; loc = pd.Index(E["pid"].astype(str)).get_indexer(pd.Index(rows.pitcher_id.astype(str).to_numpy())); have = loc >= 0; safe = np.where(have, loc, 0)
    n = np.where(have, E["pn"][safe], 0.0); rate = np.where(have[:, None], E["prate"][safe], 0.0); return ((rate * n[:, None] + 2000 * pr) / (n + 2000)[:, None]) @ G, n
def fs_table(Hall, kappa=100.0):
    F0 = Hall[(Hall.balls_before == 0) & (Hall.strikes_before == 0)]; k = (F0.STRK_YN == "T").astype(float); g = pd.DataFrame({"k": k.to_numpy()}, index=F0.pitcher_id.to_numpy()).groupby(level=0).k.agg(["sum", "size"])
    gmu = float(k.mean()); r = (g["sum"] + kappa * gmu) / (g["size"] + kappa) - gmu; return r / r.std(), float(gmu), float(r.std())
gs = []
for Y in (2022, 2023, 2024):
    H = L[L.season < Y]; Hall = d[d.season < Y]; E = eb_tables_w(H, np.eye(3)[H.y.to_numpy().astype(int)], np.ones(len(H)), alpha=2000.0)
    R = L[L.season == Y].copy(); m, n = eb_rows(E, R); w = AW / (n + AW)
    P = H[H.season == Y - 1]; mu = P.sc.mean(); tdev = P.groupby("pitcher_team_id").sc.mean() - mu; hdev = P.groupby(["pitcher_hand", "batter_hand"]).sc.mean() - mu
    Z = (R[B5].astype(float).to_numpy() - np.array(U["mu"])) / np.array(U["sd"]); cnt = 0.5 * w * (Z @ np.array(U["beta"]))
    base = m + 1.333 * R.isF.to_numpy() - 0.25 * R.pitcher_team_id.map(tdev).fillna(0).to_numpy() + 0.6 * pd.Series(list(zip(R.pitcher_hand, R.batter_hand))).map(hdev).fillna(0).to_numpy() + cnt
    Xc = c12(R); Xc = (Xc - c12(Hall).mean(0)) / np.maximum(c12(Hall).std(0), 1e-9)
    fsz, _, _ = fs_table(Hall); fz = R.pitcher_id.map(fsz).fillna(0.0).to_numpy()
    df = pd.DataFrame({"pid": R.pitcher_id.to_numpy(), "sc": R.sc.to_numpy(), "base": base, "n": n, "fs_z": fz}); df = pd.concat([df, pd.DataFrame(Xc, columns=C12, index=df.index)], axis=1)
    g = df.groupby("pid").agg(cnt=("sc", "size"), sc=("sc", "mean"), base=("base", "mean"), n=("n", "first"), fs_z=("fs_z", "first"), **{k: (k, "mean") for k in C12}); gs.append(g[g.cnt >= 50])
keys = C12 + ["fs_z"]
X = np.vstack([g[keys].to_numpy() * (AW / (g.n.to_numpy() + AW))[:, None] for g in gs]); r = np.concatenate([(g.sc - g.base).to_numpy() for g in gs])
Xm = X - X.mean(0); rc = r - r.mean(); beta = np.linalg.solve(Xm.T @ Xm + A.lam * np.eye(len(keys)), Xm.T @ rc)
print("β(2022~24 풀 적합):", dict(zip(keys, np.round(beta, 2))))
beta_dep = np.concatenate([beta[:12] * A.s_c12, beta[12:] * A.s_fs])
Hall = d; mu_c = c12(Hall).mean(0); sd_c = np.maximum(c12(Hall).std(0), 1e-9); fsz, gmu, rsd = fs_table(Hall)
out = {"mode": "S", "keys": keys, "beta": [float(b) for b in beta_dep], "mu": [float(x) for x in mu_c] + [0.0], "sd": [float(x) for x in sd_c] + [1.0], "weighted": True, "alpha": AW, "scale": 1.0,
       "fs_idx": [str(i) for i in fsz.index], "fs_z": [float(v) for v in fsz.to_numpy()],
       "note": f"C12 ×{A.s_c12} + FS(초구S율, κ100, gmu {gmu:.3f}, sd {rsd:.4f}) ×{A.s_fs} · β 2022~24 투수잔차 ridge λ={A.lam} (기준 = EB+F+팀+플래툰+B5) · w=α/(n+α) α=2000 · 표 투수 {len(fsz)}"}
json.dump(out, open(A.out, "w"), ensure_ascii=False); print("saved", A.out, "·", out["note"])
