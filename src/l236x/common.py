# -*- coding: utf-8 -*-
"""L236 고도화 공통 — 종민 코드는 읽기만(import), 산출물은 전부 산출물/dain 아래."""
import os, sys, json, pickle, numpy as np, pandas as pd
R = "/home/ubuntu/workspace/산출물"; JM = R + "/종민"; DN = R + "/dain"; X = DN + "/src/l236x"; BASE = X + "/base"
os.environ["SUBDIR"] = BASE; sys.path.insert(0, BASE); sys.path.insert(0, JM + "/src")
import train_L7 as T7                      # BASE/script.py(=L236 zip 의 script.py) 를 import 한다
import script as SC
P = lambda *a: print(*a, flush=True)
G = np.array([100.0, 30.0, 0.0])
B29 = ["row_id","season","game_month","game_dayofweek","inning","top_bottom","game_type","balls_before","strikes_before",
       "outs_before","run_top_before","run_bot_before","run_total_before","score_diff_home","score_diff_pitcher_team",
       "runner_on_1b","runner_on_2b","runner_on_3b","num_runners_on","base_state","home_win_expectancy","away_win_expectancy",
       "li","pitcher_id","batter_id","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id"]

def load_d():
    p = X + "/cache_d.pkl"
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    d = T7.load_all(); d["rev"] = (d.REVERSE_YN == "T").astype(float)
    pickle.dump(d, open(p, "wb"), protocol=4); return d

def base_meta():
    return json.load(open(BASE + "/model/meta.json", encoding="utf-8"))

def base_tables():
    return dict(np.load(BASE + "/model/tables.npz", allow_pickle=False))

def score(y, p, pid, tag=""):
    """공식 지표: Pitch=100(1−Brier/2) · Player=50(1+τ_b), 50구+ 투수"""
    from scipy.stats import kendalltau
    Y = np.eye(3)[y]; pitch = 100 * (1 - ((p - Y) ** 2).sum(1).mean() / 2)
    g = pd.DataFrame({"p": pid, "s": G[y], "h": p @ G}).groupby("p").agg(n=("s", "size"), s=("s", "mean"), h=("h", "mean"))
    g = g[g.n >= 50]; t = kendalltau(g.s.round(10).to_numpy(), g.h.round(10).to_numpy(), variant="b").statistic
    sc = .7 * pitch + .3 * 50 * (1 + t)
    if tag: P(f"   {tag:40s} Score {sc:8.4f} | Pitch {pitch:8.4f} | Player {50*(1+t):8.4f} (τ {t:+.4f})")
    return sc, pitch, 50 * (1 + t), g.h
