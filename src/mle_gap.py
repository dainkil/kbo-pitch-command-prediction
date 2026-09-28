"""C-4: F−R 격차의 MLE(Major League Equivalency) 식 추정 — 리그 평균차(구성 혼합) vs 같은 투수의 F·R 쌍(within-pitcher).
시즌별로 (a) 리그 F−R 평균차, (b) 같은 시즌 F·R 둘 다 ≥30구 던진 투수의 평균(F−R), (c) 인접시즌 쌍(F_s → R_{s+1}) 을 찍는다.
γ=−0.6 (유효 격차 = lvl_diff·(1+γ)) 의 독립 근거용. 학습 없음, 읽기만."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eda")); from _common import load
d = load(labeled_only=True)
print(f"{'시즌':<6}{'리그 F−R':>10}{'within(F−R) 같은시즌':>22}{'n투수':>6}{'  F_s → R_s+1':>16}{'n':>5}{'  라벨률 F/R':>14}")
for s in range(2019, 2025):
    cur = d[d.season == s]; lg = cur[cur.game_type == "F"].pt.mean() - cur[cur.game_type == "R"].pt.mean()
    g = cur.groupby(["pitcher_id", "game_type"]).pt.agg(["mean", "size"]).unstack("game_type")
    both = g[(g[("size", "F")] >= 30) & (g[("size", "R")] >= 30)]
    w = np.minimum(both[("size", "F")], both[("size", "R")]); wi = np.average(both[("mean", "F")] - both[("mean", "R")], weights=w) if len(both) else np.nan
    nxt = d[d.season == s + 1]; gn = nxt[nxt.game_type == "R"].groupby("pitcher_id").pt.agg(["mean", "size"]); gn = gn[gn["size"] >= 30]
    gf = cur[cur.game_type == "F"].groupby("pitcher_id").pt.agg(["mean", "size"]); gf = gf[gf["size"] >= 30]
    j = gf.join(gn, lsuffix="_F", rsuffix="_R", how="inner"); adj = np.average(j.mean_F - j.mean_R, weights=np.minimum(j.size_F, j.size_R)) if len(j) else np.nan
    from _common import D
    print(f"{s:<6}{lg:>+10.3f}{wi:>+22.3f}{len(both):>6}{adj:>+16.3f}{len(j):>5}")
print("\n리그 F−R 는 구성(누가 F 에 있나)+환경, within 은 환경만(같은 투수). 둘의 차 = 선택(F 에 보내진 투수가 약함) 성분.")
print("현행 adj: lvl_diff=−2.222 · γ=−0.6 → 유효 격차 −0.889 (2025 베팅). within 2023·2024 값과 비교.")
