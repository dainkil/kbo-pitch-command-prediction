"""EDA 16 — Brier 와 τ 의 관계. τ 는 '투수별 예측 점수 평균의 순위' 만 본다 → 모형 자체 신호를 k 배 키워도 순위는 불변(τ 그대로, Brier 만 악화).
따라서 문제는 '더 좋은 투수 순위 신호(r)를 확률에 얼마나 섞을 때 Brier 손실 < τ 이득인가'. 15 의 2024 확률로 곡선을 그린다.
또 온도 보정 / prior 이동(드리프트 대응) 이 Brier 에 주는 효과를 본다."""
from _common import *
z = np.load(OUT + "p15_lgb_2024.npz", allow_pickle=True); P, y, pid, p_n, p_skill = z["P"], z["y"], z["pid"], z["p_n"], z["p_skill"]
d = load(); past, ev = fold(d, 2024)
head("0. 기준 (15 의 LGB 2024)"); score(P, pid, y, "LGB")

head("1. 자기 신호 확대 — 순위 불변 확인 (τ 그대로, Brier 만 나빠짐)")
s = pd.Series(P @ PT).groupby(pid).transform("mean").to_numpy(); sc = (s - s.mean())/100
for k in (0.5, 1, 2):
    Pk = np.clip(P + k*sc[:, None]*np.array([1, 0, -1]), 1e-4, None); Pk /= Pk.sum(1, keepdims=True); score(Pk, pid, y, f"  k={k}")

head("2. 외부 순위 신호 r 주입 — r = 상황보정 잔차 EB 실력(종민 07 최고, α=200) [+ 신인은 배치축(li3·sd5) 보정]")
r = p_skill.copy()                       # 이력 없는 투수는 0
# 신인 tie-break: 종민 08 — li3·점수차 축의 '그 상황에 등판한 투수들의 시즌 평균 pt' 표 (past 에서, 50투구+ 투수시즌)
ps = past.groupby(["pitcher_id", "season"]).pt.agg(["size", "mean"]); ps = ps[ps["size"] >= 50]["mean"].rename("skill")
Pp = past.merge(ps, left_on=["pitcher_id", "season"], right_index=True, how="inner")
tab = Pp.groupby(["li3", "sd5"]).skill.mean(); mu_s = Pp.skill.mean()
dep = ev[["li3", "sd5"]].merge(tab.rename("v").reset_index(), on=["li3", "sd5"], how="left").v.fillna(mu_s).to_numpy() - mu_s
dep_p = pd.Series(dep).groupby(pid).transform("mean").to_numpy()
rook = p_n == 0
def curve(r_row, label):
    print(f"\n  {label}")
    rz = (r_row - r_row.mean())/(r_row.std() + 1e-9)
    for eps in (0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08):
        Pe = np.clip(P + eps*rz[:, None]*np.array([1, 0, -1]), 1e-4, None); Pe /= Pe.sum(1, keepdims=True); score(Pe, pid, y, f"    ε={eps}")
curve(r, "r = 잔차 EB 실력 (신인 0)")
r2 = r.copy(); r2[rook] = dep_p[rook]*0.5; curve(r2, "r = 잔차 EB 실력 + 신인 배치축(li3×sd5) tie-break")
sub("신호 자체의 τ (확률 무관, 투수 순위만)")
for nm, rr in (("LGB 자체", s), ("잔차EB", r), ("잔차EB+신인배치", r2)):
    print(f"    {nm:12s} τ {player_score(np.tile([0.,0.,0.],(len(y),1)) + 0, pid, y)[1] if False else kendalltau(*[pd.Series(v).groupby(pid).mean().loc[lambda q: pd.Series(y).groupby(pid).size().reindex(q.index)>=50] for v in (PT[y.astype(int)], rr)], variant='b').statistic:+.4f}")

head("3. 온도 보정 T (logit·1/T) — LGB 확률의 캘리브레이션")
L = np.log(np.clip(P, 1e-9, 1))
for T in (0.8, 0.9, 1.0, 1.1, 1.25, 1.5):
    Q = np.exp(L/T); Q /= Q.sum(1, keepdims=True); score(Q, pid, y, f"  T={T}")
head("4. prior 이동 — 2025 드리프트 대응 시뮬: 확률에 (목표prior/학습prior) 를 곱해 재정규화")
pri_tr = prior(past[past.season >= 2020]); pri_ev = prior(ev); print(f"  학습 prior {np.round(pri_tr,4)} · 2024 실제 {np.round(pri_ev,4)} · 모형 평균 {np.round(P.mean(0),4)}")
for nm, tgt in (("2023 prior", prior(past[past.season == 2023])), ("오라클 2024", pri_ev)):
    Q = P*(tgt/pri_tr); Q /= Q.sum(1, keepdims=True); score(Q, pid, y, f"  → {nm}")
print(f"\n{tick()} done")
