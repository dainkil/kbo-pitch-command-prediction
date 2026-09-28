"""EDA 19 — 15 의 LGB 에 체제(regime) 정보를 넣으면 τ 가 얼마나 오르나. 13: 상황-only LGB 가 season 열 유무로 τ +0.25 / −0.01.
변형: season 열 / regime 플래그(season≥2023) / 최근 2시즌만 학습 / +투수×game_type EB / +최근성 h=2 / 전부 결합. 2024 폴드.
※ 트리에서 season=2025 는 '≥2024 분기' 로 떨어져 2024 와 같은 잎을 탄다 → 외삽이 아니라 '최신 체제 유지' 가정과 동치."""
from _feat import *
from _feat import _enc
d = load(); Y = 2024
def pgt(past_, cur_):
    pri = prior(past_); P = table_probs(past_, cur_, ["pitcher_id", "game_type"], 100, pri)
    return pd.DataFrame(P, columns=["pgt_sh", "pgt_he", "pgt_fa"], index=cur_.index)
def run(label, seasons, season_col=False, regime=False, recency=None, extra=None, params=None):
    Xtr, ytr, Xev, ev = build(d, seasons, Y, recency_h=recency, extra=extra)
    if season_col: Xtr["season"] = d.loc[Xtr.index, "season"].to_numpy(); Xev["season"] = d.loc[Xev.index, "season"].to_numpy()
    if regime:     Xtr["regime"] = (d.loc[Xtr.index, "season"] >= 2023).astype(int).to_numpy(); Xev["regime"] = 1
    y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy()
    m = fit_lgb(Xtr, ytr, params); P = m.predict(Xev, num_iteration=m.best_iteration)
    tot, ps, tau = score(P, pid, y, f"{label} (it={m.best_iteration})")
    return P, m, Xev, ev
head("2024 폴드 — 체제 정보 변형")
P0, *_ = run("기준 15A (season 없음)", [2020, 2021, 2022, 2023])
P1, m1, Xev1, ev = run("+ season 열", [2020, 2021, 2022, 2023], season_col=True)
print("   gain%:", importance(m1, list(Xev1.columns), 10))
P2, *_ = run("+ regime 플래그(≥2023)", [2020, 2021, 2022, 2023], regime=True)
P3, *_ = run("최근 2시즌(2022–23)만 학습", [2022, 2023])
P3b, *_ = run("최근 2시즌 + season 열", [2022, 2023], season_col=True)
P4, *_ = run("+ season + 투수×game_type EB", [2020, 2021, 2022, 2023], season_col=True, extra=pgt)
P5, m5, Xev5, ev = run("+ season + 투수×gt EB + 최근성 h=2", [2020, 2021, 2022, 2023], season_col=True, recency=2, extra=pgt)
print("   gain%:", importance(m5, list(Xev5.columns), 15))
y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy()
sub("앙상블 (확률 평균)")
score((P1 + P5)/2, pid, y, "season + season·pgt·h2 평균")
score((P1 + P3b + P5)/3, pid, y, "3개 평균")
sub("신인 / 비신인 τ 분해 (P5)")
nh = Xev5["p_n"].to_numpy() == 0
g = pd.DataFrame({"pid": pid, "pred": P5 @ PT, "act": PT[y.astype(int)], "nh": nh}).groupby("pid").agg(n=("act", "size"), pred=("pred", "mean"), act=("act", "mean"), nh=("nh", "first"))
el = g[g.n >= 50]
for nm, mk in (("신인", el.nh), ("비신인", ~el.nh)):
    q = el[mk]; print(f"    {nm} {len(q)}명  τ {kendalltau(q.act.round(10), q.pred.round(10), variant='b').statistic:+.4f}")
np.savez(OUT + "p19_lgb_2024.npz", P=P5, y=y, pid=pid, row_id=ev.row_id.to_numpy(), p_n=Xev5["p_n"].to_numpy(), p_skill=Xev5["p_skill"].to_numpy())
print(f"\n{tick()} done")
