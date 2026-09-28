"""EDA 15 — LightGBM 기준선 (상황 + 확장창 이력 인코딩). 2024 폴드에서 Total/Pitch/τ, 신인 τ, 피처 중요도.
변형: 이력 없음 / ID 범주형 직접 / 최근성 인코딩 추가 / 2019 포함.  2024 확률을 out/ 에 저장 → 16 에서 사용."""
from _feat import *
d = load(); Y = 2024
head("A. 상황 + 투수·타자 EB 인코딩 (학습 2020–2023, 평가 2024)")
Xtr, ytr, Xev, ev = build(d, [2020, 2021, 2022, 2023], Y)
y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy(); print(f"{tick()} X {Xtr.shape} → 평가 {Xev.shape}  열: {list(Xev.columns)}")
m = fit_lgb(Xtr, ytr); P = m.predict(Xev, num_iteration=m.best_iteration); print(f"  best_iter {m.best_iteration}")
tot, ps, tau = score(P, pid, y, "★ LGB 상황+이력")
print("  gain%:", importance(m, list(Xev.columns)))
# 신인(이력 0) 투수 사이의 τ — 종민 08 의 배치 축을 GBM 이 스스로 찾는지
nh = Xev["p_n"].to_numpy() == 0
g = pd.DataFrame({"pid": pid, "pred": P @ PT, "act": PT[y.astype(int)], "nh": nh}).groupby("pid").agg(n=("act", "size"), pred=("pred", "mean"), act=("act", "mean"), nh=("nh", "first"))
el = g[(g.n >= 50)]; r = el[el.nh]
print(f"  50+ 투수 {len(el)} · 신인 {len(r)} · 신인끼리 τ {kendalltau(r.act.round(10), r.pred.round(10), variant='b').statistic:+.4f}"
      f" · 비신인끼리 τ {kendalltau(el[~el.nh].act.round(10), el[~el.nh].pred.round(10), variant='b').statistic:+.4f}")
np.savez(OUT + "p15_lgb_2024.npz", P=P, y=y, pid=pid, row_id=ev.row_id.to_numpy(), p_n=Xev["p_n"].to_numpy(), p_skill=Xev["p_skill"].to_numpy())

head("B. 변형")
cols_noh = CTX
m2 = fit_lgb(Xtr[cols_noh], ytr); score(m2.predict(Xev[cols_noh], num_iteration=m2.best_iteration), pid, y, "상황만 (이력 없음)")
Xtr3, ytr3, Xev3, ev3 = build(d, [2020, 2021, 2022, 2023], Y, recency_h=2)
m3 = fit_lgb(Xtr3, ytr3); score(m3.predict(Xev3, num_iteration=m3.best_iteration), pid, y, "+ 최근성(h=2) 인코딩")
Xtr4, ytr4, Xev4, ev4 = build(d, [2019, 2020, 2021, 2022, 2023], Y)
m4 = fit_lgb(Xtr4, ytr4); score(m4.predict(Xev4, num_iteration=m4.best_iteration), pid, y, "학습에 2019 포함 (가짜 신인 20%)")
sub("pitcher_id 를 LightGBM 범주형으로 직접 투입 (+인코딩) — 과적합 여부")
Xtr5, Xev5 = Xtr.copy(), Xev.copy()
cats = pd.Categorical(pd.concat([Xtr5.index.to_series().map(d.pitcher_id), Xev5.index.to_series().map(d.pitcher_id)])).categories
Xtr5["pid_cat"] = pd.Categorical(d.loc[Xtr5.index, "pitcher_id"], categories=cats); Xev5["pid_cat"] = pd.Categorical(d.loc[Xev5.index, "pitcher_id"], categories=cats)
m5 = fit_lgb(Xtr5, ytr, dict(cat_smooth=50, min_data_per_group=200)); score(m5.predict(Xev5, num_iteration=m5.best_iteration), pid, y, "+ pitcher_id 범주형")
sub("하이퍼파라미터 감도 (num_leaves / lr)")
for nl, lr in ((15, 0.05), (31, 0.05), (127, 0.05), (63, 0.02)):
    mm = fit_lgb(Xtr, ytr, dict(num_leaves=nl, learning_rate=lr)); score(mm.predict(Xev, num_iteration=mm.best_iteration), pid, y, f"leaves={nl} lr={lr} it={mm.best_iteration}")
print(f"\n{tick()} done")
