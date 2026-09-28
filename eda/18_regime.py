"""EDA 18 — 체제(regime) 인식. 13 에서 season 열을 넣은 상황-only LGB 만 τ +0.25 (없으면 −0.01) → F 경기의 2023 체제변화
(반대투구 4.9%→30%, Failure 42%→58%) 가 τ 의 핵심 레버인지 확인. 10 의 ⓒ표(전 시즌 평균)가 2023/24 에서 τ −0.3/−0.23 인 이유이기도 하다.
※ 2023 폴드는 체제변화가 '평가시즌 안에서' 일어나 어떤 방법도 못 맞춘다. 2025 는 2023–24 체제가 이어진다고 가정한다."""
from _common import *
d = load()
head("1. cnt×game_type 표를 '최근 k 시즌' 으로만 → τ / Pitch  (2024 폴드, 비교로 2022 폴드)")
for Y in (2024, 2022):
    past, ev = fold(d, Y); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy(); pri = prior(past)
    print(f"\n  [{Y}]")
    for k in (1, 2, 3, 99):
        pk = past[past.season >= Y-k]; score(table_probs(pk, ev, ["cnt", "game_type"], 50, prior(pk)), pid, y, f"  최근 {k}시즌 표 (행 {len(pk):,})")
    # 하이브리드: R 은 전 시즌, F 는 최근 2시즌
    pr = past[past.game_type == "R"]; pf = past[(past.game_type == "F") & (past.season >= Y-2)]
    score(table_probs(pd.concat([pr, pf]), ev, ["cnt", "game_type"], 50, prior(past[past.season >= Y-2])), pid, y, "  R=전체, F=최근2 하이브리드")
    sub("game_type 단독 표 (투수 정보 0) — 최근 k")
    for k in (1, 2, 99):
        pk = past[past.season >= Y-k]; score(table_probs(pk, ev, ["game_type"], 50, prior(pk)), pid, y, f"  game_type 표, 최근 {k}")

head("2. 투수 EB + 체제 인식 상황 → τ (2024 폴드)  — 종민 06b ⓑ '잔차실력 + 상황' 이 왜 나빴는지 재검")
past, ev = fold(d, 2024); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy(); pri = prior(past); pri2 = prior(past[past.season >= 2022])
P_p, nh = entity_probs(past, ev, "pitcher_id", 200, pri)
score(P_p, pid, y, "투수 EB(200) 만")
for k in (1, 2, 99):
    pk = past[past.season >= 2024-k]
    P_c = table_probs(pk, ev, ["cnt", "game_type"], 50, prior(pk))
    score(combine(P_c, P_p, pri=pri), pid, y, f"투수 EB + cnt×gt 표(최근 {k})")
sub("잔차 실력(종민 07) 을 '최근 2시즌 상황표' 로 보정하면?")
def resid_skill(past_, ctx_src, alpha=200):
    ctab = ctx_src.groupby(["cnt", "game_type"]).pt.mean().rename("c").reset_index()
    p = past_[["pitcher_id", "cnt", "game_type", "pt"]].merge(ctab, on=["cnt", "game_type"], how="left")
    g = pd.DataFrame({"pid": p.pitcher_id.to_numpy(), "r": (p.pt - p.c).to_numpy()}).groupby("pid").r.agg(["sum", "size"])
    return g["sum"]/(g["size"] + alpha)
pg = ev.groupby("pitcher_id").pt.agg(["size", "mean"]); el = pg[pg["size"] >= 50]; act = el["mean"].round(10)
def tau_of(s): return kendalltau(act, pd.Series(s).reindex(el.index).fillna(0).round(10), variant="b").statistic
ctab_recent = past[past.season >= 2022]; ctab_1 = past[past.season >= 2023]
print(f"    잔차EB(전체 상황표)            τ {tau_of(resid_skill(past, past)):+.4f}")
print(f"    잔차EB(최근2 상황표)           τ {tau_of(resid_skill(past, ctab_recent)):+.4f}")
print(f"    잔차EB(최근1 상황표)           τ {tau_of(resid_skill(past, ctab_1)):+.4f}")
print(f"    잔차EB(최근1 상황표, α=100)    τ {tau_of(resid_skill(past, ctab_1, 100)):+.4f}")
print(f"    잔차EB(이력 최근2·상황표 최근2)  τ {tau_of(resid_skill(past[past.season>=2022], ctab_recent)):+.4f}")
# 잔차실력 + 2024 상황(최근2 표) 가산
ctab = ctab_recent.groupby(["cnt", "game_type"]).pt.mean().rename("c").reset_index()
ev_c = ev[["pitcher_id", "cnt", "game_type"]].merge(ctab, on=["cnt", "game_type"], how="left").groupby("pitcher_id").c.mean()
sk = resid_skill(past, ctab_recent)
print(f"    잔차EB(최근2표) + 2024 상황평균(최근2표)   τ {tau_of(sk.reindex(el.index).fillna(0) + ev_c.reindex(el.index)):+.4f}   ← 종민 06b ⓑ 의 체제인식 버전")
print(f"    2024 상황평균(최근2표) 만                τ {tau_of(ev_c.reindex(el.index)):+.4f}")

head("3. F 경기와 투수 — 신인은 어디서 던지나, F 성적은 R 성적과 같은 실력인가 (2023–2024)")
r = d[d.season >= 2023]
g = r.groupby("pitcher_id").agg(n=("pt", "size"), f_share=("game_type", lambda s: (s == "F").mean()), pt=("pt", "mean"))
past23 = d[d.season < 2023].groupby("pitcher_id").size()
g["nhist"] = g.index.map(past23).fillna(0); g = g[g.n >= 50]
for nm, m in (("신인(2023 이전 이력 0)", g.nhist == 0), ("이력 1-500", (g.nhist > 0) & (g.nhist <= 500)), ("이력 500+", g.nhist > 500)):
    print(f"    {nm:20s} 투수 {m.sum():3d} · F 비중 중위 {g[m].f_share.median()*100:5.1f}% · F전용(>99%) {int((g[m].f_share > .99).sum()):3d}명 · 실제 pt 평균 {g[m].pt.mean():.2f}")
sub("같은 투수의 F 성적 vs R 성적 (양쪽 50투구+, 2023–24)")
fr = r.groupby(["pitcher_id", "game_type"]).pt.agg(["size", "mean"]).unstack("game_type")
ok = fr[(fr[("size", "F")] >= 50) & (fr[("size", "R")] >= 50)]
print(f"    투수 {len(ok)}명 · F 평균 {ok[('mean','F')].mean():.2f} · R 평균 {ok[('mean','R')].mean():.2f} · 상관 r {np.corrcoef(ok[('mean','F')], ok[('mean','R')])[0,1]:+.3f}")
sub("2024 평가 투수(50+)의 F 비중 vs 실제 pt — F 비중 자체가 순위 신호인가")
g24 = ev.groupby("pitcher_id").agg(n=("pt", "size"), f_share=("game_type", lambda s: (s == "F").mean()), pt=("pt", "mean")); g24 = g24[g24.n >= 50]
print(f"    τ(−F비중, 실제pt) = {kendalltau(g24.pt.round(10), (-g24.f_share).round(10), variant='b').statistic:+.4f}   (F 비중 높을수록 실제 pt 낮음 → 양수)")
print(f"\n{tick()} done")
