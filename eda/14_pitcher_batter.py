"""EDA 14 — 투수·타자 이력이 Brier 에 얼마나 기여하나. 종민 06b/07 은 τ 기준 α·최근성을 봤다; 여기서는 Pitch 와 Total 로.
(1) 투수 클래스비율 EB α, (2) 최근성 반감기, (3) 투수×카운트 상호작용, (4) 이력 크기별 이득 분해, (5) 타자 EB, (6) 팀 EB (신인 대체)."""
from _common import *
d = load()
def run(Y, verbose=True):
    past, ev = fold(d, Y); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy(); pri = prior(past); n = len(ev)
    P_ctx = table_probs(past, ev, ["cnt", "game_type", "matchup", "inn"], 50, pri)
    out = {}
    out["ctx"] = score(P_ctx, pid, y, "기준: 상황 4원 표", quiet=not verbose)
    if verbose: sub("(1) 투수 EB α — 상황표 + 투수 편차")
    for a in (20, 50, 100, 200, 500, 1000, 2000):
        P_p, nh = entity_probs(past, ev, "pitcher_id", a, pri)
        out[f"pit_a{a}"] = score(combine(P_ctx, P_p, pri=pri), pid, y, f"  α={a}", quiet=not verbose)
    if verbose: sub("(2) 최근성 반감기 h(시즌) — 투수 EB α=200")
    age = (Y - past.season).to_numpy(float)
    for h in (0.5, 1, 2, 3, 99):
        w = 0.5**(age/h); P_p, _ = entity_probs(past, ev, "pitcher_id", 200, pri, w)
        out[f"pit_h{h}"] = score(combine(P_ctx, P_p, pri=pri), pid, y, f"  h={h}", quiet=not verbose)
    if verbose: sub("(3) 투수 × 카운트 상호작용 — 투수별 카운트 반응이 다른가 (투수 EB 위에 가산)")
    P_p, nh = entity_probs(past, ev, "pitcher_id", 200, pri)
    base_pp = combine(P_ctx, P_p, pri=pri)
    for keys, a in ([["pitcher_id", "strikes_before"], 100], [["pitcher_id", "cnt"], 100], [["pitcher_id", "batter_hand"], 100], [["pitcher_id", "game_type"], 100]):
        # 셀 EB 는 '그 투수의 전체 비율' 로 수축시켜야 하지만 근사로 prior 수축 후 투수 편차와 가산
        P_pc = table_probs(past, ev, keys, a, pri)
        out["×".join(keys)] = score(combine(P_ctx, P_pc, pri=pri), pid, y, f"  상황표 + {'×'.join(keys)} (α={a})", quiet=not verbose)
    if verbose:
        sub("(4) 이력 크기별 — 상황표 vs 상황표+투수EB(200) 의 Brier, 행 비중")
        b = pd.cut(nh, [-1, 0, 100, 500, 2000, 99999], labels=["0(신인)", "1-100", "101-500", "501-2000", "2000+"])
        for lv in b.categories:
            mk = np.asarray(b == lv)
            if mk.sum() == 0: continue
            print(f"    {lv:9s} 행 {mk.mean()*100:5.1f}%  Brier 상황 {brier(P_ctx[mk], y[mk]):.4f} → +투수 {brier(base_pp[mk], y[mk]):.4f}  (Δ {brier(P_ctx[mk], y[mk])-brier(base_pp[mk], y[mk]):+.4f})")
        sub("(5) 타자 EB α (상황표 + 투수200 + 타자)")
    for a in (100, 300, 1000, 3000):
        P_b, _ = entity_probs(past, ev, "batter_id", a, pri)
        out[f"bat_a{a}"] = score(combine(P_ctx, P_p, P_b, pri=pri), pid, y, f"  타자 α={a}", quiet=not verbose)
    if verbose: sub("(6) 팀 EB — 신인 투수의 대체 정보가 되나 (상황표 + 투수200 + 투수팀)")
    P_t, _ = entity_probs(past, ev, "pitcher_team_id", 500, pri)
    out["team"] = score(combine(P_ctx, P_p, P_t, pri=pri), pid, y, "  +투수팀 α=500", quiet=not verbose)
    nh0 = nh == 0
    if verbose and nh0.sum() > 0:
        print(f"    신인 행({nh0.mean()*100:.1f}%) Brier: 상황 {brier(P_ctx[nh0], y[nh0]):.4f} · +팀 {brier(combine(P_ctx, P_t, pri=pri)[nh0], y[nh0]):.4f}")
    return out
head("2024 폴드 (상세)"); R24 = run(2024)
head("2022 / 2023 폴드 (요약) → 3폴드 평균 Total / Pitch / τ")
R = {2024: R24, 2023: run(2023, False), 2022: run(2022, False)}
A = pd.DataFrame({k: pd.DataFrame(v, index=["Total", "Pitch", "tau"]).T.stack() for k, v in R.items()})
A["mean"] = A.mean(1); print(A.round(4).to_string())
print(f"\n{tick()} done")
