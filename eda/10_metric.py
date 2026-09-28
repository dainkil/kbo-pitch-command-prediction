"""EDA 10 — 공식 지표 구현 + 바닥선/천장. Total = 0.7·Pitch(100·(1−Brier/2)) + 0.3·Player(50·(1+τ_b)).
종민 04~07 은 τ(30%) 만 봤다. 여기서는 Brier(70%) 를 포함한 Total 로 '점수가 어디에 있는지' 잰다. 3폴드(2022/23/24)."""
from _common import *
d = load(); print(f"{tick()} 라벨 행 {len(d):,}")

rows = []
for Y in (2022, 2023, 2024):
    past, ev = fold(d, Y); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy(); pri = prior(past); n = len(ev)
    head(f"평가 시즌 {Y}   평가 {n:,}행 · 과거 {len(past):,}행   과거 prior {np.round(pri,4)}  /  {Y} 실제 {np.round(prior(ev),4)}")
    R = {}
    R["a_const"]  = score(const(pri, n), pid, y, "ⓐ 상수 prior(과거 전체)")
    R["b_oracle"] = score(const(prior(ev), n), pid, y, "ⓑ 오라클 prior(평가시즌 자체) — 드리프트 비용")
    P_ctx = table_probs(past, ev, ["cnt", "game_type"], 50, pri)
    R["c_ctx"]    = score(P_ctx, pid, y, "ⓒ 카운트12×game_type 표")
    P_ctx2 = table_probs(past, ev, ["cnt", "game_type", "matchup", "inn"], 50, pri)
    R["c2_ctx"]   = score(P_ctx2, pid, y, "ⓒ' + 손매치업 + 이닝 (4원 표)")
    for a in (50, 200, 1000):
        P_p, nh = entity_probs(past, ev, "pitcher_id", a, pri)
        R[f"d_pit{a}"] = score(P_p, pid, y, f"ⓓ 투수 클래스비율 EB α={a}")
        R[f"e_ctx_pit{a}"] = score(combine(P_ctx, P_p, pri=pri), pid, y, f"ⓔ ⓒ + ⓓ(α={a})")
    P_p, nh = entity_probs(past, ev, "pitcher_id", 200, pri)
    P_b, _  = entity_probs(past, ev, "batter_id", 200, pri)
    R["f_ctx_pit_bat"] = score(combine(P_ctx, P_p, P_b, pri=pri), pid, y, "ⓕ ⓒ + 투수(200) + 타자(200)")
    R["f2"] = score(combine(P_ctx2, P_p, P_b, pri=pri), pid, y, "ⓕ' ⓒ' + 투수(200) + 타자(200)")

    sub("천장 참고 (평가시즌 자체 정보를 쓰는 in-sample 오라클 — 도달 불가, 정보량의 상한만 본다)")
    # 투수별 평가시즌 클래스 비율을 '안다'고 가정
    P_or = table_probs(ev, ev, ["pitcher_id"], 0, prior(ev))
    R["g_or_pit"] = score(P_or, pid, y, "ⓖ 오라클: 투수별 평가시즌 클래스비율")
    P_or2 = table_probs(ev, ev, ["pitcher_id", "cnt"], 0, prior(ev))
    R["h_or_pit_cnt"] = score(P_or2, pid, y, "ⓗ 오라클: 투수×카운트12 (과적합 포함)")
    sub("점수 감도: Brier 0.01 = Pitch 0.5 = Total 0.35  ·  τ 0.01 = Player 0.5 = Total 0.15")
    rows.append(pd.DataFrame(R, index=["Total", "Pitch", "tau"]).T.assign(Y=Y))

head("3폴드 평균 (Total / Pitch / τ)")
A = pd.concat(rows).groupby(level=0).agg(Total=("Total","mean"), Pitch=("Pitch","mean"), tau=("tau","mean"))
print(A.round(4).to_string())
print(f"\n{tick()} done")
