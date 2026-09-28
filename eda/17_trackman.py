"""EDA 17 — Trackman 은 평가셋에 없다. 하지만 '투수의 과거 Trackman 프로필'(구종 믹스·구속·회전·릴리스 일관성)은 사전 정보다.
(1) 프로필 지표가 투수의 실제 클래스 비율과 상관있나 (투수 단위, 2024), (2) 15 의 LGB 에 프로필을 넣으면 Pitch/τ 가 오르나."""
from _feat import *
from _feat import _resid_skill
d = load()
tm = pd.read_csv(D + "train_trackman.csv", dtype={"row_id": str}, usecols=["row_id", "season", "pitcher_id", "pitch_type_group", "rel_speed", "spin_rate",
     "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed", "auto_pitch_type"])
tm = tm[tm.pitcher_id.notna() & tm.pitcher_id.str.startswith("P_")].copy()
print(f"{tick()} trackman {len(tm):,}행 (P_ 투수) · row_id 연결 {tm.row_id.notna().mean()*100:.1f}%")
for g_ in ("fastball", "breaking", "offspeed"): tm[f"mix_{g_}"] = (tm.pitch_type_group == g_).astype(float)
NUM = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]
def profile(past_tm):
    g = past_tm.groupby("pitcher_id")
    f = g[NUM].mean().add_prefix("tm_"); f["tm_n"] = g.size()
    for c in ("rel_height", "rel_side", "rel_speed", "extension"): f[f"tm_sd_{c}"] = g[c].std()
    for c in ("fastball", "breaking", "offspeed"): f[f"tm_mix_{c}"] = g[f"mix_{c}"].mean()
    # 직구 구속·회전 (구종 믹스 차이 제거)
    fb = past_tm[past_tm.pitch_type_group == "fastball"].groupby("pitcher_id")[["rel_speed", "spin_rate", "induced_vert_break"]].mean().add_prefix("tm_fb_")
    return f.join(fb)

head("1. 투수 단위 상관 (2024 평가 50+ 투수 · 과거 Trackman 200구+) — Spearman ρ vs 2024 실제 클래스 비율")
past, ev = fold(d, 2024); prof = profile(tm[tm.season < 2024]); prof = prof[prof.tm_n >= 200]
Y1 = onehot(ev.y.to_numpy()); e = ev.assign(S=Y1[:, 0], H=Y1[:, 1], F=Y1[:, 2])
act = e.groupby("pitcher_id").agg(n=("pt", "size"), pt=("pt", "mean"), S=("S", "mean"), H=("H", "mean"), F=("F", "mean")); act = act[act.n >= 50]
j = act.join(prof, how="inner"); print(f"  교집합 투수 {len(j)}명")
# 비교 기준: 과거 EB 실력
skl = _resid_skill(past, act.reset_index())
j["ref_resid_skill"] = pd.Series(skl, index=act.index).reindex(j.index)
rows = []
for c in [c for c in j.columns if c.startswith("tm_") and c != "tm_n"] + ["ref_resid_skill"]:
    rows.append(dict(feat=c, **{k: j[c].corr(j[k], method="spearman") for k in ("pt", "S", "H", "F")}))
R = pd.DataFrame(rows).set_index("feat"); print(R.round(3).sort_values("pt", key=abs, ascending=False).to_string())
sub("프로필 → 실력 부분상관: EB 실력을 통제한 뒤 남는 상관 (pt 잔차 vs 프로필)")
res = j.pt - np.polyval(np.polyfit(j.ref_resid_skill, j.pt, 1), j.ref_resid_skill)
for c in ("tm_sd_rel_side", "tm_sd_rel_height", "tm_mix_fastball", "tm_rel_speed", "tm_spin_rate", "tm_extension", "tm_fb_rel_speed"):
    print(f"    {c:20s} ρ(잔차) {res.corr(j[c], method='spearman'):+.3f}")

head("2. LGB 에 투수 Trackman 프로필 추가 (확장창: season<Y 의 Trackman 만) — 2024 폴드")
def extra(past_, cur_):
    Yc = int(cur_.season.iloc[0]); pr = profile(tm[tm.season < Yc]); pr = pr.drop(columns=["tm_n"]).join(profile(tm[tm.season < Yc])[["tm_n"]])
    return cur_[["pitcher_id"]].join(pr, on="pitcher_id").drop(columns=["pitcher_id"])
Xtr, ytr, Xev, ev = build(d, [2020, 2021, 2022, 2023], 2024, extra=extra); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy()
Xtr["season"] = d.loc[Xtr.index, "season"].to_numpy(); Xev["season"] = d.loc[Xev.index, "season"].to_numpy()   # 19: season 열이 τ +0.12
base_cols = [c for c in Xev.columns if not c.startswith("tm_")]
m0 = fit_lgb(Xtr[base_cols], ytr); score(m0.predict(Xev[base_cols], num_iteration=m0.best_iteration), pid, y, "기준 (19 의 +season 모형)")
m1 = fit_lgb(Xtr, ytr); score(m1.predict(Xev, num_iteration=m1.best_iteration), pid, y, "+ Trackman 프로필")
print("  gain% (tm_ 만):", {k: v for k, v in importance(m1, list(Xev.columns), 40).items() if k.startswith("tm_")})
print(f"\n{tick()} done")
