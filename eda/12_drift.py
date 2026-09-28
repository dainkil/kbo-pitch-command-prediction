"""EDA 12 — 시간 드리프트와 지속성. (1) 클래스 비율의 시즌/월/game_type 추세, (2) 2025 prior 를 무엇으로 잡아야 Brier 손실이 작은가,
(3) 투수별 클래스 비율의 해간 지속성 — 어느 클래스가 '투수 고유' 인가 (Brier 에서 투수 EB 가 벌 수 있는 근거),
(4) Failure 의 분해: 반대투구(REVERSE) vs 존 밖 — 2023 F 경기의 체제 변화."""
from _common import *
d = load(); Y1 = onehot(d.y.to_numpy())
for i, c in enumerate(CLASSES): d[c] = Y1[:, i]

head("1. 클래스 비율 추세")
print((d.groupby("season")[CLASSES].mean()*100).round(2).to_string())
sub("시즌 × game_type")
print((d.groupby(["season", "game_type"])[CLASSES].mean()*100).round(2).unstack("game_type").to_string())
sub("시즌 × 월 (Shadow% / Failure%) — 시즌 초/말 효과")
print((d.pivot_table(index="season", columns="game_month", values="Failure", aggfunc="mean")*100).round(1).to_string())
sub("시즌 × 카운트12 Failure% — 카운트 효과가 시즌마다 안정적인가")
print((d.pivot_table(index="cnt", columns="season", values="Failure", aggfunc="mean")*100).round(1).to_string())

head("2. 2025 prior 선택 — 2024 를 표적으로 후보 prior 의 Brier 비용 (3폴드)")
def lin_extrap(past, Y):
    g = past.groupby("season")[CLASSES].mean(); x = g.index.to_numpy(float)
    out = np.array([np.polyval(np.polyfit(x, g[c].to_numpy(), 1), Y) for c in CLASSES]); out = np.clip(out, 1e-3, None)
    return out/out.sum()
for Y in (2022, 2023, 2024):
    past, ev = fold(d, Y); y = ev.y.to_numpy(); n = len(ev); pid = ev.pitcher_id.to_numpy()
    cands = {"과거 전체": prior(past), "직전 1시즌": prior(past[past.season == Y-1]), "직전 2시즌": prior(past[past.season >= Y-2]),
             "직전 3시즌": prior(past[past.season >= Y-3]), "선형 외삽": lin_extrap(past, Y), "오라클(자체)": prior(ev)}
    print(f"\n  [{Y}]")
    for k, p in cands.items():
        print(f"   {k:12s} prior {np.round(p,4)}  Brier {brier(const(p,n), y):.5f}  Pitch {pitch_score(const(p,n), y):.4f}")

head("3. 투수별 클래스 비율의 해간 지속성 (양 시즌 100투구+ 투수, Pearson r / Spearman ρ)")
for Y in (2022, 2023, 2024):
    a = d[d.season == Y-1].groupby("pitcher_id")[CLASSES + ["pt"]].agg(["mean", "size"])
    b = d[d.season == Y].groupby("pitcher_id")[CLASSES + ["pt"]].agg(["mean", "size"])
    ok = a.index[(a[("pt", "size")] >= 100)].intersection(b.index[(b[("pt", "size")] >= 100)])
    line = f"  {Y-1}→{Y} n={len(ok):3d}  "
    for c in CLASSES + ["pt"]:
        x, z = a.loc[ok, (c, "mean")], b.loc[ok, (c, "mean")]
        line += f"{c}: r {np.corrcoef(x, z)[0,1]:+.3f} ρ {x.corr(z, method='spearman'):+.3f}   "
    print(line)
sub("과거 전체(≤Y-1, 200투구+) → Y (100투구+) 지속성")
for Y in (2022, 2023, 2024):
    past, ev = fold(d, Y)
    a = past.groupby("pitcher_id")[CLASSES + ["pt"]].agg(["mean", "size"]); b = ev.groupby("pitcher_id")[CLASSES + ["pt"]].agg(["mean", "size"])
    ok = a.index[(a[("pt", "size")] >= 200)].intersection(b.index[(b[("pt", "size")] >= 100)])
    line = f"  →{Y} n={len(ok):3d}  "
    for c in CLASSES + ["pt"]:
        x, z = a.loc[ok, (c, "mean")], b.loc[ok, (c, "mean")]
        line += f"{c}: r {np.corrcoef(x, z)[0,1]:+.3f}   "
    print(line)
sub("투수간 분산 vs 투구 잡음 — 클래스별 '참 sd' (2024, 50+)  ← 투수 EB 가 벌 수 있는 크기")
ev = d[d.season == 2024]; g = ev.groupby("pitcher_id"); n = g.size(); n = n[n >= 50]
for c in CLASSES:
    m = g[c].mean().reindex(n.index); w = n.to_numpy(float); mu = (w*m).sum()/w.sum()
    vb = (w*(m-mu)**2).sum()/w.sum(); vw = (g[c].var().reindex(n.index)/w).mean(); sb = np.sqrt(max(vb-vw, 0))
    print(f"  {c:8s} 평균 {mu:.4f} · 투수간 관측 sd {np.sqrt(vb):.4f} → 참 sd {sb:.4f}  (상대 {sb/mu*100:.1f}%)")

head("4. Failure 분해 — 반대투구(REVERSE_YN) vs 그 외")
d["fail_rev"] = d.Failure * d.rev; d["fail_out"] = d.Failure * (~d.rev)
print((d.groupby(["season", "game_type"])[["Failure", "fail_rev", "fail_out"]].mean()*100).round(2).unstack("game_type").to_string())
sub("반대투구율의 투수 지속성 (해간, 100투구+) — 반대투구가 투수 고유 성향이면 큰 Brier 재료")
for Y in (2023, 2024):
    a = d[d.season == Y-1].groupby("pitcher_id").rev.agg(["mean", "size"]); b = d[d.season == Y].groupby("pitcher_id").rev.agg(["mean", "size"])
    ok = a.index[a["size"] >= 100].intersection(b.index[b["size"] >= 100])
    print(f"  {Y-1}→{Y} n={len(ok)}  r {np.corrcoef(a.loc[ok,'mean'], b.loc[ok,'mean'])[0,1]:+.3f}")
sub("반대투구율 — 카운트12 × 타자손 (R, 2023+)")
r = d[(d.game_type == "R") & (d.season >= 2023)]
print((r.pivot_table(index="cnt", columns="matchup", values="rev", aggfunc="mean")*100).round(1).to_string())
print(f"\n{tick()} done")
