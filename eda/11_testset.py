"""EDA 11 — 평가셋의 정체. 공개 test 는 5행 더미, 실제는 218,992행(대회 안내). 시즌별 라벨 행수와 비교해
'평가셋 = 2025 라벨 확정 투구만' 인지 추정하고, 라벨 결측이 무엇과 상관있는지, 투수 구성(신인 비율·투구 수)을 본다."""
from _common import *
N_TEST = 218_992
d = load(labeled_only=False); d["lab"] = d.y.notna()
head("1. 시즌별 전체/라벨 행수 vs 실제 평가셋 218,992행")
t = d.groupby("season").agg(전체=("lab", "size"), 라벨=("lab", "sum")); t["미라벨%"] = (100 - 100*t.라벨/t.전체).round(2)
print(t.to_string()); print(f"\n  2024 라벨 {t.loc[2024,'라벨']:,} vs 평가 {N_TEST:,} → 차이 {N_TEST - t.loc[2024,'라벨']:+,}"
      f"  ({N_TEST/t.loc[2024,'전체']*100:.1f}% of 2024 전체)  ⇒ 평가셋은 **라벨 확정 투구만** 으로 보인다 (전체면 ~25만).")

head("2. 라벨 결측은 정보인가 — 결측률을 사전 정보로 쪼갬 (평가셋이 라벨행만이면 학습도 라벨행만 쓰면 되지만, 편향 여부는 확인)")
for k in ("game_type", "cnt", "inn", "outs_before", "num_runners_on", "li3", "game_month", "top_bottom", "pitcher_hand"):
    m = d.groupby(k, observed=True).lab.mean().mul(100).round(1)
    print(f"  {k:16s} 라벨률%  " + "  ".join(f"{i}:{v}" for i, v in m.items()))
sub("투수별 라벨률 분포 (시즌 2024, 50투구+)")
pm = d[d.season == 2024].groupby("pitcher_id").lab.agg(["size", "mean"]); pm = pm[pm["size"] >= 50]
print(f"  투수 {len(pm)}명 · 라벨률 분위 5/25/50/75/95: {np.percentile(pm['mean']*100, [5,25,50,75,95]).round(1).tolist()}")
sub("라벨 행 vs 미라벨 행의 사전정보 분포 차이 (2024)")
e = d[d.season == 2024]
for k in ("cnt", "inn", "li"):
    print(f"  {k:6s} 라벨 mean {e[e.lab][k].mean():.3f} · 미라벨 mean {e[~e.lab][k].mean():.3f}")

head("3. 평가 투수 구성 — τ 는 '평가 투구 50개+' 투수만 본다")
L = d[d.lab]
for Y in (2022, 2023, 2024):
    past, ev = fold(L, Y)
    g = ev.groupby("pitcher_id").size(); el = g[g >= 50]
    hist = past.groupby("pitcher_id").size()
    nh = hist.reindex(el.index).fillna(0)
    share = el.sum()/len(ev)*100
    print(f"  {Y}: 투수 {ev.pitcher_id.nunique()}명 · 50+ {len(el)}명(투구의 {share:.1f}%) · 신인(이력0) {int((nh==0).sum())}명"
          f" · 이력<200 {int((nh<200).sum())}명 · 이력<1000 {int((nh<1000).sum())}명 · 50+투수 투구수 분위10/50/90 {np.percentile(el,[10,50,90]).round(0).tolist()}")
sub("2024 평가 행 기준: 투수 이력 크기별 행 비중 (Brier 개선이 어디서 나오는지 가늠)")
past, ev = fold(L, 2024); hist = past.groupby("pitcher_id").size()
nh = ev.pitcher_id.map(hist).fillna(0)
b = pd.cut(nh, [-1, 0, 100, 500, 2000, 99999], labels=["0(신인)", "1-100", "101-500", "501-2000", "2000+"])
print((b.value_counts(normalize=True).sort_index()*100).round(2).to_string())
sub("타자 이력 크기별 (2024)")
bh = ev.batter_id.map(past.groupby("batter_id").size()).fillna(0)
b2 = pd.cut(bh, [-1, 0, 100, 500, 2000, 99999], labels=["0", "1-100", "101-500", "501-2000", "2000+"])
print((b2.value_counts(normalize=True).sort_index()*100).round(2).to_string())

head("4. 시즌 간 투수 겹침 — 2025 에 나올 투수를 2024 로 얼마나 커버하나 (대리: 2024 를 2023 이하로)")
for Y in (2022, 2023, 2024):
    past, ev = fold(L, Y); prev = L[L.season == Y-1]
    pit = set(ev.pitcher_id); print(f"  {Y}: 투수 {len(pit)}명 · 과거 전체 등장 {len(pit & set(past.pitcher_id))}명 · 직전 시즌 등장 {len(pit & set(prev.pitcher_id))}명"
                                     f" · 투구 가중 커버(과거) {ev.pitcher_id.isin(set(past.pitcher_id)).mean()*100:.1f}%")
print(f"\n{tick()} done")
