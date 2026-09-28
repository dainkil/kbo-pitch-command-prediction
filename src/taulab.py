"""τ 실험실 — 투수 단위 순위 신호를 Kendall τ 로 직접 평가·게이트·결합한다 (2022·2023·2024 3폴드).
기반 = L2 EB(α=2000) 의 투수별 평균 예측점수. 후보 신호 = season<Y 이력에서 만든 투수 표(build_tables) 열.
게이트(사전): 3폴드 τ 부호 일치 AND |평균 τ| ≥ 0.05. 결합 = z(EB) + ε·Σ sign·z(신호) — 노이즈 강건: 수축(EB)·부호게이트·등가중.
주입은 행별 pitcher_id 표 조회 → 규칙 4 충족(행 독립)."""
import sys, os, time
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
T0 = time.time(); tk = lambda: f"[{time.time()-T0:4.0f}s]"
d_all = load(labeled_only=False); tm = load_tm(); d = d_all[d_all.y.notna()]
CAND = ["p_skill", "p_sh", "p_he", "p_fa", "pr_fa", "p1_fa", "p1_sh", "pgtR_fa", "pgtF_fa", "pgtR_sh", "p_fshare", "p_n_all", "p_rev", "p_out",
        "p_h_sd", "p_side_sd", "p_abs_side", "p_h_mean", "pgtR_rev", "tm_mix_fastball", "tm_mix_offspeed", "tm_mix_breaking", "tm_sd_rel_height",
        "tm_sd_rel_side", "tm_extension", "tm_induced_vert_break", "tm_horz_break", "tm_rel_speed", "tm_spin_rate", "tm_fb_rel_speed", "tm_n"]
F = {}
for Y in (2022, 2023, 2024):
    lab, ev = fold(d, Y); T = build_tables(lab, d_all[d_all.season < Y], tm[tm.season < Y], Y)
    E = eb_tables(lab, alpha=2000); Pe = eb_predict(ev, E)
    g = pd.DataFrame({"pid": ev.pitcher_id.to_numpy(), "act": ev.pt.to_numpy(), "eb": Pe @ PT}).groupby("pid").agg(n=("act", "size"), act=("act", "mean"), eb=("eb", "mean"))
    g = g[g.n >= 50]; X = T["pitcher"].reindex(g.index)[CAND].astype(float)
    # GBDT(noF) 투수 평균도 후보로
    z = np.load(OUT + f"p_val_noF_{Y}.npz", allow_pickle=True); X["gbdt_noF"] = pd.Series(z["P"] @ PT).groupby(z["pid"]).mean().reindex(g.index)
    F[Y] = (g, X); print(f"{tk()} {Y}: 투수 {len(g)} · 신인(p_n NaN) {int(X.p_fa.isna().sum())}", flush=True)
def tau(a, b): return kendalltau(np.asarray(a).round(10), np.asarray(b).round(10), variant="b").statistic
head("1. 후보 신호 단독 τ (결측 → 그 폴드 중앙값)  ·  게이트: 부호 3/3 + |mean| ≥ 0.05")
rows = []
for c in list(CAND) + ["gbdt_noF", "eb"]:
    ts = []
    for Y in (2022, 2023, 2024):
        g, X = F[Y]; v = (g.eb if c == "eb" else X[c]).astype(float); ts.append(tau(g.act, v.fillna(v.median())))
    ts = np.array(ts); sg = int(np.sign(ts.mean())); gate = (np.sign(ts) == sg).all() and abs(ts.mean()) >= 0.05
    rows.append(dict(sig=c, t22=ts[0], t23=ts[1], t24=ts[2], mean=ts.mean(), sign=sg, gate="✓" if gate else "✗"))
R = pd.DataFrame(rows).set_index("sig").sort_values("mean", key=abs, ascending=False); print(R.round(4).to_string())
G = [c for c in R.index if R.loc[c, "gate"] == "✓" and c not in ("eb", "gbdt_noF")]; print(f"\n게이트 통과 {len(G)}: {G}")
def zs(v): v = v.astype(float); v = v.fillna(v.median()); s = v.std(); return (v - v.mean()) / (s if s > 0 else 1)
head("2. 결합: z(EB) + ε·Σ sign·z(신호)  — 3폴드 τ, EB 단독 대비 Δ")
base = {Y: tau(F[Y][0].act, F[Y][0].eb) for Y in F}; print("  EB 단독 τ:", {Y: round(v, 4) for Y, v in base.items()})
def combo(Y, sigs, eps, weights=None):
    g, X = F[Y]; s = zs(g.eb).copy()
    for i, c in enumerate(sigs): s = s + eps * (weights[i] if weights else 1.0) * R.loc[c, "sign"] * zs(X[c])
    return tau(g.act, s)
sets = {"게이트 전부": G, "이력만(tm 제외)": [c for c in G if not c.startswith("tm_")], "tm만": [c for c in G if c.startswith("tm_")],
        "상위5(|mean|)": G[:5], "상위3": G[:3]}
for nm, sigs in sets.items():
    if not sigs: continue
    print(f"\n  [{nm}] {sigs}")
    for eps in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        ts = [combo(Y, sigs, eps) for Y in (2022, 2023, 2024)]; dl = [t - base[Y] for t, Y in zip(ts, (2022, 2023, 2024))]
        ok = "✓3/3" if all(x > 0 for x in dl) else ("✗" if all(x < 0 for x in dl) else "~")
        print(f"    ε={eps:<4} τ {ts[0]:+.4f} {ts[1]:+.4f} {ts[2]:+.4f}  mean {np.mean(ts):+.4f}  Δ {dl[0]:+.4f} {dl[1]:+.4f} {dl[2]:+.4f}  {ok}")
head("3. 신호별 단독 주입 (ε=0.2) — 어느 신호가 EB 에 실제로 보태나 (EB 와 중복이면 Δ≈0)")
for c in G:
    ts = [combo(Y, [c], 0.2) for Y in (2022, 2023, 2024)]; dl = [t - base[Y] for t, Y in zip(ts, (2022, 2023, 2024))]
    ok = "✓3/3" if all(x > 0 for x in dl) else ("✗" if all(x < 0 for x in dl) else "~")
    print(f"    {c:22s} Δτ {dl[0]:+.4f} {dl[1]:+.4f} {dl[2]:+.4f}  mean {np.mean(dl):+.4f}  {ok}")
print(f"\n{tk()} done")
