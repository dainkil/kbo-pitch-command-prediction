"""22: E1 후속 — 다음 시즌 점수를 가장 잘 예측하는 등급 성분 가중 (Shadow/Heart/reverse/위치이탈) · 시즌별 표본외 검증 (R 행 · 투수-시즌 ≥200구)."""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import kendalltau
sys.path.insert(0, os.path.dirname(__file__)); from _common import load, head, sub
d = load(labeled_only=True); d = d[d.game_type == "R"]
def tau(a, b): r = kendalltau(np.round(a, 10), np.round(b, 10), variant="b").statistic; return 0.0 if np.isnan(r) else r
oh = np.eye(3)[d.y.to_numpy().astype(int)]
ps = pd.DataFrame({"pid": d.pitcher_id.to_numpy(), "s": d.season.to_numpy(), "sh": oh[:, 0], "he": oh[:, 1], "rev": d.rev.to_numpy(float), "pt": d.pt.to_numpy()})
ps["fa_nr"] = (oh[:, 2]) * (1 - ps.rev)
a = ps.groupby(["pid", "s"]).agg(n=("pt", "size"), sh=("sh", "mean"), he=("he", "mean"), rev=("rev", "mean"), fa_nr=("fa_nr", "mean"), pt=("pt", "mean")).reset_index(); a = a[a.n >= 200]
b = a.copy(); b["s"] = b.s - 1; j = a.merge(b[["pid", "s", "pt"]], on=["pid", "s"], suffixes=("", "_next"))
head("22. 다음시즌 점수 예측 가중 — 시즌 s 로 학습(≤s−1 쌍) → 시즌 s+1 예측 (표본외)")
print(f"{'평가 s+1':<8}{'쌍':>5}  {'pt 단독 τ':>10}  {'sh+he τ':>9}  {'sh+he+rev τ':>12}  {'rev 단독 τ':>11}   계수(sh,he,rev)")
for s in (2021, 2022, 2023):
    tr = j[j.s < s]; te = j[j.s == s]
    if len(te) < 20: continue
    def fit(cols):
        X = np.column_stack([tr[c] for c in cols] + [np.ones(len(tr))]); beta = np.linalg.lstsq(X, tr.pt_next, rcond=None)[0]
        Xt = np.column_stack([te[c] for c in cols] + [np.ones(len(te))]); return Xt @ beta, beta
    p2, b2 = fit(["sh", "he"]); p3, b3 = fit(["sh", "he", "rev"])
    print(f"{s+1:<8}{len(te):>5}  {tau(te.pt_next, te.pt):>+10.3f}  {tau(te.pt_next, p2):>+9.3f}  {tau(te.pt_next, p3):>+12.3f}  {tau(te.pt_next, -te.rev):>+11.3f}   {np.round(b3[:3], 1)}")
print("점수 가중 100/30/0 은 '이번 시즌 점수'의 정의. 다음 시즌 순위엔 reverse 가 더 무겁고 위치이탈(fa_nr)은 거의 무정보 → EB 목표의 성분 가중을 바꾸는 여지.")
