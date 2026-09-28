"""dain EDA 공통 — 데이터 로드(캐시), 공식 지표(Total = 0.7·Pitch + 0.3·Player), 표/EB 확률 헬퍼.
종민 eda 의 관례(pt=100/30/0, 50투구+ 투수, τ_b, round(10))를 그대로 따른다."""
import os, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import kendalltau
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 50)

D   = "/home/ubuntu/workspace/data/"
OUT = "/home/ubuntu/workspace/산출물/dain/out/"
CLASSES = ["Shadow", "Heart", "Failure"]          # 제출 열 순서 = prob_shadow, prob_heart, prob_failure
PT = np.array([100., 30., 0.])                     # 제구점수
CACHE = OUT + "_train_merged.pkl"
T0 = time.time()

def head(t): print("\n" + "═"*90 + f"\n{t}\n" + "═"*90, flush=True)
def sub(t):  print("\n" + "─"*4 + " " + t, flush=True)
def tick():  return f"[{time.time()-T0:5.0f}s]"

def load(labeled_only=True):
    """train + outcomes 를 row_id 로 병합. 컨텍스트 파생열 포함. 첫 실행 시 pickle 캐시."""
    if os.path.exists(CACHE):
        d = pd.read_pickle(CACHE)
    else:
        tr = pd.read_csv(D+"train.csv", dtype={"row_id": str})
        oc = pd.read_csv(D+"train_outcomes.csv", dtype={"row_id": str})
        d = tr.merge(oc, on="row_id", validate="one_to_one")
        d["y"]  = d.control_grade.map({c: i for i, c in enumerate(CLASSES)}).astype(float)  # 미라벨 = NaN
        d["pt"] = d.y.map({0: 100., 1: 30., 2: 0.})
        d["cnt"] = d.balls_before.clip(0, 3)*3 + d.strikes_before.clip(0, 2)               # 0..11
        d["inn"] = d.inning.clip(1, 10)
        d["li3"] = pd.cut(d.li, [-1, .6, 1.4, 99], labels=False)
        d["sd5"] = pd.cut(d.score_diff_pitcher_team, [-99, -3, -1, 1, 3, 99], labels=False)
        d["hw5"] = pd.cut(d.home_win_expectancy, [-1, 20, 40, 60, 80, 101], labels=False)
        d["matchup"] = d.pitcher_hand.astype(int)*10 + d.batter_hand.astype(int)           # 11 12 21 22
        d["rev"] = d.REVERSE_YN.eq("T")
        os.makedirs(OUT, exist_ok=True); d.to_pickle(CACHE)
    return d[d.y.notna()].copy() if labeled_only else d

def fold(d, Y): return d[d.season < Y], d[d.season == Y]

# ── 공식 지표 ─────────────────────────────────────────────────────────────
def onehot(y): return np.eye(3)[np.asarray(y).astype(int)]
def prior(df): return onehot(df.y.to_numpy()).mean(0)
def brier(P, y): return float(((np.asarray(P) - onehot(y))**2).sum(1).mean())
def pitch_score(P, y): return 100*(1 - brier(P, y)/2)
def player_score(P, pid, y, min_n=50):
    s = pd.DataFrame({"pid": np.asarray(pid), "pred": np.asarray(P) @ PT, "act": PT[np.asarray(y).astype(int)]})
    g = s.groupby("pid").agg(n=("act", "size"), pred=("pred", "mean"), act=("act", "mean"))
    el = g[g.n >= min_n]
    tau = kendalltau(el.act.round(10), el.pred.round(10), variant="b").statistic
    tau = 0.0 if np.isnan(tau) else float(tau)
    return 50*(1+tau), tau, len(el)
def score(P, pid, y, label="", quiet=False):
    """한 줄 요약 출력 후 (total, pitch, tau) 반환."""
    b = brier(P, y); ps = 100*(1-b/2); pl, tau, n = player_score(P, pid, y); tot = 0.7*ps + 0.3*pl
    if not quiet:
        print(f"  {label:<46s} Brier {b:.4f}  Pitch {ps:6.3f}  τ {tau:+.4f}  Player {pl:6.2f}  ★Total {tot:6.3f}", flush=True)
    return tot, ps, tau

# ── 확률 모형 헬퍼 ────────────────────────────────────────────────────────
def table_probs(past, ev, keys, alpha=50, pri=None):
    """past 의 keys 셀별 클래스 비율(α 로 prior 수축)을 ev 에 매핑. 미관측 셀 → prior."""
    pri = prior(past) if pri is None else pri
    tmp = past[keys].reset_index(drop=True).copy(); yy = past.y.to_numpy()
    for i, c in enumerate(CLASSES): tmp[c] = (yy == i).astype(float)
    g = tmp.groupby(keys, observed=True)[CLASSES].sum(); n = g.sum(1)
    for i, c in enumerate(CLASSES): g[c] = (g[c] + alpha*pri[i])/(n + alpha)
    m = ev[keys].reset_index(drop=True).merge(g, left_on=keys, right_index=True, how="left")
    P = m[CLASSES].to_numpy(); miss = np.isnan(P[:, 0]); P[miss] = pri
    return P

def entity_probs(past, ev, key, alpha, pri=None, w=None):
    """개체(투수/타자) 별 클래스 비율 EB. w = past 행 가중(최근성). 반환 (P, n_hist)."""
    pri = prior(past) if pri is None else pri
    w = np.ones(len(past)) if w is None else np.asarray(w, float)
    tmp = pd.DataFrame({key: past[key].to_numpy()}); yy = past.y.to_numpy()
    for i, c in enumerate(CLASSES): tmp[c] = (yy == i)*w
    tmp["n"] = w
    g = tmp.groupby(key).sum()
    for i, c in enumerate(CLASSES): g[c] = (g[c] + alpha*pri[i])/(g.n + alpha)
    m = pd.DataFrame({key: ev[key].to_numpy()}).merge(g, on=key, how="left")
    P = m[CLASSES].to_numpy(); miss = np.isnan(P[:, 0]); P[miss] = pri
    return P, m.n.fillna(0).to_numpy()

def combine(*Ps, pri, eps=1e-4):
    """prior 로부터의 편차를 가산 결합 후 클리핑·정규화 (EDA 용 근사)."""
    P = pri + sum(p - pri for p in Ps); P = np.clip(P, eps, None)
    return P/P.sum(1, keepdims=True)

def const(pri, n): return np.tile(pri, (n, 1))
