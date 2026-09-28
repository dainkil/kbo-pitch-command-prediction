"""E5 — 카운트 잔차 EB 수준 (위치·체제 미사용): 클래스별 y_c − m_c(카운트12) 잔차를 투수 EB(배치셀 prior, α) 로 수축, 행 확률 = 전역평균 + 잔차 EB.
변형: 잔차 테이블 = cnt / cnt×batter_hand / cnt×pitcher_hand; 혼합 = (1−w)·기준EB + w·잔차EB 점수를 anchor 목표로."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from pipeline import *
d = load()
def resid_eb_rows(ev, lab, keys, alpha=2000.0, ac=200.0):
    oh = np.eye(3)[lab.y.to_numpy().astype(int)]; gm = oh.mean(0)
    tab = pd.DataFrame(oh, columns=CLASSES).assign(**{k: lab[k].to_numpy() for k in keys}).groupby(keys)[CLASSES].mean()
    m = lab[keys].merge(tab, left_on=keys, right_index=True, how="left")[CLASSES].to_numpy(); r = oh - m          # 잔차
    pid = lab.pitcher_id.to_numpy(); cell = cells(lab)
    g = pd.DataFrame(r, columns=CLASSES).assign(p=pid).groupby("p").agg(["sum", "size"]); pn = g[(CLASSES[0], "size")].to_numpy(float); rs = g.loc[:, [(c, "sum") for c in CLASSES]].to_numpy(float)
    prate = rs / pn[:, None]; pidx = pd.Index(g.index.astype(object))
    # 배치셀 prior (잔차의 투수×시즌 평균, 50투구+, ac 수축 → 0 이 아닌 전역 잔차평균 0)
    ps = pd.DataFrame(r, columns=CLASSES).assign(p=pid, s=lab.season.to_numpy()).groupby(["p", "s"]).agg(["mean", "size"]); keep = ps[(CLASSES[0], "size")].to_numpy() >= 50
    rates = ps.loc[:, [(c, "mean") for c in CLASSES]].to_numpy(float)[keep]; key = ps.index[keep]
    sel = pd.Series(np.arange(len(key)), index=key).reindex(pd.MultiIndex.from_arrays([pid, lab.season.to_numpy()])).to_numpy(); ok = ~np.isnan(sel)
    K = cell[ok]; R = rates[sel[ok].astype(int)]; cnt = np.bincount(K, minlength=NCELL).astype(float)
    prior = np.column_stack([np.bincount(K, weights=R[:, j], minlength=NCELL) for j in range(3)]) / np.maximum(cnt[:, None] + ac, 1)   # 0 수축
    # 평가 행
    pr = prior[cells(ev)]; loc = pidx.get_indexer(pd.Index(ev.pitcher_id.to_numpy())); have = loc >= 0; safe = np.where(have, loc, 0)
    n = np.where(have, pn[safe], 0.0)[:, None]; rate = np.where(have[:, None], prate[safe], 0.0)
    p = gm + (rate * n + alpha * pr) / (n + alpha); p = np.clip(p, 1e-6, None); return p / p.sum(1, keepdims=True)
rows = []
for Y in (2022, 2023, 2024):
    lab, ev = fold(d, Y); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy(); E = eb_tables(lab, 2000); Pe = eb_predict(ev, E)
    z = np.load(OUT + f"p_val_v3_{Y}.npz", allow_pickle=True); assert (z["pid"] == pid).all(); Pg = z["P"]
    head(f"{Y}"); r = score(anchor(Pg, Pe, 1.0), pid, y, "기준 EB"); rows.append(dict(Y=Y, m="EB", tot=r[0], tau=r[2]))
    for nm, keys in (("cnt", ["cnt"]), ("cnt×bh", ["cnt", "batter_hand"]), ("cnt×ph", ["cnt", "pitcher_hand"]), ("cnt×inn", ["cnt", "inn"])):
        for a in (1000.0, 2000.0):
            Pr = resid_eb_rows(ev, lab, keys, a)
            for w in (0.5, 1.0):
                Pm = (1 - w) * Pe + w * Pr; Pm /= Pm.sum(1, keepdims=True)
                r = score(anchor(Pg, Pm, 1.0), pid, y, f"잔차EB {nm} α={a:g} w={w}"); rows.append(dict(Y=Y, m=f"{nm}_a{a:g}_w{w}", tot=r[0], tau=r[2]))
R = pd.DataFrame(rows).set_index(["m", "Y"]).unstack("Y"); base = R.loc["EB"]
R["gate"] = [("✓" if ((R.loc[m, "tau"] > base["tau"]).all() and (R.loc[m, "tot"] > base["tot"]).all()) else "✗") for m in R.index]
for k in ("tot", "tau"): R[(k, "mean")] = R[k].mean(1)
head("3폴드 요약"); print(R[[("tau", 2022), ("tau", 2023), ("tau", 2024), ("tau", "mean"), ("tot", "mean"), ("gate", "")]].round(4).sort_values(("tau", "mean"), ascending=False).to_string())
