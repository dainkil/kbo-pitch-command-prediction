# -*- coding: utf-8 -*-
"""투수 단위 오프셋 신호 — factory_orth.py 의 정의를 그대로 이식 + 새 신호(GEOF·GEOF23·DIRc·DSHsc·SITb·CNTH·CPs).
전부 학습 데이터의 투수별 집계 상수 → pitcher_id 조회 → 행 독립."""
import os, sys, json, numpy as np, pandas as pd
from common import *

def build(d, meta):
    dl = d[d.lab].copy()
    x = dl.plate_loc_side.abs(); h = dl.plate_loc_height
    heart = (x <= 0.263) & (h >= 0.537) & (h <= 0.963); shadow = (x <= 0.373) & (h >= 0.427) & (h <= 1.073) & ~heart
    dl["geo"] = np.where(heart, 30.0, np.where(shadow, 100.0, 0.0)); rbar = float(dl.rev.mean())
    cen = lambda v: v - v.groupby(dl.season).transform("mean") + float(v.mean())
    dl["ptc"] = cen(dl.pt); dl["geoc"] = cen(dl.geo); dl["sc"] = cen((dl.y == 0).astype(float)); dl["hc"] = cen((dl.y == 1).astype(float))
    def EB(col, a, scale=1.0, w=None):
        ww = pd.Series(1.0, index=dl.index) if w is None else w; mu = float(np.average(dl[col], weights=ww))
        g = pd.DataFrame({"s": dl[col] * ww, "w": ww, "p": dl.pitcher_id}).groupby("p").sum(); return scale * (g.s + a * mu) / (g.w + a)
    n = dl.groupby("pitcher_id").size(); fsh = dl.groupby("pitcher_id").isF.mean(); cw = lambda o: o - float((o * n).sum() / n.sum())
    eA = EB("ptc", 1800); geo = cw(EB("geoc", 1800, 1 - rbar) - eA); L164p = eA + 0.3 * geo
    act = n.index[(dl[dl.season == 2024].groupby("pitcher_id").size().reindex(n.index).fillna(0) >= 50).to_numpy()]
    Z = {}
    # ===== L236 의 6축 (factory_orth 와 동일) =====
    Z["RK"] = pd.read_csv(JM + "/out/rk_score.csv", index_col=0).iloc[:, 0].reindex(n.index).fillna(0.0)
    pk = pd.to_numeric(dl.PITCH_KIND, errors="coerce"); top = pk.value_counts().index[:7]; pkg = np.where(pk.isin(top), pk, -1).astype(int).astype(str)
    cell = dl.cnt.astype(int).astype(str) + "_" + dl.pitcher_hand.astype(str) + dl.batter_hand.astype(str) + "_" + dl.outs_before.astype(int).astype(str) + "_" + (pd.to_numeric(dl.num_runners_on, errors="coerce").fillna(0) > 0).astype(int).astype(str) + "_" + pkg
    g = dl.ptc.groupby(cell).agg(["mean", "size"]); cm = g["mean"].where(g["size"] >= 200, dl.ptc.mean()); dl["sit"] = dl.ptc - cell.map(cm).to_numpy() + dl.ptc.mean(); Z["SIT"] = EB("sit", 1800) - eA
    Z["SH"] = 100 * EB("sc", 1000) + 50 * EB("hc", 600) - eA
    dl["_sh"] = (dl.pitcher_hand == dl.batter_hand).astype(float)
    Z["PLT"] = (dl.groupby("pitcher_id")._sh.mean() - 0.5) * 4.0
    _cs = pd.DataFrame({"s": (dl.y == 0).groupby(dl.pitcher_id).sum(), "h": (dl.y == 1).groupby(dl.pitcher_id).sum(), "f": (dl.y == 2).groupby(dl.pitcher_id).sum()})
    _pr = np.array([(dl.y == 0).mean(), (dl.y == 1).mean(), (dl.y == 2).mean()]); _AS, _AH = 900.0, 500.0
    _ps = (_cs.s + _AS * _pr[0]) / (_cs.sum(1) + _AS); _ph = (_cs.h + _AH * _pr[1]) / (_cs.sum(1) + _AH)
    Z["DIR"] = (100 * _ps + 30 * _ph).reindex(n.index) - eA
    _n1 = dl[dl._sh == 1].groupby("pitcher_id").size().reindex(n.index).fillna(0.0)
    _r1 = (dl[dl._sh == 1].y == 0).groupby(dl[dl._sh == 1].pitcher_id).mean().reindex(n.index); _m = float((dl.y == 0).mean())
    Z["DSHs"] = ((_r1 - _m).fillna(0.0) * (_n1 / (_n1 + 300.0)) * 100.0)
    # ===== 시즌중심화 판 (F1 진단용) =====
    Z["DIRc"] = 100 * EB("sc", 900) + 30 * EB("hc", 500) - eA
    _r1c = dl[dl._sh == 1].sc.groupby(dl[dl._sh == 1].pitcher_id).mean().reindex(n.index); _mc = float(dl.sc.mean())
    Z["DSHsc"] = ((_r1c - _mc).fillna(0.0) * (_n1 / (_n1 + 300.0)) * 100.0)
    # ===== ① GEOF: F 행만 기하 재채점 (R 행 반대투구 정보 보존) =====
    rF23 = float(dl[(dl.isF == 1) & (dl.season >= 2023)].rev.mean())
    dl["gf"] = np.where(dl.isF == 1, dl.geo * (1 - rF23), dl.pt); dl["gfc"] = cen(dl.gf); Z["GEOF"] = EB("gfc", 1800) - eA
    dl["gf23"] = np.where((dl.isF == 1) & (dl.season <= 2022), dl.geo * (1 - rF23), dl.pt); dl["gf23c"] = cen(dl.gf23); Z["GEOF23"] = EB("gf23c", 1800) - eA
    P(f"  GEOF: F 행 2023+ 반대투구율 {rF23:.4f} · F 행 {int(dl.isF.sum()):,} (2019~22 {int(((dl.isF==1)&(dl.season<=2022)).sum()):,})")
    # ===== SITb: 배치셀(li3×inn4×sd5) 잔차 EB =====
    bc = SC.cells(dl); gb = dl.ptc.groupby(bc).agg(["mean", "size"]); cmb = gb["mean"].where(gb["size"] >= 200, dl.ptc.mean())
    dl["sitb"] = dl.ptc - pd.Series(bc, index=dl.index).map(cmb).to_numpy() + dl.ptc.mean(); Z["SITb"] = EB("sitb", 1800) - eA
    # ===== CNTH: 전체행(라벨+미라벨) 카운트 이력 z̄ · d22 β =====
    ks = meta["use_keys"]; bb = np.array(meta["use_b"]); mu = np.array(meta["use_mu"]); sd = np.array(meta["use_sd"])
    Zr = (d[ks].apply(pd.to_numeric, errors="coerce").fillna(pd.Series(mu, index=ks)).to_numpy(float) - mu) / sd
    zz = pd.DataFrame(Zr, index=d.index, columns=ks).groupby(d.pitcher_id).agg(["mean", "size"])
    na = d.groupby("pitcher_id").size(); zbar = pd.DataFrame({k: zz[(k, "mean")] for k in ks}).mul(na / (na + 200.0), axis=0)
    Z["CNTH"] = (zbar.to_numpy() @ bb); Z["CNTH"] = pd.Series(Z["CNTH"], index=zbar.index).reindex(n.index).fillna(0.0)
    # ===== ② CPs: comparable prior (소이력 한정) =====
    try:
        ci, cv = T7.comp_table(d, int(d.season.max()))
        cp = pd.Series(cv, index=pd.Index(ci)).reindex(n.index.astype(str)); cp.index = n.index
        Z["CPs"] = (cp.fillna(0.0) * (1800.0 / (n + 1800.0)))
    except Exception as e:
        P("  CPs 실패:", e); Z["CPs"] = pd.Series(0.0, index=n.index)
    return dict(dl=dl, n=n, fsh=fsh, eA=eA, geo=geo, L164p=L164p, act=act, Z=Z, cw=cw)

def make_poff(S, spec, bc=False):
    """spec = 'RK:0.25,SIT:0.25,...' (활동투수 sd 로 스케일) · bc=True 면 n 구간 더미까지 직교화"""
    n, fsh, L164p, act, Z, cw = S["n"], S["fsh"], S["L164p"], S["act"], S["Z"], S["cw"]
    A_ = np.c_[np.ones(len(n)), L164p.to_numpy(), np.log1p(n.to_numpy()), fsh.reindex(n.index).to_numpy()]
    if bc:
        b = pd.cut(n, [0, 100, 300, 1000, 3000, 1e9], labels=False).to_numpy(); A_ = np.c_[A_, np.eye(5)[b][:, :4]]
    W = np.sqrt(n.to_numpy())
    def orth(z):
        co = np.linalg.lstsq(A_ * W[:, None], z.to_numpy() * W, rcond=None)[0]; return z - A_ @ co
    comp = {"GEO": 0.3 * S["geo"]}
    for c in spec.split(","):
        k, sdv = c.split(":"); z = orth(Z[k].reindex(n.index).fillna(0.0)); kap = float(sdv) / float(z.reindex(act).std()); comp[k] = kap * cw(z)
    o = sum(comp.values())
    bins = pd.cut(n, [0, 100, 300, 1000, 3000, 1e9])
    tab = pd.DataFrame({k: v.reindex(act).groupby(bins.reindex(act), observed=True).mean() for k, v in comp.items()}); tab["TOTAL"] = o.reindex(act).groupby(bins.reindex(act), observed=True).mean()
    P("  성분별 블록 평균(2024 활동투수):\n" + tab.round(3).to_string())
    P("  성분 sd(활동):", {k: round(float(v.reindex(act).std()), 3) for k, v in comp.items()}, "TOTAL", round(float(o.reindex(act).std()), 3))
    C = pd.DataFrame({k: v.reindex(act) for k, v in comp.items() if k != "GEO"}).corr().round(2); P("  성분 상관:\n" + C.to_string())
    return o, comp
