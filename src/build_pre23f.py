# -*- coding: utf-8 -*-
"""2019~2022 퓨처스(F) 라벨 체제 처리 — anchor 목표 EB 표(eb_tables.npz / eb_meta.json)만 다시 만든다. GBDT·tables·adj.json 불변.
근거: 2023 부터 F 반대투구 기록 규칙이 바뀌어(존 안 반대투구율 .100 → .309) 2019~22 F 행의 등급이 부풀려져 있다.
사용: build_pre23f.py <keep|drop|soft|w0.3> <model_dir> [--upto 2025] [--alpha 2000] [--ref eb_tables.npz]
  keep  : 현행 그대로(재구현 검증용)      drop : 2019~22 F 행 제외      w0.3 : 그 행 가중 0.3
  soft  : 2023+ F 라벨행으로 p̂(반대투구|행) 을 학습해 2019~22 F 비반대투구 행을 (1−p̂)·onehot(등급) + p̂·Failure 로 재라벨
pipeline.py / train.py / submit/script.py 는 건드리지 않는다 (다른 작업에서 편집 중). 표만 바뀌므로 추론 코드 변경 없음."""
import sys, os, json, argparse, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eda")); from _common import load, PT

LI_BINS = (0.6, 1.4); INN_BINS = (4, 7, 9); SD_BINS = (-3, -1, 2, 4); NCELL = 3 * 4 * 5   # submit/script.py 의 cells() 와 동일 고정 경계
def cells(df):
    li = pd.to_numeric(df["li"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    inn = pd.to_numeric(df["inning"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    sd = pd.to_numeric(df["score_diff_pitcher_team"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return (np.digitize(li, LI_BINS) * 4 + np.digitize(inn, INN_BINS)) * 5 + np.digitize(sd, SD_BINS)

def eb_tables_w(lab, oh, w, alpha=2000.0, ac=200.0, min_ps=50):
    """pipeline.eb_tables 의 가중·소프트 라벨 판. oh: (n,3) 기대 one-hot, w: (n,) 행 가중. oh=eye, w=1 이면 원본과 동일(부동소수 오차 내)."""
    w = np.asarray(w, float); ohw = oh * w[:, None]; gm = ohw.sum(0) / w.sum()
    cols = ["s0", "s1", "s2", "w"]
    g = pd.DataFrame(np.column_stack([ohw, w]), index=lab.pitcher_id.to_numpy(), columns=cols).groupby(level=0).sum()
    pid = g.index.to_numpy().astype("U32"); pn = g["w"].to_numpy(float); prate = g[["s0", "s1", "s2"]].to_numpy(float) / pn[:, None]
    mi = pd.MultiIndex.from_arrays([lab.pitcher_id, lab.season])
    ps = pd.DataFrame(np.column_stack([ohw, w]), index=mi, columns=cols).groupby(level=[0, 1]).sum()
    keep = ps["w"].to_numpy() >= min_ps; rates = (ps[["s0", "s1", "s2"]].to_numpy(float) / ps["w"].to_numpy(float)[:, None])[keep]; key = ps.index[keep]
    sel = pd.Series(np.arange(len(key)), index=key).reindex(mi).to_numpy(); ok = ~np.isnan(sel)
    K = cells(lab)[ok]; R = rates[sel[ok].astype(int)]; wk = w[ok]
    cnt = np.bincount(K, weights=wk, minlength=NCELL); sm = np.column_stack([np.bincount(K, weights=R[:, j] * wk, minlength=NCELL) for j in range(3)])
    prior = (sm + ac * gm) / (cnt[:, None] + ac); prior /= prior.sum(1, keepdims=True)
    return dict(prior=prior, prate=prate, pn=pn, pid=pid, alpha=float(alpha), ac=float(ac), gm=gm)

def rev_feats(d):
    s = d.plate_loc_side.astype(float); h = d.plate_loc_height.astype(float)
    return pd.DataFrame({"side": s, "h": h, "abs_side": s.abs(), "heart": ((s.abs() <= 0.263) & h.between(0.537, 0.963)).astype(float),
                         "box": ((s.abs() <= 0.373) & h.between(0.427, 1.073)).astype(float), "pk": d.PITCH_KIND.astype(float),
                         "b": d.balls_before.astype(float), "st": d.strikes_before.astype(float), "ph": d.pitcher_hand.astype(float),
                         "bh": d.batter_hand.astype(float), "inn": d.inning.astype(float)}, index=d.index)

RP = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=12, seed=0)

def soft_relabel(lab, upto):
    """2023+ F(season<upto) 로 p̂(rev|행) 학습 → 2019~22 F 비반대투구 행 재라벨. 반환 (oh, 진단 dict)."""
    F = lab[lab.game_type == "F"]; tr = F[F.season >= 2023]; rev = tr.rev.astype(bool).to_numpy()
    diag = {}
    if upto >= 2025 and (tr.season == 2024).any():   # 2023 학습 → 2024 검증 AUC
        from sklearn.metrics import roc_auc_score
        a, b = tr[tr.season == 2023], tr[tr.season == 2024]
        m0 = lgb.train(RP, lgb.Dataset(rev_feats(a), a.rev.astype(int)), 300); p0 = m0.predict(rev_feats(b))
        diag["auc_2023→2024"] = float(roc_auc_score(b.rev.astype(int), p0)); diag["p̂_mean_2024"] = float(p0.mean()); diag["rev_mean_2024"] = float(b.rev.astype(float).mean())
    m = lgb.train(RP, lgb.Dataset(rev_feats(tr), rev.astype(int)), 300)
    tg = ((lab.game_type == "F") & (lab.season <= 2022) & ~lab.rev.astype(bool)).to_numpy()
    p = m.predict(rev_feats(lab[tg]))
    oh = np.eye(3)[lab.y.to_numpy().astype(int)]; idx = np.where(tg)[0]
    before = float((oh[(lab.game_type == "F").to_numpy() & (lab.season <= 2022).to_numpy()] @ PT).mean())
    oh[idx] = (1.0 - p)[:, None] * oh[idx] + p[:, None] * np.array([0.0, 0.0, 1.0])
    after = float((oh[(lab.game_type == "F").to_numpy() & (lab.season <= 2022).to_numpy()] @ PT).mean())
    diag.update(n_target=int(tg.sum()), p̂_mean=float(p.mean()), preF_pt_before=before, preF_pt_after=after,
                F2023plus_pt=float((np.eye(3)[tr.y.to_numpy().astype(int)] @ PT).mean()))
    return oh, diag

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", help="keep|drop|soft|w0.3 (+ '+ctr' 로 시즌중심화: 종민 L83 CTR 의 소프트 one-hot 판, 시즌 리그평균 등급비율을 전역 평균에 맞춤 — F/R 무관)"); ap.add_argument("mdir")
    ap.add_argument("--upto", type=int, default=2025); ap.add_argument("--alpha", type=float, default=2000.0); ap.add_argument("--ref", default="")
    A = ap.parse_args(); t0 = time.time(); ctr = A.mode.endswith("+ctr"); base = A.mode.replace("+ctr", "")
    assert base in ("keep", "drop", "soft", "w0.3"), A.mode
    d = load(labeled_only=False); lab = d[d.y.notna() & (d.season < A.upto)].copy()
    mask = ((lab.game_type == "F") & (lab.season <= 2022)).to_numpy()
    print(f"mode {A.mode} · upto {A.upto} · 라벨행 {len(lab):,} · 2019~22 F 행 {int(mask.sum()):,} ({mask.mean():.3%})", flush=True)
    oh = np.eye(3)[lab.y.to_numpy().astype(int)]; w = np.ones(len(lab)); diag = {}
    if base == "drop": lab, oh, w = lab[~mask], oh[~mask], w[~mask]
    elif base == "w0.3": w = np.where(mask, 0.3, 1.0)
    elif base == "soft": oh, diag = soft_relabel(lab, A.upto)
    if ctr:   # 시즌중심화: 시즌별 (가중) 평균 등급비율을 전역 평균으로 이동 (레짐 수준차 제거 · 표본 유지 · F/R 무관)
        sea = lab.season.to_numpy(); ohw = oh * w[:, None]; gmean = ohw.sum(0) / w.sum(); rows = []
        for s_ in np.unique(sea):
            mk = sea == s_; smean = ohw[mk].sum(0) / w[mk].sum(); oh[mk] = oh[mk] + (gmean - smean)
            rows.append(f"{int(s_)}: pt {float(smean @ PT):.2f}→{float(gmean @ PT):.2f} (ΔS {gmean[0]-smean[0]:+.3f} ΔH {gmean[1]-smean[1]:+.3f} ΔF {gmean[2]-smean[2]:+.3f})")
        diag["ctr"] = " · ".join(rows)
    E = eb_tables_w(lab, oh, w, alpha=A.alpha)
    os.makedirs(A.mdir, exist_ok=True)
    np.savez_compressed(os.path.join(A.mdir, "eb_tables.npz"), prior=E["prior"], prate=E["prate"], pn=E["pn"], pid=E["pid"])
    json.dump({"alpha": E["alpha"], "ac": E["ac"], "n_pitcher": int(len(E["pid"])), "gm": E["gm"].tolist(), "pre23f": A.mode, "upto": A.upto, **diag},
              open(os.path.join(A.mdir, "eb_meta.json"), "w"), ensure_ascii=False)
    mp = os.path.join(A.mdir, "meta.json")
    if os.path.exists(mp):
        m = json.load(open(mp, encoding="utf-8")); m["pre23f"] = A.mode; json.dump(m, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  투수 {len(E['pid'])} · gm {np.round(E['gm'], 4)} · prior 범위 {E['prior'].min():.3f}~{E['prior'].max():.3f}" + (f" · 진단 {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in diag.items()}, ensure_ascii=False)}" if diag else ""), flush=True)
    if A.ref:
        R = np.load(A.ref, allow_pickle=False); rp = pd.Index(R["pid"].astype(str)); loc = rp.get_indexer(pd.Index(E["pid"].astype(str)))
        common = loc >= 0; dpn = np.abs(E["pn"][common] - R["pn"][loc[common]]); dpr = np.abs(E["prate"][common] - R["prate"][loc[common]]).max(1)
        s_new = E["prate"] @ PT; s_ref = R["prate"] @ PT
        print(f"  vs ref: 투수 공통 {common.sum()}/{len(rp)} (ref 에만 {len(rp)-common.sum()}) · pn 변경 {int((dpn > 1e-9).sum())}명 · prate 변경 {int((dpr > 1e-12).sum())}명 · max|Δpn| {dpn.max():.3g} · max|Δprate| {dpr.max():.3g} · prior max|Δ| {np.abs(E['prior'] - R['prior']).max():.3g}")
        ch = common & (dpr > 1e-12)
        if ch.any(): dd = s_new[ch] - s_ref[loc[ch]]; print(f"  변경 투수 점수 이동: mean {dd.mean():+.2f} · min {dd.min():+.2f} · max {dd.max():+.2f} · |Δ|>1점 {int((np.abs(dd) > 1).sum())}명 · >2점 {int((np.abs(dd) > 2).sum())}명")
    print(f"  done {time.time()-t0:.0f}s → {A.mdir}", flush=True)

if __name__ == "__main__": main()
