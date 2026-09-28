"""표(이력) 생성 — 학습·추론 공통. 시즌 Y 의 표는 season<Y 데이터로만 만든다 (누수 없음).
script.py 의 apply_tables() 가 이 표를 행 단위로 join 한다."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eda"))
from _common import load, D, OUT, CLASSES, PT, prior, score, fold, head, sub
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "submit"))
from script import apply_tables, CTX, EB, cells, eb_predict, NCELL, anchor, inject

TM_NUM = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]
PK_CODES = [1, 2, 4, 5, 7, 9, 10]   # 직구·커브·슬라이더·포크·체인지업·커터·투심 (특수구·너클·싱커는 표본 <400 → 제외)
ALPHA_PK = 100.0                    # 구종 비율 수축 (리그 비율로)

def load_tm():
    tm = pd.read_csv(D + "train_trackman.csv", dtype={"row_id": str}, usecols=["season", "pitcher_id", "pitch_type_group"] + TM_NUM)
    tm = tm[tm.pitcher_id.notna() & tm.pitcher_id.str.startswith("P_")].copy()
    for g in ("fastball", "breaking", "offspeed"): tm[f"mix_{g}"] = (tm.pitch_type_group == g).astype(float)
    return tm

def tm_profile(tm):
    g = tm.groupby("pitcher_id")
    f = g[TM_NUM].mean().add_prefix("tm_"); f["tm_n"] = g.size().astype(float)
    for c in ("rel_height", "rel_side", "rel_speed", "extension"): f[f"tm_sd_{c}"] = g[c].std()
    for c in ("fastball", "breaking", "offspeed"): f[f"tm_mix_{c}"] = g[f"mix_{c}"].mean()
    fb = tm[tm.pitch_type_group == "fastball"].groupby("pitcher_id")[["rel_speed", "spin_rate", "induced_vert_break"]].mean().add_prefix("tm_fb_")
    return f.join(fb)

def eb_table(df, key, alpha, pri, w=None):
    w = np.ones(len(df)) if w is None else np.asarray(w, float); yy = df.y.to_numpy()
    t = pd.DataFrame({"k": df[key].to_numpy(), "sh": (yy == 0)*w, "he": (yy == 1)*w, "fa": (yy == 2)*w, "n": w})
    g = t.groupby("k").sum()
    for i, c in enumerate(EB): g[c] = (g[c] + alpha*pri[i])/(g.n + alpha)
    return g

def build_tables(lab, allrows, tm, Y, alpha_p=200, alpha_b=300, alpha_gt=100, h=2.0):
    """lab: 라벨 행(season<Y), allrows: 전체 행(season<Y, 미라벨 포함), tm: Trackman(season<Y). Y: 표를 적용할 시즌."""
    pri = prior(lab)
    p = eb_table(lab, "pitcher_id", alpha_p, pri).add_prefix("p_")
    p1 = eb_table(lab[lab.season == Y-1], "pitcher_id", alpha_p, pri).add_prefix("p1_")
    pr = eb_table(lab, "pitcher_id", alpha_p, pri, 0.5**((Y - lab.season.to_numpy(float))/h)).add_prefix("pr_")
    # 잔차 실력 (종민 07): 최근 2시즌 cnt×game_type 평균을 걷어낸 pt 잔차의 투수 EB(0 수축)
    src = lab[lab.season >= Y-2]
    ctab = src.groupby(["cnt", "game_type"]).pt.mean().rename("c").reset_index()
    q = lab[["pitcher_id", "cnt", "game_type", "pt"]].merge(ctab, on=["cnt", "game_type"], how="left")
    g = pd.DataFrame({"k": q.pitcher_id.to_numpy(), "r": (q.pt - q.c).to_numpy()}).groupby("k").r.agg(["sum", "size"])
    skill = (g["sum"]/(g["size"] + alpha_p)).rename("p_skill")
    # 투수×game_type
    pg = {}
    for gt in ("R", "F"):
        pg[gt] = eb_table(lab[lab.game_type == gt], "pitcher_id", alpha_gt, pri)[EB].add_prefix(f"pgt{gt}_")
    # 전체 행 기반: 과거 F 비중, 총 투구수(미라벨 포함)
    fa = allrows.groupby("pitcher_id").agg(p_fshare=("game_type", lambda s: float((s == "F").mean())), p_n_all=("game_type", "size"))
    # d10: 구종 구성 (PITCH_KIND, 전체 행 100% 커버) — 리그 비율로 α_pk 수축. 소이력 prior 공변량(skill prior) 전용, GBDT 피처 아님
    pk = pd.get_dummies(allrows.PITCH_KIND.astype(int)).reindex(columns=PK_CODES, fill_value=0).astype(float)
    pk.index = allrows.pitcher_id.to_numpy(); lg_mix = pk.mean(0).to_numpy(); gpk = pk.groupby(level=0).agg(["sum", "size"])
    npk = gpk.loc[:, (PK_CODES[0], "size")].to_numpy(float)
    pkmix = pd.DataFrame({f"pk_mix_{c}": (gpk.loc[:, (c, "sum")].to_numpy(float) + ALPHA_PK * lg_mix[i]) / (npk + ALPHA_PK) for i, c in enumerate(PK_CODES)}, index=gpk.index)
    prof = tm_profile(tm)
    # v2: 투수의 과거 투구 위치·반대투구·존이탈 이력 (라벨의 원재료; 규칙상 학습기간 이력 집계는 허용)
    l2 = lab.assign(revf=lab.rev.astype(float), abs_side=lab.plate_loc_side.abs(),
                    outf=((lab.plate_loc_side.abs() > 0.373) | (lab.plate_loc_height < 0.427) | (lab.plate_loc_height > 1.073)).astype(float))
    g = l2.groupby("pitcher_id"); n = g.size(); rp, op = l2.revf.mean(), l2.outf.mean()
    ext = pd.DataFrame({"p_rev": (g.revf.sum() + alpha_p*rp)/(n + alpha_p), "p_out": (g.outf.sum() + alpha_p*op)/(n + alpha_p),
                        "p_side_mean": g.plate_loc_side.mean(), "p_side_sd": g.plate_loc_side.std(), "p_h_mean": g.plate_loc_height.mean(),
                        "p_h_sd": g.plate_loc_height.std(), "p_abs_side": g.abs_side.mean()})
    for gt in ("R", "F"):
        s = l2[l2.game_type == gt]; gg = s.groupby("pitcher_id"); ext[f"pgt{gt}_rev"] = (gg.revf.sum() + alpha_gt*s.revf.mean())/(gg.size() + alpha_gt)
    extra_cols = ["p_rev", "p_out", "p_side_mean", "p_side_sd", "p_h_mean", "p_h_sd", "p_abs_side"]
    P = p.join([p1, pr, skill, pg["R"], pg["F"], fa, prof, ext, miss_table(lab), pkmix], how="outer")
    P.index.name = "pitcher_id"
    B = eb_table(lab, "batter_id", alpha_b, pri).add_prefix("b_")
    # d10: 타자 레벨 (전체 행) — 상대 타자의 F 출전 비중·출전량 = 이 행 투수의 수준 증거 (skill prior 전용)
    bl = allrows.groupby("batter_id").agg(b_fshare=("game_type", lambda s: float((s == "F").mean())), b_n_all=("game_type", "size"))
    B = B.join(bl, how="outer"); B.index.name = "batter_id"
    tm_cols = list(prof.columns)
    feats = CTX + ["season"] + [f"{t}_{c}" for t in ("p", "p1", "pr") for c in EB + ["n"]] + ["p_skill"] + [f"pgt_{c}" for c in EB] \
            + ["p_fshare", "p_n_all"] + [f"b_{c}" for c in EB + ["n"]] + tm_cols + extra_cols + ["pgt_rev"]
    meta = {"prior": pri.tolist(), "features": feats, "tm_cols": tm_cols, "extra_cols": extra_cols, "Y": int(Y), "alpha": {"p": alpha_p, "b": alpha_b, "gt": alpha_gt, "h": h}}
    return {"pitcher": P, "batter": B, "meta": meta}

def save_tables(T, mdir):
    os.makedirs(os.path.join(mdir, "tables"), exist_ok=True)
    T["pitcher"].to_csv(os.path.join(mdir, "tables", "pitcher.csv")); T["batter"].to_csv(os.path.join(mdir, "tables", "batter.csv"))
    json.dump(T["meta"], open(os.path.join(mdir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

def build_xy(d_all, tm, seasons, Y_eval=None):
    """확장창: 각 시즌 Y 의 행은 season<Y 표로 피처화. 반환 (Xtr, ytr, Xev, ev, T_eval)."""
    lab_all = d_all[d_all.y.notna()]
    parts = {}; T_eval = None
    for Y in sorted(set(seasons) | ({Y_eval} if Y_eval else set())):
        lab = lab_all[lab_all.season < Y]; allr = d_all[d_all.season < Y]; tmY = tm[tm.season < Y]
        T = build_tables(lab, allr, tmY, Y)
        cur = lab_all[lab_all.season == Y]
        parts[Y] = apply_tables(cur, T)
        if Y == Y_eval: T_eval = T
    Xtr = pd.concat([parts[Y] for Y in seasons]); ytr = lab_all.loc[Xtr.index, "y"].to_numpy()
    if Y_eval is None: return Xtr, ytr, None, None, None
    return Xtr, ytr, parts[Y_eval], lab_all.loc[parts[Y_eval].index], T_eval

def eb_tables(lab, alpha=2000.0, ac=200.0, min_ps=50):
    """L2 표 (train_L2.py 재구현). lab = 라벨 행(season<Y). prior 타깃 = 투수×시즌(50투구+) 등급비율의 셀 평균."""
    oh = np.eye(3)[lab.y.to_numpy().astype(int)]; gm = oh.mean(0)
    g = pd.DataFrame(oh, index=lab.pitcher_id.to_numpy()).groupby(level=0).agg(["sum", "size"])
    pid = g.index.to_numpy().astype("U32"); pn = g.loc[:, (0, "size")].to_numpy(float); prate = g.loc[:, (slice(None), "sum")].to_numpy(float) / pn[:, None]
    ps = pd.DataFrame(oh, index=pd.MultiIndex.from_arrays([lab.pitcher_id, lab.season])).groupby(level=[0, 1]).agg(["mean", "size"])
    keep = ps.loc[:, (0, "size")].to_numpy() >= min_ps; rates = ps.loc[:, (slice(None), "mean")].to_numpy(float)[keep]; key = ps.index[keep]
    sel = pd.Series(np.arange(len(key)), index=key).reindex(pd.MultiIndex.from_arrays([lab.pitcher_id, lab.season])).to_numpy(); ok = ~np.isnan(sel)
    K = cells(lab)[ok]; R = rates[sel[ok].astype(int)]
    cnt = np.bincount(K, minlength=NCELL).astype(float); sm = np.column_stack([np.bincount(K, weights=R[:, j], minlength=NCELL) for j in range(3)])
    prior = (sm + ac * gm) / (cnt[:, None] + ac); prior /= prior.sum(1, keepdims=True)
    return {"prior": prior, "prate": prate, "pn": pn, "pid": pid, "pidx": pd.Index(pid.astype(object)), "alpha": float(alpha), "ac": float(ac), "gm": gm}

# ── 빗나간 크기(등급이 버린 연속 정보) — Statcast 4존(Heart/Shadow/Chase/Waste) + 경계 밖 거리. 정투구만. 60 배치셀 역할보정 잔차 EB.
OUT_X, OUT_Y0, OUT_Y1, BALL = 0.373, 0.427, 1.073, 0.073
def miss_table(lab, alpha=2000.0, alpha_sd=500.0):
    nr = lab.REVERSE_YN.ne("T").to_numpy(); x = lab.plate_loc_side.to_numpy(float); y = lab.plate_loc_height.to_numpy(float)
    out = np.maximum(np.abs(x) - OUT_X, np.maximum(OUT_Y0 - y, y - OUT_Y1)); cell = cells(lab); pid = lab.pitcher_id.to_numpy()
    M = {"far15": out > 0.15, "far30": out > 0.30, "far50": out > 0.50, "waste": out > BALL, "chase": (out > 0) & (out <= BALL),
         "outpos": np.maximum(out, 0.0), "outsq": np.maximum(out, 0.0) ** 2, "cdist": np.sqrt(x * x + (y - 0.75) ** 2)}
    res = {}
    for k, v in M.items():
        vv = v.astype(float)[nr]; cc = cell[nr]; pp = pid[nr]
        cm = np.bincount(cc, weights=vv, minlength=NCELL) / np.maximum(np.bincount(cc, minlength=NCELL), 1)
        g = pd.DataFrame({"p": pp, "r": vv - cm[cc], "v": vv}).groupby("p").agg(rs=("r", "sum"), n=("r", "size"), vm=("v", "mean"))
        res[f"mr_{k}"] = g.rs / (g.n + alpha); res[f"raw_{k}"] = g.vm
    g2 = pd.DataFrame({"p": pid[nr], "x": x[nr], "y": y[nr]}).groupby("p").agg(sx=("x", "std"), sy=("y", "std"), n=("x", "size"))
    for k in ("sx", "sy"):
        gm = g2[k].mean(); res[f"sd_{k}"] = (g2[k].fillna(gm) * g2.n + alpha_sd * gm) / (g2.n + alpha_sd)
    T = pd.DataFrame(res); T.index.name = "pitcher_id"; return T

def neutralize(z, Tp):
    """집단 간 이동 제거: 학습표의 (과거 F 비중 4구간 × 이력 크기 3구간, 고정 경계) 셀 안에서 평균 0. 행 독립(표 파생)."""
    f = Tp.p_fshare.reindex(z.index).fillna(0.0).to_numpy(); n = Tp.p_n.reindex(z.index).fillna(0.0).to_numpy()
    key = np.digitize(f, [0.1, 0.5, 0.9]) * 10 + np.digitize(n, [200, 1000]); zz = z.to_numpy(float).copy()
    for k in np.unique(key):
        m = key == k; zz[m] = zz[m] - zz[m].mean()
    return pd.Series(zz, index=z.index)

SIGSETS = {"tm": {"tm_mix_offspeed": 1, "tm_horz_break": 1, "tm_spin_rate": -1, "tm_rel_speed": -1},
           "top3": {"pgtR_fa": -1, "p_skill": 1, "pgtR_sh": 1},
           "top5": {"pgtR_fa": -1, "p_skill": 1, "pgtR_sh": 1, "p_fa": -1, "pgtR_rev": -1},
           "loc": {"p_h_sd": -1, "pgtR_rev": -1, "p_out": -1},
           "miss": {"mr_far50": -1, "mr_chase": 1}}   # E1: 큰 미스(Waste) − / 살짝 밖(Chase) +, 중립화 ε=0.4

def inject_z(Tp, sigs, eb_sd, neutral=False):
    """투수표 Tp 에서 z(신호) 의 부호합을 만들어 EB 점수 σ 단위로 정규화. 평균/표준편차는 학습표(투수 200투구+)에서 → 행 독립.
    반환: Series(pitcher_id → inj_z), 단위 = 'EB 점수 sd' (taulab 의 z(EB)+ε·Σsign·z 와 동일 스케일)."""
    ref = Tp[Tp.p_n.fillna(0) >= 200]; z = pd.Series(0.0, index=Tp.index)
    for c, sg in sigs.items():
        v = Tp[c].astype(float); med = ref[c].median(); mu, sd = ref[c].mean(), ref[c].std()
        z = z + sg * ((v.fillna(med) - mu) / (sd if sd > 0 else 1.0))
    return neutralize(z, Tp) if neutral else z   # 점수 단위 환산은 inject(eps_pts = ε · eb_sd)

def eb_level_sd(lab, E):
    """학습 행에서 EB 예측 점수의 투수간 sd (50투구+) — 주입 스케일 ε 를 점수로 바꾸는 상수(학습 파생)."""
    s = pd.Series(eb_predict(lab, E) @ PT).groupby(lab.pitcher_id.to_numpy()).agg(["size", "mean"]); s = s[s["size"] >= 50]
    return float(s["mean"].std())

def save_eb(E, mdir):
    np.savez_compressed(os.path.join(mdir, "eb_tables.npz"), prior=E["prior"], prate=E["prate"], pn=E["pn"], pid=E["pid"])
    json.dump({"alpha": E["alpha"], "ac": E["ac"], "n_pitcher": int(len(E["pid"])), "gm": E["gm"].tolist()}, open(os.path.join(mdir, "eb_meta.json"), "w"))

CAT_IDX = lambda feats: [feats.index("base_state")]
PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=12, verbose=-1, seed=0)   # 공통규칙: dain 12스레드
