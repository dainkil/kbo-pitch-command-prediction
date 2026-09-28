"""학습. 모드 val: 2020–2023 학습 → 2024 검증(Total/Pitch/τ), 모드 final: 2020–2024 학습 → model/ 저장 (표는 2019–2024 전체로 → 2025 적용).
사용: python3 train.py val [seeds]   /   python3 train.py final [seeds] [rounds]"""
import sys, os, json, time
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import *
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("seeds", nargs="?", default="0,1,2"); ap.add_argument("rounds", nargs="?", type=int, default=None)
ap.add_argument("--drop", default="b_", help="제거할 피처 접두어(콤마) — hp.py: 타자 b_* 제거가 τ +0.012")
ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--minleaf", type=int, default=1000); ap.add_argument("--leaves", type=int, default=63)
ap.add_argument("--year", type=int, default=2024, help="val: 평가 시즌 Y (학습 2020..Y-1)")
ap.add_argument("--upto", type=int, default=2025, help="final: 표/학습을 season<upto 로 (배포동일 검산용, 기본 2025=실제 제출)")
ap.add_argument("--out", default=None, help="final: 모델 저장 디렉토리 (기본 submit/model)")
ap.add_argument("--tag", default="v3", help="val 산출물 이름표")
ap.add_argument("--combine", default="gbdt", choices=["gbdt", "anchor", "blend"]); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--eb_weight", type=float, default=0.7)
ap.add_argument("--eb_alpha", type=float, default=2000.0); ap.add_argument("--inject", default="", help="SIGSETS 이름 (tm/top3/top5/loc)"); ap.add_argument("--eps", type=float, default=0.1)
ap.add_argument("--skip_gbdt", action="store_true", help="기존 GBDT 모델 재사용, 표·EB·주입만 다시")
ap.add_argument("--neutral", action="store_true", help="주입 z 를 (과거 F비중×이력크기) 셀 안에서 평균 0 으로 중립화")
ap.add_argument("--inject_file", default="", help="미리 계산한 투수별 z 표(csv: pitcher_id,z) 를 inj_z 로 (예: out/pfn_z_2025.csv)")
ap.add_argument("--skill_dir", default="", help="d10: skill_prior.py 산출물 디렉토리 (skill_prior.txt + skill_meta.json) 를 model/ 에 복사 → EB prior 수준을 q_skill(row) 로")
ap.add_argument("--skill_w", type=float, default=1.0, help="skill prior 가중 (0 = 현행 셀 prior)")
ap.add_argument("--center_k", type=float, default=0.0, help="투수-중심 anchor 계수 κ: 학습행 GBDT 점수의 투수 평균 표(s_hat) 를 만들고 추론 시 κ·(s_row − s_hat) 을 목표에 더한다")
A = ap.parse_args(); mode = A.mode; seeds = [int(s) for s in A.seeds.split(",")]
DROP = tuple(p for p in A.drop.split(",") if p); PARAMS.update(learning_rate=A.lr, min_data_in_leaf=A.minleaf, num_leaves=A.leaves)
def sel(feats): return [c for c in feats if not any(c.startswith(p) for p in DROP)]
print(f"mode {mode} seeds {seeds} drop {DROP} params lr={A.lr} minleaf={A.minleaf} leaves={A.leaves}")
T0 = time.time(); tk = lambda: f"[{time.time()-T0:5.0f}s]"
d_all = load(labeled_only=False); tm = load_tm(); print(f"{tk()} loaded {d_all.shape}, tm {tm.shape}")

if mode == "val":
    Y = A.year; Xtr, ytr, Xev, ev, T = build_xy(d_all, tm, list(range(2020, Y)), Y); feats = sel(list(Xtr.columns)); Xtr, Xev = Xtr[feats], Xev[feats]
    print(f"평가 시즌 {Y} · 학습 시즌 {list(range(2020, Y))}")
    print(f"{tk()} Xtr {Xtr.shape} Xev {Xev.shape}"); y = ev.y.to_numpy(); pid = ev.pitcher_id.to_numpy()
    rng = np.random.default_rng(0); msk = rng.random(len(Xtr)) < 0.9
    dtr = lgb.Dataset(Xtr[msk], ytr[msk], categorical_feature=CAT_IDX(feats)); dva = lgb.Dataset(Xtr[~msk], ytr[~msk], reference=dtr)
    Ps = []; iters = []
    for s in seeds:
        m = lgb.train({**PARAMS, "seed": s}, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        P = m.predict(Xev.to_numpy(dtype=np.float32), num_iteration=m.best_iteration); Ps.append(P); iters.append(m.best_iteration)
        score(P, pid, y, f"{tk()} seed {s} it={m.best_iteration}")
        if s == seeds[0]:
            imp = pd.Series(m.feature_importance("gain"), index=feats); imp = (imp/imp.sum()*100).sort_values(ascending=False)
            print("   gain% top15:", imp.round(1).head(15).to_dict())
    Pm = np.mean(Ps, 0); score(Pm, pid, y, f"★ 앙상블 {len(seeds)} seeds")
    # 신인/비신인 τ
    nh = Xev["p_n"].to_numpy() == 0
    g = pd.DataFrame({"pid": pid, "pred": Pm @ PT, "act": PT[y.astype(int)], "nh": nh}).groupby("pid").agg(n=("act", "size"), pred=("pred", "mean"), act=("act", "mean"), nh=("nh", "first"))
    el = g[g.n >= 50]
    from scipy.stats import kendalltau
    for nm, mk in (("신인", el.nh), ("비신인", ~el.nh)):
        q = el[mk]; print(f"    {nm} {len(q)}명 τ {kendalltau(q.act.round(10), q.pred.round(10), variant='b').statistic:+.4f}")
    json.dump({"iters": iters, "seeds": seeds}, open(OUT + f"val_iters_{A.tag}_{Y}.json", "w"))
    np.savez(OUT + f"p_val_{A.tag}_{Y}.npz", P=Pm, y=y, pid=pid)
    if A.tag == "v3" and Y == 2024: json.dump({"iters": iters, "seeds": seeds}, open(OUT + "val_iters.json", "w")); np.savez(OUT + "p_val_2024.npz", P=Pm, y=y, pid=pid)
    print(f"{tk()} val done. best_iters {iters}")

elif mode == "final":
    rounds = A.rounds or int(np.mean(json.load(open(OUT + "val_iters.json"))["iters"])*1.15)
    mdir = A.out or os.path.join(os.path.dirname(__file__), "submit", "model"); os.makedirs(mdir, exist_ok=True)
    prev_models = json.load(open(os.path.join(mdir, "meta.json")))["models"] if A.skip_gbdt else None   # save_tables 가 meta.json 을 덮어쓰기 전에 읽는다
    U = A.upto; Xtr, ytr, _, _, _ = build_xy(d_all, tm, list(range(2020, U))); feats = sel(list(Xtr.columns)); Xtr = Xtr[feats]
    print(f"학습 시즌 {list(range(2020, U))} · 표 season<{U} → {mdir}")
    print(f"{tk()} Xtr {Xtr.shape}  rounds {rounds}")
    lab_all = d_all[d_all.y.notna()]
    T = build_tables(lab_all[lab_all.season < U], d_all[d_all.season < U], tm[tm.season < U], U); T["meta"]["features"] = sel(T["meta"]["features"]); save_tables(T, mdir)   # 적용용 표: season<U 전체
    assert T["meta"]["features"] == feats
    if A.skip_gbdt:
        models = prev_models; print(f"{tk()} GBDT 재사용 {models}")
    else:
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=CAT_IDX(feats)); models = []
        for s in seeds:
            m = lgb.train({**PARAMS, "seed": s}, dtr, rounds); f = f"lgb_seed{s}.txt"; m.save_model(os.path.join(mdir, f)); models.append(f)
            print(f"{tk()} saved {f}")
    if A.combine != "gbdt":
        labU = lab_all[lab_all.season < U]; E = eb_tables(labU, alpha=A.eb_alpha); save_eb(E, mdir)
        T["meta"].update(mode=A.combine, lam=A.lam, eb_weight=A.eb_weight, eb_alpha=A.eb_alpha); print(f"{tk()} EB 표 저장 (α={A.eb_alpha}, 투수 {len(E['pid'])}) · mode={A.combine}")
        if A.inject or A.inject_file:
            sd_e = eb_level_sd(labU, E); z = pd.Series(0.0, index=T["pitcher"].index)
            if A.inject: z = z + inject_z(T["pitcher"], SIGSETS[A.inject], sd_e, neutral=A.neutral)
            if A.inject_file: z = z + pd.read_csv(A.inject_file, dtype={"pitcher_id": str}).set_index("pitcher_id")["z"].reindex(T["pitcher"].index).fillna(0.0)
            T["pitcher"]["inj_z"] = z; save_tables(T, mdir)
            T["meta"].update(inject=A.inject or os.path.basename(A.inject_file), inject_eps=float(A.eps * sd_e), neutral=A.neutral); print(f"{tk()} 주입 {A.inject} ε={A.eps} × sd {sd_e:.3f} = {A.eps*sd_e:.3f}점/σ")
    if A.center_k:
        Ps = np.zeros((len(Xtr), 3))
        for f in models: Ps += lgb.Booster(model_file=os.path.join(mdir, f)).predict(Xtr.to_numpy(dtype=np.float32))
        s_tr = (Ps / len(models)) @ PT; lab_tr = lab_all.loc[Xtr.index]
        sh = pd.Series(s_tr, index=lab_tr.pitcher_id.to_numpy()).groupby(level=0).mean(); T["pitcher"]["s_hat"] = sh.reindex(T["pitcher"].index)
        save_tables(T, mdir); T["meta"]["center_k"] = A.center_k; print(f"{tk()} s_hat 표 {sh.notna().sum()} 투수 · center_k={A.center_k} · s_tr mean {s_tr.mean():.2f} sd {s_tr.std():.2f}")
    if A.skill_dir:
        import shutil; sm = json.load(open(os.path.join(A.skill_dir, "skill_meta.json"), encoding="utf-8")); sm["w"] = A.skill_w
        shutil.copy(os.path.join(A.skill_dir, sm["model"]), os.path.join(mdir, sm["model"])); json.dump(sm, open(os.path.join(mdir, "skill_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        T["meta"].update(skill=os.path.basename(A.skill_dir.rstrip("/")), skill_w=A.skill_w, skill_variant=sm["variant"]); print(f"{tk()} skill prior {sm['variant']} ({A.skill_dir}) w={A.skill_w} → {mdir}")
    T["meta"]["models"] = models; T["meta"]["rounds"] = rounds; T["meta"]["params"] = {k: v for k, v in PARAMS.items() if k != "num_threads"}; json.dump(T["meta"], open(os.path.join(mdir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{tk()} final done → {mdir}")
