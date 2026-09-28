"""L7 학습 — L6 + **대리등급 목표값**. 라벨 없는 253,122행을 제구력 추정에 쓴다.
프로토콜: 2019~2023 학습 → 2024 홀드아웃 → 2019~2024 전체 재학습. 학습 피처는 LOSO."""
import os, sys, json, numpy as np, pandas as pd
SUB=os.environ.get("SUBDIR","/home/ubuntu/workspace/산출물/종민/submit_L7")
sys.path.insert(0, SUB)
import lightgbm as lgb
from scipy.stats import kendalltau
from script import (prep, build_tables, feats, save_tables, load_tables, predict, make_bundle, cells, NCELL, PHY, CLS, anchor, A_P, BETA, GLOBAL_PRIOR, RECENT_HIST, ENV_ADJ, SEA_ADJ, SEA_A, SEA_NMAX, SEA_KEYS, SEA_W, LVL_MODE, LVL_TEAM_G, HAND_G, EXP_G, HTM_G, USE_G, USE_KEYS, ENVAX, envax_spec, envax_key, LOCSD_G, CELL_G, BAT_G, TRG_G, TRF_G, RGSPLIT, RG_G, NPRI_G, LVLN_G, LVLF_G, RGV, RGV_H, RGV_AR, FRG_G, TINV_G, TINV_P, ANCHOR_MODE, COMP_G, USE_W, USE_RONLY)
D="/home/ubuntu/workspace/data/"
OUT=os.environ.get("MODEL_OUT", SUB+"/model")
NTH=int(os.environ.get("NTH","12")); AC=200.0
A_T=float(os.environ.get("A_T","4000")); BETA_T=float(os.environ.get("BETA_T","0.6"))
MAG_TH=float(os.environ.get("MAG_TH","0.50"))   # 빗나간 크기 임계값(m)
RECENT_N=int(os.environ.get("RECENT_N","0"))   # >0 이면 prior 표를 최근 N 시즌으로만
USE_PROXY=os.environ.get("USE_PROXY","1")=="1"
HOLD_Y=int(os.environ.get("HOLD_Y","2024"))
# ★★★★ 대리등급 표(anchor 목표의 투수 이력)의 시즌 처리 — RECENT_HIST 는 이 표를 거치지 않는다.
#   측정(2024 평가·단순 EB): 최근1시즌 τ +0.2723 · 최근2 +0.2485 · **전체6 +0.2362** · 시즌중심화 +0.2616
#   τ(이력 평균시즌, 과거EB) = −0.1652 → 옛 시즌이 많은 투수의 EB 가 부풀려진다(베테랑 편향).
#   원인: Failure 비율 2019 0.485 → 2022~24 0.527 로 레짐이 다르고, 2023 은 기록규칙이 바뀌었다
#         (같은 F 투수 70명 반대투구율 2022 4.62% → 2023 29.24%).
NSEED=int(os.environ.get("NSEED","1"))   # ★ q 시드 앙상블 개수 (Player 불변 · Pitch 전용)
PGR_HIST=int(os.environ.get("PGR_HIST","0"))      # 하드 창: 최근 N 시즌만
PGR_CTR=os.environ.get("PGR_CTR","0")=="1"        # 시즌중심화: 레짐 수준차 제거하고 표본은 유지
PGR_DECAY=float(os.environ.get("PGR_DECAY","1"))  # 지수 최근성 가중 (1=균등, 0.5=1년당 절반)   # ★ 롤링 백테스트용 홀드아웃 연도
AC_P=float(os.environ.get("AC_P","200"))
G=np.array([100.,30.,0.])

def load_all():
    tr=pd.read_csv(D+"train.csv",dtype={"row_id":str},low_memory=False); tr.columns=[c.lstrip("﻿") for c in tr.columns]
    oc=pd.read_csv(D+"train_outcomes.csv",dtype={"row_id":str}); oc.columns=[c.lstrip("﻿") for c in oc.columns]
    tm=pd.read_csv(D+"train_trackman.csv",dtype={"row_id":str},low_memory=False,
                   usecols=["row_id","pitch_type_group"]+PHY); tm.columns=[c.lstrip("﻿") for c in tm.columns]
    d=tr.merge(oc,on="row_id",validate="one_to_one").merge(
        tm[tm.row_id.notna()].drop_duplicates("row_id"),on="row_id",how="left",validate="one_to_one")
    d=prep(d); d["y"]=d.control_grade.map(CLS); d["lab"]=d.y.notna()
    d["pt"]=d.y.map({0:100.,1:30.,2:0.})
    d["rev"]=(d.REVERSE_YN=="T").astype(float); d["absside"]=d.plate_loc_side.abs()
    XS,YS0,YS1=0.373,0.427,1.073
    _x=d.plate_loc_side.to_numpy(); _y=d.plate_loc_height.to_numpy()
    _out=np.maximum(np.abs(_x)-XS,np.maximum(YS0-_y,_y-YS1))
    d["mag"]=(_out>MAG_TH).astype(float)
    d["isF"]=(d.game_type=="F").astype(int)      # ★ 빗나간 크기 (Shadow 밖 MAG_TH m 초과)
    d["S"]=(d.STRK_YN=="T").astype(np.int64); d["Bl"]=(d.BALL_YN=="T").astype(np.int64)
    Rc,_=pd.factorize(pd.to_numeric(d.RESULT_ID,errors="coerce").fillna(-1))
    Kc,_=pd.factorize(pd.to_numeric(d.PITCH_KIND,errors="coerce").fillna(-1))
    nR=int(Rc.max())+1; nK=int(Kc.max())+1
    d["pkey"]=((((d.S*2+d.Bl).to_numpy()*nR+Rc)*nK+Kc)*2+d.rev.to_numpy().astype(np.int64))*12+d.cnt.to_numpy()
    for j in range(3): d[f"c{j}"]=(d.y==j).astype(float)
    return d

def proxy_pitcher(dd):
    """라벨 행으로 대리표 적합 → 모든 행(라벨+결측)의 등급을 채워 투수별 합/개수"""
    L=dd[dd.lab]; mu=L.pt.mean(); NK=int(dd.pkey.max())+1
    c=np.bincount(L.pkey.to_numpy(),minlength=NK).astype(float)
    s=np.bincount(L.pkey.to_numpy(),weights=L.pt.to_numpy(),minlength=NK)
    PX=(s+30.*mu)/(c+30.)
    r2=1-((L.pt.to_numpy()-PX[L.pkey.to_numpy()])**2).mean()/L.pt.var()
    fill=np.where(dd.lab.to_numpy(),np.nan_to_num(dd.pt.to_numpy()),PX[dd.pkey.to_numpy()])
    sea=dd.season.to_numpy()
    if PGR_CTR:                      # ★ 시즌 수준차 제거 (레짐 균질화) — 표본은 버리지 않는다
        sm=pd.Series(fill).groupby(sea).transform("mean").to_numpy()
        fill=fill-sm+float(fill.mean())
    w=np.ones(len(fill))
    if PGR_DECAY!=1.0:               # ★ 지수 최근성 가중 — 표본을 버리지 않고 최근을 더 본다
        w=PGR_DECAY**(int(sea.max())-sea).astype(float)
    F=pd.DataFrame({"p":dd.pitcher_id.to_numpy(),"v":fill*w,"w":w})
    g=F.groupby("p").agg(**{"sum":("v","sum"),"size":("w","sum")})
    return g, float(r2)

def sea_tables(P, T, sea_only=False):
    """★ 환경 중심화 잔차의 투수별 합/개수 + 셀 prior + 되더할 상수.
    축은 SEA_KEYS 로 정한다. SEA_W<1 이면 부분 중심화(팀 전력 신호를 일부 남긴다)."""
    key=[k.strip() for k in SEA_KEYS.split(",") if k.strip()]
    if sea_only: key=["season"]
    g=P.groupby(key).pt.transform("mean")
    gm_all=float(P.pt.mean())
    r=(P.pt-gm_all-SEA_W*(g-gm_all)).to_numpy()      # W=1 완전 중심화 · W=0 무보정
    print(f"    중심화 축 {key} · 강도 {SEA_W} · 셀 {P.groupby(key).ngroups}",flush=True)
    gp=pd.DataFrame({"p":P.pitcher_id.to_numpy(),"r":r}).groupby("p").r.agg(["sum","size"])
    T["sea_sum"]=gp["sum"].to_numpy(float); T["sea_n"]=gp["size"].to_numpy(float)
    T["sea_idx"]=gp.index.to_numpy()
    # 셀 prior: 투수×시즌 평균 잔차를 배치셀로 집계
    ps=P.assign(r=r).groupby(["pitcher_id","season"]).agg(n=("r","size"),m=("r","mean"))
    ps=ps[ps.n>=50]
    ix=pd.MultiIndex.from_arrays([P.pitcher_id,P.season])
    sel=pd.Series(np.arange(len(ps)),index=ps.index).reindex(ix).to_numpy()
    ok=~np.isnan(sel); K=cells(P)[ok]
    c=np.bincount(K,minlength=NCELL).astype(float)
    sm=np.bincount(K,weights=ps.m.to_numpy()[sel[ok].astype(int)],minlength=NCELL)
    T["sea_prior"]=(sm+200.*0.0)/(c+200.)
    yr=int(P.season.max())
    T["sea_c"]=float(P[P.season==yr].pt.mean())      # 순위에 무영향 · Pitch 눈금용
    return T


def env_center(P):
    """★ 시즌×레벨(R/F) 평균을 이력에서 제거한다. 반환: 잔차가 들어간 복사본 + 목표 환경값
    목표 환경 = **최근 시즌**의 R/F 평균 (2025 의 환경 추정치)"""
    g=P.groupby(["season","isF"]).pt.transform("mean")
    Q=P.copy(); Q["pt"]=P.pt-g
    yr=int(P.season.max()); L=P[P.season==yr]
    eR=float(L[L.isF==0].pt.mean()); eF=float(L[L.isF==1].pt.mean()) if (L.isF==1).any() else eR
    return Q, eR, eF


def hand_dev_of(P):
    """★ 플래툰(투수손×타자손) 환경 편차 — 최근 시즌의 (조합 평균 − 리그 평균).
    63번: 효과 sd 1.310 이고 최근 **증가**(+0.480) → EB 가 과소 흡수 → γ>0 이 교정"""
    yr=int(P.season.max()); L=P[P.season==yr]; mu=float(L.pt.mean())
    ph=pd.to_numeric(L.pitcher_hand,errors="coerce").fillna(1).clip(1,2).astype(int)
    bh=pd.to_numeric(L.batter_hand,errors="coerce").fillna(1).clip(1,2).astype(int)
    k=((ph-1)*2+(bh-1)).to_numpy()
    dev=np.zeros(4)
    for j in range(4):
        m=k==j
        if m.sum()>=500: dev[j]=float(L.pt.to_numpy()[m].mean()-mu)
    return dev


def exp_of(P):
    """★ 경력 궤적 — 그 투수의 경과 시즌수(마지막 시즌 − 첫 시즌 + 1). Ridge 계수 −0.405"""
    yr=int(P.season.max())
    g=P.groupby("pitcher_id").season.agg(["min","size"])
    v=(yr-g["min"]+1).astype(float)
    v=v[g["size"]>=50]
    return v.index.to_numpy(), v.to_numpy(float), float(v.mean()), float(v.std())


def team_dev_of(P):
    """★ 평가면 측 팀 환경 편차 — **최근 시즌**의 (팀 평균 − 리그 평균)"""
    yr=int(P.season.max()); L=P[P.season==yr]
    mu=float(L.pt.mean())
    g=L.groupby("pitcher_team_id").pt.agg(["size","mean"])
    g=g[g["size"]>=500]
    dev=(g["mean"]-mu)
    return dev.index.astype(str).to_numpy(), dev.to_numpy(float), mu


def usage_fit(dl, seasons, Hstd, rh=2, lam=50.0):
    """★ 경기 상태 → 실력 패널 회귀 (행분산 벌점 릿지).
    패널 yp: 이력(<yp, 시즌중심화 EB α=A_T) 대비 잔차 r_p 를 투수별 상태 평균 z̄_p(×w_p)에 회귀.
    벌점 = USE_COST·Cov_row(z) + 0.1·I → 행 단위로 비싼 방향(투수 내 변동이 큰 변수)을 덜 쓴다 = Pitch 손실 최소화.
    표준화 상수는 Hstd(이력 행)에서만. 계수는 학습부 시즌에서만."""
    U=list(USE_KEYS); cost=float(os.environ.get("USE_COST","30")); aT=float(os.environ.get("A_T","4000")); aW=float(os.environ.get("USE_AW","0")) or aT
    mus=np.array([float(pd.to_numeric(Hstd[k],errors="coerce").mean()) for k in U])
    sds=np.array([max(float(pd.to_numeric(Hstd[k],errors="coerce").std()),1e-9) for k in U])
    gm=float(dl.pt.mean()); ptc=dl.pt-dl.groupby("season").pt.transform("mean")+gm
    D=dl.assign(_ptc=ptc)
    Xs,rs,Cs=[],[],[]
    for s_ in seasons:
        H=D[D.season<=s_-1]
        if not len(H): continue
        mu=float(H._ptc.mean()); g=H.groupby("pitcher_id")._ptc.agg(["sum","size"])
        eb=(g["sum"]+aT*mu)/(g["size"]+aT)
        S=D[D.season==s_]
        Z=(S[U].apply(pd.to_numeric,errors="coerce").fillna(pd.Series(mus,index=U)).to_numpy(float)-mus)/sds
        if USE_RONLY: Z=Z*(1.0-S.isF.to_numpy(float))[:,None]
        df=pd.DataFrame(Z,columns=U,index=S.index); df["p"]=S.pitcher_id.to_numpy(); df["y"]=S.pt.to_numpy(); df["f"]=S.isF.to_numpy(float)
        ag=df.groupby("p").agg(n=("y","size"),y=("y","mean"),f=("f","mean"),**{k:(k,"mean") for k in U}); ag=ag[ag.n>=50]
        if len(ag)<30: continue
        e=ag.index.map(eb).fillna(mu).to_numpy(float); hn=ag.index.map(g["size"]).fillna(0.).to_numpy(float)
        X=ag[U].to_numpy(float)*((aW/(hn+aW))[:,None] if USE_W else 1.0)
        r_=ag.y.to_numpy(float)-e
        if os.environ.get("USE_FSADJ","0")=="1":      # ★ 97번: F 비중의 선형효과(시즌마다 부호가 바뀐다)를 잔차에서 먼저 제거 → 스트라이크 계수 오염 방지
            A_=np.vstack([np.ones(len(ag)),ag.f.to_numpy(float)]).T; r_=r_-A_@np.linalg.lstsq(A_,r_,rcond=None)[0]
        if os.environ.get("USE_SC","0")=="1":         # ★ 감사: 전체 중심화만 하면 시즌간 수준차(r 의 시즌 평균·z̄ 의 시즌 평균)가 계수에 샌다 → 시즌 고정효과 제거
            X=X-X.mean(0); r_=r_-r_.mean()
        Xs.append(X); rs.append(r_); Cs.append(np.atleast_2d(np.cov(Z.T)))
    X=np.vstack(Xs); r=np.concatenate(rs); Xc=X-X.mean(0); rc=r-r.mean()
    Pn=(np.mean(Cs,axis=0)*cost+0.1*np.eye(len(U))) if cost>0 else float(os.environ.get("USE_LAM","1"))*np.eye(len(U))
    b=np.linalg.solve(Xc.T@Xc+Pn,Xc.T@rc)
    return b,mus,sds

def envax_tables(P, T, minn=300):
    """★ 평가면 환경 dev — **최근 1시즌**의 (그 값의 평균 − 리그 평균). 학습부만 본다."""
    yr=int(P.season.max()); L=P[P.season==yr]; mu=float(L.pt.mean())
    for ax,_g in envax_spec():
        k=envax_key(L,ax)
        g=pd.DataFrame({"k":k,"v":L.pt.to_numpy(float)}).groupby("k").v.agg(["size","mean"])
        g=g[g["size"]>=minn]; dv=(g["mean"]-mu)
        T["envax_i__"+ax]=dv.index.astype(str).to_numpy(); T["envax_v__"+ax]=dv.to_numpy(float)
        print(f"  환경축 {ax}: {len(dv)}값 · dev {dv.min():+.3f}~{dv.max():+.3f} · sd {dv.std():.3f}",flush=True)
    return T


def locsd_of(H, a=200.0):
    """★ 투수별 로케이션 산포(수축) — sqrt(sd_side^2 + sd_height^2). 표는 학습 이력 행에서만.
    산포는 라벨보다 빨리 안정하므로 수축을 약하게(a=200) 둔다."""
    g=H.groupby("pitcher_id"); n=g.size()
    sd=g[["plate_loc_side","plate_loc_height"]].std()
    r=np.sqrt((sd**2).sum(axis=1))
    ok=r.notna()
    m=float(np.average(r[ok],weights=n.reindex(r[ok].index)))
    v=((r*n+a*m)/(n+a)).fillna(m)
    return v.index.astype(str).to_numpy(), v.to_numpy(float), m, float(v.std())


def locsd_fit(dl, Hbuild, seasons, rh=2, lam=20.0):
    """★ 산포 → 다음시즌 등급 회귀. EB 를 통제한 **증분** 계수(등급점수/1z)를 학습부에서만 적합."""
    idx,val,m,sdv=locsd_of(Hbuild)
    S=pd.Series(val,index=pd.Index(idx))
    zs,rs=[],[]
    for s_ in seasons:
        H=dl[(dl.season<=s_-1)&(dl.season>=s_-rh)]
        if not len(H): continue
        mu=float(H.pt.mean()); g=H.groupby("pitcher_id").pt.agg(["sum","size"])
        ap=A_P+BETA*g["size"]; eb=(g["sum"]+ap*mu)/(g["size"]+ap)
        i2,v2,m2,_=locsd_of(H); S2=pd.Series(v2,index=pd.Index(i2))
        P=dl[dl.season==s_]
        ag=P.groupby("pitcher_id").agg(n=("pt","size"),y=("pt","mean"))
        ag=ag[ag.n>=50].copy()
        if len(ag)<30: continue
        ag["eb"]=ag.index.map(eb).fillna(mu); ag["sp"]=ag.index.map(S2).fillna(m2)
        y=ag.y.to_numpy(float); e=ag.eb.to_numpy(float)
        sl=np.cov(y,e)[0,1]/max(np.var(e),1e-12)
        rs.append(y-y.mean()-(e-e.mean())*sl)
        zz=(ag.sp.to_numpy(float)-m2)/max(float(ag.sp.std()),1e-9)
        zs.append(zz-zz.mean())
    z=np.concatenate(zs); r=np.concatenate(rs)
    b=float(-(z@r)/(z@z+lam))     # 부호 반전해 저장 → 적용부에서 t -= b*z
    return idx,val,m,sdv,b


def pcell_of(H, pri):
    """★ 투수별 **과거 상황수준 평균** = 그 투수 과거 행들의 셀 리그평균 등급의 평균.
    지표가 평균하는 것은 2025 상황 구성이므로, 과거 구성을 빼고 평가행 구성을 더해야 한다."""
    lv=(pri[cells(H)]@G)
    v=pd.Series(lv,index=H.pitcher_id.to_numpy()).groupby(level=0).mean()
    return v.index.astype(str).to_numpy(), v.to_numpy(float), float(lv.mean())


def bdev_of(H, a=400.0):
    """★ 타자별 편차(수축) — 그 타자를 상대할 때 리그 평균 대비 제구점수 차이."""
    mu=float(H.pt.mean()); g=H.groupby("batter_id").pt.agg(["sum","size"])
    v=(g["sum"]+a*mu)/(g["size"]+a)-mu
    return v.index.astype(str).to_numpy(), v.to_numpy(float)


def trans_of(H):
    """★ 투수별 전환 상태 — 가장 최근 시즌의 소속팀, 직전시즌 F비중, 전체 F비중.
    평가행의 pitcher_team_id 와 비교해 '팀이 바뀌었나' 를 **자기 행에서만** 판정한다(행 독립)."""
    yr=int(H.season.max())
    L=H[H.season==yr]
    lt=L.groupby("pitcher_id").pitcher_team_id.agg(lambda x: x.astype(str).mode().iat[0] if len(x) else "")
    f1=L.groupby("pitcher_id").isF.mean()
    fa=H.groupby("pitcher_id").isF.mean()
    ix=fa.index
    return (ix.astype(str).to_numpy(),
            lt.reindex(ix).fillna("").astype(str).to_numpy(),
            f1.reindex(ix).fillna(fa.reindex(ix)).to_numpy(float),
            fa.to_numpy(float))


def rg_of(H, ag=2000.0, ar=1200.0):
    """★ 곱 분해 표 — 투수별 (기하전용 등급점수 EB, 반대투구율 EB) + 리그 F−R 반대투구율차.
    반대투구율은 해간 지속성이 높고(23→24 τ +0.402) 레짐에 민감하므로 성분을 분리한다."""
    Hn=H[H.rev==0]
    g=Hn.groupby("pitcher_id").pt.agg(["sum","size"]); gmu=float(Hn.pt.mean())
    gv=(g["sum"]+ag*gmu)/(g["size"]+ag)
    r=H.groupby("pitcher_id").rev.agg(["sum","size"]); rmu=float(H.rev.mean())
    rv=(r["sum"]+ar*rmu)/(r["size"]+ar)
    ix=r.index
    gv=gv.reindex(ix).fillna(gmu)
    yr=int(H.season.max()); L=H[H.season==yr]
    rd=float(L[L.isF==1].rev.mean()-L[L.isF==0].rev.mean()) if (L.isF==1).any() else 0.0
    return ix.astype(str).to_numpy(), gv.to_numpy(float), rv.to_numpy(float), gmu, rmu, rd


def npri_fit(dl, Tn, seasons, rh=0, lam=20.0):
    """★ 이력 크기 → 다음시즌 등급 회귀. EB 를 통제한 **증분** 계수(등급점수/1z).
    표준화 상수는 배포에 쓰는 이력표(Tn=pit_n)에서 뽑아 학습/추론을 일치시킨다."""
    lz=np.log1p(Tn.to_numpy(float)); mu=float(lz.mean()); sd=max(float(lz.std()),1e-9)
    zs,rs=[],[]
    for s_ in seasons:
        H=dl[dl.season<=s_-1]
        if not len(H): continue
        mu0=float(H.pt.mean()); g=H.groupby("pitcher_id").pt.agg(["sum","size"])
        ap=A_T_EFF(g["size"]); eb=(g["sum"]+ap*mu0)/(g["size"]+ap)
        P=dl[dl.season==s_]
        ag=P.groupby("pitcher_id").agg(n=("pt","size"),y=("pt","mean"))
        ag=ag[ag.n>=50].copy()
        if len(ag)<30: continue
        ag["eb"]=ag.index.map(eb).fillna(mu0)
        ag["hn"]=ag.index.map(g["size"]).fillna(0.0)
        y=ag.y.to_numpy(float); e=ag.eb.to_numpy(float)
        sl=np.cov(y,e)[0,1]/max(np.var(e),1e-12)
        rs.append(y-y.mean()-(e-e.mean())*sl)
        z=(np.log1p(ag.hn.to_numpy(float))-mu)/sd
        zs.append(z-z.mean())
    z=np.concatenate(zs); r=np.concatenate(rs)
    return float((z@r)/(z@z+lam)), mu, sd


def A_T_EFF(n):
    return float(os.environ.get("A_T","4000"))+float(os.environ.get("BETA_T","0.6"))*n


def rgv_of(dall, dlab, ag):
    """★ 성분별 시즌 범위 분리.
    r = 반대투구율 EB (최근 RGV_H 시즌 · 전체 행: 반대투구는 미라벨 행에도 기록돼 있다)
    g = 정투구 등급점수 EB (전 시즌 · **시즌중심화**)
    """
    yr=int(dall.season.max())
    Rw=dall[dall.season>=yr-RGV_H+1]
    mur=float(Rw.rev.mean()); gr=Rw.groupby("pitcher_id").rev.agg(["sum","size"])
    r=(gr["sum"]+RGV_AR*mur)/(gr["size"]+RGV_AR)
    Hn=dlab[dlab.rev==0]
    _v=Hn.pt.to_numpy(float)
    _m=pd.Series(_v).groupby(Hn.season.to_numpy()).transform("mean").to_numpy()
    ptc=_v-_m+float(_v.mean())
    G2=pd.DataFrame({"p":Hn.pitcher_id.to_numpy(),"v":ptc})
    gg=G2.groupby("p").v.agg(["sum","size"]); mug=float(ptc.mean())
    g=(gg["sum"]+ag*mug)/(gg["size"]+ag)
    ix=r.index.union(g.index)
    r=r.reindex(ix).fillna(mur); g=g.reindex(ix).fillna(mug)
    return ix.astype(str).to_numpy(), r.to_numpy(float), g.to_numpy(float), mur, mug


COMP_PHY=["rel_speed","rel_height","spin_rate","extension","induced_vert_break"]
def _comp_covs(Hall, kinds):
    g=Hall.groupby("pitcher_id"); X=pd.DataFrame({"n_all":g.size()})
    pk=pd.to_numeric(Hall.PITCH_KIND,errors="coerce")
    for k in kinds: X[f"pk_{int(k)}"]=(pk==k).groupby(Hall.pitcher_id).mean()
    for c in COMP_PHY: X[c]=g[c].mean()
    X["early"]=(Hall.inning<=3).groupby(Hall.pitcher_id).mean(); X["hand"]=g.pitcher_hand.mean(); X["hfs"]=g.isF.mean()
    return X


def comp_table(d, ymax, lam=200.0):
    """★ comparable prior 표. 계수는 평가시즌 yp<=ymax 의 패널(이력 < yp → yp 의 **R 행** 평균)에서만 적합하고,
    표는 season<=ymax 이력으로 만든다. m_p 는 과거 F 비중 구간 안에서 평균 0 으로 중화한다."""
    pk=pd.to_numeric(d.PITCH_KIND,errors="coerce"); kinds=pk.value_counts().index[:7].tolist()
    feats=[f"pk_{int(k)}" for k in kinds]+COMP_PHY+["early","hand"]
    Zs,ys,ws=[],[],[]
    for yp in range(int(d.season.min())+1,ymax+1):
        S=d[d.lab&(d.season==yp)&(d.isF==0)]
        Y=S.groupby("pitcher_id").pt.agg(["size","mean"]); Y=Y[Y["size"]>=50]
        X=_comp_covs(d[d.season<=yp-1],kinds)
        Pp=Y.join(X,how="inner"); Pp=Pp[Pp.n_all>=30]
        if len(Pp)<30: continue
        nl=d[d.lab&(d.season<=yp-1)].groupby("pitcher_id").size().reindex(Pp.index).fillna(0.).to_numpy()
        Zs.append(Pp[feats]); ys.append((Pp["mean"]-Pp["mean"].mean()).to_numpy()); ws.append(np.minimum(nl+50,2000))
    Z=pd.concat(Zs); yy=np.concatenate(ys); ww=np.concatenate(ws); med=Z.median(); sdv=Z.std().replace(0,1.0)
    Zn=((Z.fillna(med)-med)/sdv).to_numpy()
    beta=np.linalg.solve(Zn.T@(Zn*ww[:,None])+lam*np.eye(Zn.shape[1]),Zn.T@(yy*ww))
    X=_comp_covs(d[d.season<=ymax],kinds)
    mx=((X[feats].fillna(med)-med)/sdv).to_numpy()@beta
    hfs=X.hfs.fillna(0.).to_numpy(); bk=np.where(hfs<=0.02,0,np.where(hfs<0.5,1,2))
    for b_ in (0,1,2):
        m_=bk==b_
        if m_.sum()>3: mx[m_]-=mx[m_].mean()
    print(f"  ★comparable prior: 패널 {len(Zs)}개·{len(yy)}투수시즌 · 표 {len(X)}명 · m sd {mx.std():.3f} · 계수 "+" ".join(f"{f}={b:+.2f}" for f,b in zip(feats,beta)),flush=True)
    return X.index.astype(str).to_numpy(), mx.astype(float)


def lvl_diff_of(P):
    """리그 R/F 평균차 — **최근 1시즌**에서만. 2023 에 부호가 뒤집혔다."""
    yr=int(P.season.max()); Q=P[P.season==yr]
    R=Q[Q.isF==0].pt.mean(); F=Q[Q.isF==1].pt.mean() if (Q.isF==1).any() else R
    return float(F-R)


def mag_tables(P, T):
    """빗나간 크기 신호의 셀 prior + 학습 투수 분포의 표준화 상수(평가 분포를 안 본다)"""
    from script import MAG_A
    g0=float(P["mag"].mean())
    pz=P.groupby(["pitcher_id","season"]).agg(nn=("mag","size"),m=("mag","mean"))
    pz=pz[pz.nn>=50]
    ix=pd.MultiIndex.from_arrays([P.pitcher_id,P.season])
    sel=pd.Series(np.arange(len(pz)),index=pz.index).reindex(ix).to_numpy()
    ok=~np.isnan(sel); K=cells(P)[ok]
    c=np.bincount(K,minlength=NCELL).astype(float)
    s=np.bincount(K,weights=pz.m.to_numpy()[sel[ok].astype(int)],minlength=NCELL)
    T["mag_prior"]=(s+200.*g0)/(c+200.)
    # 학습 투수들의 EB 신호 분포 → mu/sd 고정
    n=P.groupby("pitcher_id").size(); h=P.groupby("pitcher_id")["mag"].sum()
    cellm=float(T["mag_prior"].mean())
    e=((h+MAG_A*cellm)/(n+MAG_A))[n>=50]
    T["mag_mu"]=float(e.mean()); T["mag_sd"]=float(e.std())
    return T


def prior_table(P, gm):
    if RECENT_N>0:
        yr=int(P.season.max()); P=P[P.season>=yr-RECENT_N+1]
    oh=np.eye(3)[P.y.to_numpy().astype(int)]
    ix=pd.MultiIndex.from_arrays([P.pitcher_id,P.season])
    ps=pd.DataFrame(oh,index=ix).groupby(level=[0,1]).agg(["mean","size"])
    keep=ps.loc[:,(0,"size")].to_numpy()>=50
    R=ps.loc[:,(slice(None),"mean")].to_numpy(dtype=float)[keep]
    sel=pd.Series(np.arange(int(keep.sum())),index=ps.index[keep]).reindex(ix).to_numpy()
    ok=~np.isnan(sel); K=cells(P)[ok]; RR=R[sel[ok].astype(int)]
    c=np.bincount(K,minlength=NCELL).astype(float)
    s=np.column_stack([np.bincount(K,weights=RR[:,j],minlength=NCELL) for j in range(3)])
    t=(s+AC_P*gm)/(c[:,None]+AC_P); t=t/t.sum(1,keepdims=True)
    if GLOBAL_PRIOR: t=np.tile(gm,(NCELL,1))          # ★ 대조: 전역평균
    return t

def loso(P):
    parts=[]
    for s in sorted(P.season.unique()):
        Po=P[P.season!=s]; Pi=P[P.season==s]
        if len(Po)<50000: continue
        gmo=np.array([(Po.y==j).mean() for j in range(3)])
        Poh = Po[Po.season>=Po.season.max()-RECENT_HIST+1] if RECENT_HIST>0 else Po
        To=build_tables(Poh,gmo); To["prior"]=prior_table(Po,gmo); To=mag_tables(Poh,To)
        parts.append(feats(Pi,To,To["prior"][cells(Pi)]))
    X=pd.concat(parts); return X, P.loc[X.index]

def sc(y,p,pit,tag):
    Y=np.eye(3)[y]; pitch=100*(1-((p-Y)**2).sum(1).mean()/2)
    g=pd.DataFrame({"p":pit,"s":G[y],"h":p@G}).groupby("p").agg(n=("s","size"),s=("s","mean"),h=("h","mean"))
    g=g[g.n>=50]; a=g.s.round(10); h=g.h.round(10)
    t=0. if (len(g)<2 or a.nunique()<2 or h.nunique()<2) else kendalltau(a.to_numpy(),h.to_numpy(),variant="b").statistic
    print(f"   {tag:32s} Score {.7*pitch+.3*50*(1+t):7.3f} | Pitch {pitch:7.4f} | Player {50*(1+t):7.3f} (τ {t:+.4f})",flush=True)
    return .7*pitch+.3*50*(1+t)

PAR=dict(objective="multiclass",num_class=3,learning_rate=.05,num_leaves=63,min_data_in_leaf=300,
         feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=5.,num_threads=NTH,
         verbose=-1,metric="multi_logloss")

def tgt_of(dfrow, gtab, prior_rows):
    loc=gtab.index.get_indexer(pd.Index(dfrow.pitcher_id.to_numpy())); have=loc>=0
    safe=np.where(have,loc,0)
    n=np.where(have,gtab["size"].to_numpy()[safe],0.)
    h=np.where(have,gtab["sum"].to_numpy()[safe],0.)
    pg=prior_rows@G; ae=A_T+BETA_T*n
    return (h+ae*pg)/(n+ae)

def main():
    d=load_all(); print(f"전체 {len(d):,} · 라벨 {int(d.lab.sum()):,}",flush=True)
    dl=d[d.lab].copy()
    A=dl[dl.season<=HOLD_Y-1].copy(); B=dl[dl.season==HOLD_Y].copy()
    print(f"  ★홀드아웃 {HOLD_Y} · 학습 {sorted(A.season.unique())} · 평가행 {len(B):,}",flush=True)
    Aall=d[d.season<=HOLD_Y-1]
    gmA=np.array([(A.y==j).mean() for j in range(3)]); priA=prior_table(A,gmA)
    _pw=lambda x: x if PGR_HIST<=0 else x[x.season>=int(x.season.max())-PGR_HIST+1]
    gtabA,r2A=proxy_pitcher(_pw(Aall))
    print(f"  ★대리등급 표: 시즌 {sorted(_pw(Aall).season.unique())} · PGR_HIST={PGR_HIST} CTR={int(PGR_CTR)} DECAY={PGR_DECAY}",flush=True); print(f"대리표 R² {r2A:.4f} · 투수 {len(gtabA)} · 행/라벨행 {gtabA['size'].sum()/len(A):.3f}",flush=True)
    Ah = A[A.season>=A.season.max()-RECENT_HIST+1] if RECENT_HIST>0 else A
    envR=envF=0.0
    if ENV_ADJ:
        Ah, envR, envF = env_center(Ah)
        print(f"  환경 제거: 목표 환경 R {envR:.3f} · F {envF:.3f} (차이 {envF-envR:+.3f})",flush=True)
    XA,A2=loso(A); TA=build_tables(Ah,gmA); TA["prior"]=priA; TA=mag_tables(Ah,TA)
    TA["env_R"]=envR; TA["env_F"]=envF
    if SEA_ADJ:
        TA=sea_tables(Ah,TA,sea_only=os.environ.get("SEA_ONLY","0")=="1")
        print(f"  (시즌×레벨) 중심화 · 상수 {TA['sea_c']:.3f} · prior 범위 {TA['sea_prior'].min():+.3f}~{TA['sea_prior'].max():+.3f}",flush=True)
    if RECENT_HIST>0: print(f"  투수 이력 표 = 최근 {RECENT_HIST}시즌 ({sorted(Ah.season.unique())}) · {len(Ah):,}행",flush=True)
    TA["lvl_diff"]=lvl_diff_of(A)
    if HAND_G:
        TA["hand_dev"]=hand_dev_of(A)
        print(f"  플래툰 편차 (좌우조합) {np.round(TA['hand_dev'],3)}",flush=True)
    if EXP_G:
        ei,ev,em,es=exp_of(A); TA["exp_idx"]=ei; TA["exp_val"]=ev; TA["exp_mu"]=em; TA["exp_sd"]=es
        print(f"  경력 {len(ei)}명 · 평균 {em:.2f} sd {es:.2f} · 범위 {ev.min():.0f}~{ev.max():.0f}",flush=True)
    if USE_G:
        _ss=[x for x in sorted(A.season.unique()) if x>=int(A.season.min())+2]
        ub,um,us=usage_fit(A,_ss,A); TA["use_b"]=ub; TA["use_mu"]=um; TA["use_sd"]=us
        print(f"  기용 계수 적합 시즌 {_ss} · "+"  ".join(f"{k}={v:+.4f}" for k,v in zip(USE_KEYS,ub)),flush=True)
    if LVLN_G and not NPRI_G:
        _lz=np.log1p(TA["pit_n"].to_numpy(float)); TA["npri_mu"]=float(_lz.mean()); TA["npri_sd"]=max(float(_lz.std()),1e-9)
    if NPRI_G:
        _ss=[x for x in sorted(A.season.unique()) if x>=int(A.season.min())+1]
        nb,nm,ns=npri_fit(A,TA["pit_n"],_ss); TA["npri_b"]=nb; TA["npri_mu"]=nm; TA["npri_sd"]=ns
        print(f"  ★이력크기 사전: 계수 b={nb:+.4f} 등급점수/1z · log1p(n) 평균 {nm:.3f} sd {ns:.3f} (적합 {_ss})",flush=True)
    if RGV:
        vi,vr,vg,vrm,vgm=rgv_of(d[d.season<=HOLD_Y-1],A,float(os.environ.get("A_T","1800")))
        TA["rgv_idx"]=vi; TA["rgv_r"]=vr; TA["rgv_g"]=vg; TA["rgv_rmu"]=vrm; TA["rgv_gmu"]=vgm
        print(f"  ★성분분리: 투수 {len(vi)} · r 창 최근{RGV_H}시즌 α_r={RGV_AR} 평균 {vrm:.4f} sd {vr.std():.4f} · g 전시즌+중심화 평균 {vgm:.3f} sd {vg.std():.3f}",flush=True)
    if RGSPLIT:
        ri,gv,rv,gmu,rmu,rd=rg_of(Ah)
        TA["rg_idx"]=ri; TA["rg_g_val"]=gv; TA["rg_r_val"]=rv
        TA["rg_gmu"]=gmu; TA["rg_rmu"]=rmu; TA["rg_rdiff"]=rd
        print(f"  ★곱분해: 투수 {len(ri)} · 기하점수 평균 {gmu:.3f} sd {gv.std():.3f} · 반대투구율 평균 {rmu:.4f} sd {rv.std():.4f} · F−R 차 {rd:+.4f}",flush=True)
    if TRG_G or TRF_G:
        ti,lv,f1,fa=trans_of(A); TA["lastteam_idx"]=ti; TA["lastteam_val"]=lv
        TA["fs1_val"]=f1; TA["fsall_val"]=fa
        print(f"  ★전환표: 투수 {len(ti)} · 최근시즌 팀기록 {int((lv!='').sum())} · F비중이동 평균 {(f1-fa).mean():+.4f} sd {(f1-fa).std():.4f}",flush=True)
    if CELL_G:
        ci,cv,cm=pcell_of(Ah,priA); TA["pcell_idx"]=ci; TA["pcell_val"]=cv; TA["pcell_mu"]=cm
        print(f"  ★상황구성: 투수 {len(ci)} · 과거 상황수준 평균 {cm:.3f} · 투수간 sd {cv.std():.3f} · 범위 {cv.min():.2f}~{cv.max():.2f}",flush=True)
    if BAT_G:
        bi,bv=bdev_of(Ah); TA["bdev_idx"]=bi; TA["bdev_val"]=bv
        print(f"  ★타자 편차: 타자 {len(bi)} · sd {bv.std():.3f} · 범위 {bv.min():+.2f}~{bv.max():+.2f}",flush=True)
    if LOCSD_G:
        _ss=[x for x in sorted(A.season.unique()) if x>=int(A.season.min())+2]
        li,lv,lm,lsd,lb=locsd_fit(A,Ah,_ss); TA["locsd_idx"]=li; TA["locsd_val"]=lv
        TA["locsd_mu"]=lm; TA["locsd_sd"]=lsd; TA["locsd_b"]=lb
        print(f"  ★로케이션 산포: 투수 {len(li)} · 평균 {lm:.4f} sd {lsd:.4f} · 계수 b={lb:+.4f} 등급점수/1z (적합 {_ss})",flush=True)
    if ENVAX: TA=envax_tables(A,TA)
    if LVL_TEAM_G or HTM_G:
        ti,td,tmu=team_dev_of(A); TA["team_idx"]=ti; TA["team_dev"]=td
        print(f"  팀 환경 편차 {len(ti)}팀 · 범위 {td.min():+.3f}~{td.max():+.3f} (리그 {tmu:.3f})",flush=True)
    XB=feats(B,TA,priA[cells(B)],list(XA.columns))
    yA=A2.y.to_numpy().astype(int); yB=B.y.to_numpy().astype(int); pidB=B.pitcher_id.to_numpy()
    va=(A2.season.to_numpy()==A2.season.max())
    bst=lgb.train(PAR,lgb.Dataset(XA[~va],label=yA[~va]),4000,
                  valid_sets=[lgb.Dataset(XA[va],label=yA[va])],
                  callbacks=[lgb.early_stopping(80,verbose=False)])
    nb=bst.best_iteration; print(f"★ 최적 트리 {nb}",flush=True)
    BSTS=[bst]
    for _k in range(1,NSEED):
        _p=dict(PAR); _p.update(seed=_k*101,bagging_seed=_k*101+1,feature_fraction_seed=_k*101+2,data_random_seed=_k*101+3)
        BSTS.append(lgb.train(_p,lgb.Dataset(XA[~va],label=yA[~va]),nb))
    if NSEED>1: print(f"  ★q 시드 앙상블 {NSEED}개 (트리 {nb})",flush=True)
    P3=np.zeros((len(B),3))
    for _b in BSTS: P3+=np.asarray(_b.predict(XB.to_numpy(float),num_iteration=nb)).reshape(len(B),3)
    P3/=float(len(BSTS))
    EB=XB[[f"pit_c{j}" for j in range(3)]].to_numpy(); EB=EB/EB.sum(1,keepdims=True)
    EBG=XB[[f"pit_c{j}_g" for j in range(3)]].to_numpy(); EBG=EBG/EBG.sum(1,keepdims=True)
    _NPRI=None
    if NPRI_G:
        _pn=B.pitcher_id.map(TA["pit_n"]).fillna(0.).to_numpy(float)
        _NPRI=NPRI_G*TA["npri_b"]*(np.log1p(_pn)-TA["npri_mu"])/max(TA["npri_sd"],1e-9)
    if RGV:
        _l=pd.Index(TA["rgv_idx"]).get_indexer(pd.Index(B.pitcher_id.to_numpy()))
        _ok=_l>=0; _sf=np.where(_ok,_l,0)
        _r=np.where(_ok,TA["rgv_r"][_sf],TA["rgv_rmu"])
        _g=np.where(_ok,TA["rgv_g"][_sf],TA["rgv_gmu"])
        tB=(1.0-_r)*_g
        print(f"  ★성분분리 목표 · t 평균 {tB.mean():.3f} sd {tB.std():.3f} · 적용 {_ok.mean()*100:.1f}%",flush=True)
    elif RGSPLIT:
        _l=pd.Index(TA["rg_idx"]).get_indexer(pd.Index(B.pitcher_id.to_numpy()))
        _ok=_l>=0; _sf=np.where(_ok,_l,0)
        _g=np.where(_ok,TA["rg_g_val"][_sf],TA["rg_gmu"])
        _r=np.where(_ok,TA["rg_r_val"][_sf],TA["rg_rmu"])
        if RG_G:
            _isF=(B.game_type.to_numpy()=="F").astype(float)
            _r=np.clip(_r+RG_G*TA["rg_rdiff"]*_isF,0.,1.)
        tB=(1.0-_r)*_g
        print(f"  ★곱분해 목표 RG_G={RG_G} · t 평균 {tB.mean():.3f} sd {tB.std():.3f} · 적용 {_ok.mean()*100:.1f}%",flush=True)
    else:
        tB = tgt_of(B,gtabA,priA[cells(B)]) if USE_PROXY else (EB@G)
    if COMP_G:
        ci_,cv_=comp_table(d,HOLD_Y-1); TA["comp_idx"]=ci_; TA["comp_val"]=cv_
        _l=pd.Index(ci_).get_indexer(pd.Index(B.pitcher_id.to_numpy()))
        _m=np.where(_l>=0,cv_[np.where(_l>=0,_l,0)],0.)
        if USE_PROXY:
            _l2=pd.Index(gtabA.index.astype(str)).get_indexer(pd.Index(B.pitcher_id.to_numpy()))
            _n=np.where(_l2>=0,gtabA["size"].to_numpy(float)[np.where(_l2>=0,_l2,0)],0.)
            _ae=float(os.environ.get("A_T","4000"))+float(os.environ.get("BETA_T","0.6"))*_n
        else:
            _n=B.pitcher_id.map(TA["pit_n"]).fillna(0.).to_numpy(float); _ae=A_P+BETA*_n
        _tc=COMP_G*(_ae/(_n+_ae))*_m
        tB = tB + _tc
        print(f"  ★comparable 보정 g={COMP_G} · 항 sd {_tc.std():.4f} · 범위 {_tc.min():+.3f}~{_tc.max():+.3f} · 표 적중 {(_l>=0).mean()*100:.1f}%",flush=True)
    TA["tstar"]=str(d[(d.season<=HOLD_Y-1)&(d.game_type=="F")].pitcher_team_id.value_counts().index[0])
    if TINV_G or TINV_P:
        _isR=(B.game_type.to_numpy()!="F").astype(float)
        _pT=(B.pitcher_team_id.astype(str).to_numpy()==TA["tstar"]); _bT=(B.batter_team_id.astype(str).to_numpy()==TA["tstar"])
        _tt=TINV_G*_isR*(_pT|_bT).astype(float)+TINV_P*_isR*_pT.astype(float)
        tB = tB + _tt
        print(f"  ★T*관여 보정 g={TINV_G} p={TINV_P} · T*={TA['tstar'][:14]} · R행 중 관여 {((_pT|_bT)&(_isR>0)).mean()*100:.1f}% · T*투수행 {(_pT&(_isR>0)).mean()*100:.1f}%",flush=True)
    if FRG_G:
        _fm=float(TA["fshare"].mean())
        _hf0=B.pitcher_id.map(TA["fshare"]).fillna(_fm).to_numpy(float)
        _tg=FRG_G*(_hf0-_fm)
        tB = tB + _tg
        print(f"  ★주변부 보정 g={FRG_G} · 과거F비중 평균 {_fm:.4f} · 항 평균 {_tg.mean():+.4f} sd {_tg.std():.4f} · 범위 {_tg.min():+.3f}~{_tg.max():+.3f}",flush=True)
    if LVLF_G:
        _hfm=float(TA["fshare"].mean())
        _hf=B.pitcher_id.map(TA["fshare"]).fillna(_hfm).to_numpy(float)
        _isF2=(B.game_type.to_numpy()=="F").astype(float)
        _tf=LVLF_G*TA["lvl_diff"]*_isF2*(_hf-_hfm)
        tB = tB + _tf
        print(f"  ★F×과거F비중 g={LVLF_G} · hf 평균 {_hfm:.4f} sd {_hf.std():.4f} · 항 평균 {_tf.mean():+.4f} sd {_tf.std():.4f} · 범위 {_tf.min():+.3f}~{_tf.max():+.3f}",flush=True)
    if LVLN_G:
        _pn=B.pitcher_id.map(TA["pit_n"]).fillna(0.).to_numpy(float)
        _z=(np.log1p(_pn)-TA["npri_mu"])/max(TA["npri_sd"],1e-9)
        _isF=(B.game_type.to_numpy()=="F").astype(float)
        _tm=LVLN_G*TA["lvl_diff"]*_isF*_z
        tB = tB + _tm
        print(f"  ★F×이력크기 g={LVLN_G} · 항 평균 {_tm.mean():+.4f} sd {_tm.std():.4f} · 범위 {_tm.min():+.3f}~{_tm.max():+.3f}",flush=True)
    if _NPRI is not None:
        tB = tB + _NPRI
        print(f"  ★이력크기 보정 g={NPRI_G} · 항 평균 {_NPRI.mean():+.4f} sd {_NPRI.std():.4f} · 범위 {_NPRI.min():+.3f}~{_NPRI.max():+.3f}",flush=True)
    if ENV_ADJ:
        _isF=(B.game_type.to_numpy()=="F").astype(float)
        tB = tB + (1.0-_isF)*TA["env_R"] + _isF*TA["env_F"]
    if SEA_ADJ:
        _loc=pd.Index(TA["sea_idx"]).get_indexer(pd.Index(B.pitcher_id.to_numpy())); _hv=_loc>=0
        _sf=np.where(_hv,_loc,0)
        _n=np.where(_hv,TA["sea_n"][_sf],0.); _h=np.where(_hv,TA["sea_sum"][_sf],0.)
        _ts=(_h+SEA_A*TA["sea_prior"][cells(B)])/(_n+SEA_A)+TA["sea_c"]
        tB = np.where(_n<=SEA_NMAX,_ts,tB) if SEA_NMAX>0 else _ts
        print(f"  SEA_ADJ 적용 α={SEA_A} nmax={SEA_NMAX}",flush=True)
    from script import MAG_A, MAG_G, LVL_G
    if LVL_G:
        _isF=(B.game_type.to_numpy()=="F").astype(float)
        if LVL_MODE in ("hist","both"):
            _hf=B.pitcher_id.map(TA["fshare"]).fillna(0.).to_numpy(float)
            _w=_hf if LVL_MODE=="hist" else 0.5*(_isF+_hf)
        else: _w=_isF
        tB = tB + LVL_G*TA["lvl_diff"]*_w
        print(f"  레벨 보정 γ={LVL_G} mode={LVL_MODE} · 평균차 {TA['lvl_diff']:+.3f} · 가중평균 {_w.mean():.4f}",flush=True)
    if HAND_G:
        _ph=pd.to_numeric(B.pitcher_hand,errors="coerce").fillna(1).clip(1,2).astype(int).to_numpy()
        _bh=pd.to_numeric(B.batter_hand,errors="coerce").fillna(1).clip(1,2).astype(int).to_numpy()
        tB = tB + HAND_G*TA["hand_dev"][(_ph-1)*2+(_bh-1)]
        print(f"  플래툰 보정 γ={HAND_G}",flush=True)
    if EXP_G:
        _l=pd.Index(TA["exp_idx"]).get_indexer(pd.Index(B.pitcher_id.to_numpy()))
        _v=np.where(_l>=0,TA["exp_val"][np.where(_l>=0,_l,0)],TA["exp_mu"])
        tB = tB + EXP_G*(_v-TA["exp_mu"])/max(TA["exp_sd"],1e-9)
        print(f"  경력 보정 γ={EXP_G} · 적용 {(_l>=0).mean()*100:.1f}%",flush=True)
    if LVL_TEAM_G:
        _l=pd.Index(TA["team_idx"]).get_indexer(pd.Index(B.pitcher_team_id.astype(str).to_numpy()))
        _d=np.where(_l>=0,TA["team_dev"][np.where(_l>=0,_l,0)],0.)
        tB = tB + LVL_TEAM_G*_d
        print(f"  팀 보정 γ={LVL_TEAM_G} · 적용 {(_l>=0).mean()*100:.1f}%",flush=True)
    if TRG_G or TRF_G:
        _isF=(B.game_type.to_numpy()=="F").astype(float); _ld=TA["lvl_diff"]
        _l=pd.Index(TA["lastteam_idx"]).get_indexer(pd.Index(B.pitcher_id.to_numpy()))
        _ok=_l>=0; _sf=np.where(_ok,_l,0)
        if TRG_G:
            _lt=np.where(_ok,TA["lastteam_val"][_sf],"")
            _chg=(_ok&(_lt!=B.pitcher_team_id.astype(str).to_numpy())).astype(float)
            tB = tB + TRG_G*_ld*_isF*_chg
            print(f"  ★전환게이트(팀) g={TRG_G} · 팀변경 {_chg.mean()*100:.1f}% · F&변경 {(_chg*_isF).mean()*100:.2f}% · 항sd {(TRG_G*_ld*_isF*_chg).std():.4f}",flush=True)
        if TRF_G:
            _sh=np.where(_ok,TA["fs1_val"][_sf]-TA["fsall_val"][_sf],0.)
            tB = tB + TRF_G*_ld*_isF*_sh
            print(f"  ★전환게이트(역할) g={TRF_G} · 항sd {(TRF_G*_ld*_isF*_sh).std():.4f}",flush=True)
    if CELL_G:
        _lv=(priA[cells(B)]@G)
        _l=pd.Index(TA["pcell_idx"]).get_indexer(pd.Index(B.pitcher_id.to_numpy()))
        _pa=np.where(_l>=0,TA["pcell_val"][np.where(_l>=0,_l,0)],TA["pcell_mu"])
        tB = tB + CELL_G*(_lv-_pa)
        print(f"  ★상황구성 보정 g={CELL_G} · 항 평균 {(CELL_G*(_lv-_pa)).mean():+.4f} sd {(CELL_G*(_lv-_pa)).std():.4f} · 적용 {(_l>=0).mean()*100:.1f}%",flush=True)
    if BAT_G:
        _l=pd.Index(TA["bdev_idx"]).get_indexer(pd.Index(B.batter_id.to_numpy()))
        _bd=np.where(_l>=0,TA["bdev_val"][np.where(_l>=0,_l,0)],0.)
        tB = tB + BAT_G*_bd
        print(f"  ★타자 보정 g={BAT_G} · 항sd {(BAT_G*_bd).std():.4f} · 적용 {(_l>=0).mean()*100:.1f}%",flush=True)
    if LOCSD_G:
        _l=pd.Index(TA["locsd_idx"]).get_indexer(pd.Index(B.pitcher_id.to_numpy()))
        _v=np.where(_l>=0,TA["locsd_val"][np.where(_l>=0,_l,0)],TA["locsd_mu"])
        _z=(_v-TA["locsd_mu"])/max(TA["locsd_sd"],1e-9)
        tB = tB - LOCSD_G*TA["locsd_b"]*_z
        print(f"  ★산포 보정 g={LOCSD_G} b={TA['locsd_b']:+.4f} · 적용 {(_l>=0).mean()*100:.1f}% · 항sd {(LOCSD_G*TA['locsd_b']*_z).std():.4f}",flush=True)
    for _ax,_gv in envax_spec():
        if not _gv: continue
        _l=pd.Index(TA["envax_i__"+_ax]).get_indexer(pd.Index(envax_key(B,_ax)))
        _dv=np.where(_l>=0,TA["envax_v__"+_ax][np.where(_l>=0,_l,0)],0.)
        tB = tB + _gv*_dv
        print(f"  ★환경축 {_ax} g={_gv:+.2f} · 적용 {(_l>=0).mean()*100:.1f}% · 항sd {(_gv*_dv).std():.4f}",flush=True)
    if HTM_G:
        _l=pd.Index(TA["team_idx"]).get_indexer(pd.Index(B.pitcher_team_id.astype(str).to_numpy()))
        _d=np.where(_l>=0,TA["team_dev"][np.where(_l>=0,_l,0)],0.)
        _pn=B.pitcher_id.map(TA["pit_n"]).fillna(0.).to_numpy(float)
        _ap=A_P+BETA*_pn; _w=_ap/(_pn+_ap)
        tB = tB + HTM_G*_w*_d
        _lo=_w[_pn<100].mean() if (_pn<100).any() else 0.0
        _hi=_w[_pn>5000].mean() if (_pn>5000).any() else 0.0
        print(f"  ★계층EB g={HTM_G} · 가중평균 {_w.mean():.4f} (n<100 {_lo:.4f} · n>5000 {_hi:.4f}) · 팀매칭 {(_l>=0).mean()*100:.1f}%",flush=True)
    if USE_G:
        _ut=np.zeros(len(B))
        for _k,_b,_m,_sd in zip(USE_KEYS,TA["use_b"],TA["use_mu"],TA["use_sd"]):
            _ut=_ut+_b*(pd.to_numeric(B[_k],errors="coerce").fillna(_m).to_numpy(float)-_m)/max(_sd,1e-9)
        if USE_W:
            _pnu=B.pitcher_id.map(TA["pit_n"]).fillna(0.).to_numpy(float); _aT=float(os.environ.get("USE_AW","0")) or float(os.environ.get("A_T","4000"))
            _ut=_ut*(_aT/(_pnu+_aT))
        if USE_RONLY: _ut=_ut*(1.0-B.isF.to_numpy(float))
        tB = tB + USE_G*_ut
        _pm=pd.Series(USE_G*_ut).groupby(B.pitcher_id.to_numpy()).mean()
        print(f"  ★경기상태 항 g={USE_G} w={int(USE_W)} · 행항 sd {np.std(USE_G*_ut):.3f} · 투수평균항 sd {_pm.std():.3f}",flush=True)
        print(f"  ★기용 보정 g={USE_G} · "+"  ".join(f"{k}={v:+.4f}" for k,v in zip(USE_KEYS,TA["use_b"])),flush=True)
    if MAG_G:
        _pn=B.pitcher_id.map(TA["pit_n"]).fillna(0.).to_numpy(float)
        _hm=B.pitcher_id.map(TA["pmag_sum"]).fillna(0.).to_numpy(float)
        _e=(_hm+MAG_A*TA["mag_prior"][cells(B)])/(_pn+MAG_A)
        tB = tB - MAG_G*(_e-TA["mag_mu"])/max(TA["mag_sd"],1e-9)
        print(f"  빗나간크기 보정 적용 γ={MAG_G} α={MAG_A} (mu={TA['mag_mu']:.4f} sd={TA['mag_sd']:.4f})",flush=True)
    print(f"\n[{HOLD_Y} 홀드아웃]",flush=True)
    sc(yB,EBG,pidB,"투수EB(전역prior)=베이스라인설계")
    sc(yB,EB,pidB,"투수EB(배치셀prior) α=2000·β=0.6")
    sc(yB,P3,pidB,"LightGBM")
    sc(yB,anchor(P3,EB@G,1.0),pidB,"L6 anchor (라벨행만)")
    for _m in ("mult","sf","proj"):
        _pm=anchor(P3,tB,1.0,mode=_m); _raw=P3+(((tB-P3@G)/(G@(np.array([1.,0.,-1.]) if _m=="sf" else np.array([170.,-40.,-130.])/3.)))[:,None]*(np.array([1.,0.,-1.]) if _m=="sf" else np.array([170.,-40.,-130.])/3.)) if _m!="mult" else _pm
        sc(yB,_pm,pidB,f"  anchor[{_m}] · 클립행 {((_raw<1e-6).any(1)).mean()*100:.2f}%")
    best=sc(yB,anchor(P3,tB,1.0),pidB,f"★L7 anchor[{ANCHOR_MODE}] (A_T={A_T:.0f} β={BETA_T})")
    # ★★ 게이트: 홀드아웃 모델을 **제출 경로 그대로** 다시 돌려 점수가 재현되는가
    #    (L8 사고: 학습은 EB@G 를, 추론은 prior 를 목표로 써서 Player τ −0.0798)
    import tempfile, shutil
    TMPROOT=tempfile.mkdtemp(); TMP=os.path.join(TMPROOT,"model"); os.makedirs(TMP,exist_ok=True)
    bst.save_model(f"{TMP}/main.txt",num_iteration=nb)
    for _k,_b in enumerate(BSTS[1:],1): _b.save_model(f"{TMP}/main_{_k}.txt",num_iteration=nb)
    TG=dict(TA)
    if USE_PROXY:
        TG["pgr_sum"]=gtabA["sum"].to_numpy(float); TG["pgr_n"]=gtabA["size"].to_numpy(float)
        TG["pgr_idx"]=gtabA.index.to_numpy()
    save_tables(TG,list(XA.columns),1.0,TMP)
    mm=json.load(open(f"{TMP}/meta.json",encoding="utf-8")); mm["mode"]="anchor"
    json.dump(mm,open(f"{TMP}/meta.json","w",encoding="utf-8"),ensure_ascii=False)
    B2=make_bundle(TMPROOT)
    B29=["row_id","season","game_month","game_dayofweek","inning","top_bottom","game_type",
     "balls_before","strikes_before","outs_before","run_top_before","run_bot_before","run_total_before",
     "score_diff_home","score_diff_pitcher_team","runner_on_1b","runner_on_2b","runner_on_3b",
     "num_runners_on","base_state","home_win_expectancy","away_win_expectancy","li","pitcher_id",
     "batter_id","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id"]
    FT=B[B29].reset_index(drop=True)
    PG=predict(FT,B2)
    print("\n[★ 게이트] 제출 경로(29열) 재현",flush=True)
    sg=sc(yB,PG,pidB,"제출 script.py 경로")
    dd=abs(sg-best)
    print(f"   학습 내부 {best:.6f} vs 제출 경로 {sg:.6f} · 차이 {dd:.2e}  "
          f"{'✓ 통과' if dd<1e-6 else '★★ 실패 — 학습/추론 불일치'}",flush=True)
    HO=os.environ.get("HOLD_OUT","")
    if HO:
        shutil.rmtree(HO,ignore_errors=True); shutil.copytree(TMPROOT,HO); shutil.copy(os.path.join(SUB,"script.py"),os.path.join(HO,"script.py")); print("  홀드아웃 번들 저장",HO,flush=True)
    shutil.rmtree(TMPROOT,ignore_errors=True)
    if dd>=1e-6:
        raise SystemExit("학습/추론 불일치 — 제출 금지")

    if os.environ.get("SKIP_FINAL","0")=="1":
        print("  (SKIP_FINAL=1 · 전체 재학습 생략)",flush=True); return
    print("\n=== 전체 재학습 (2019~2024)",flush=True)
    gm=np.array([(dl.y==j).mean() for j in range(3)]); pri=prior_table(dl,gm)
    _pw=lambda x: x if PGR_HIST<=0 else x[x.season>=int(x.season.max())-PGR_HIST+1]
    gtab,r2=proxy_pitcher(_pw(d))
    print(f"  ★대리등급 표(최종): 시즌 {sorted(_pw(d).season.unique())}",flush=True); print(f"  대리표 R² {r2:.4f} · 투수 {len(gtab)}",flush=True)
    XF,dF=loso(dl); nf=int(round(nb*len(dl)/len(A)))
    print(f"  트리 {nb} → {nf}",flush=True)
    _dsF=lgb.Dataset(XF,label=dF.y.to_numpy().astype(int))
    bf=lgb.train(PAR,_dsF,nf); BFS=[bf]
    for _k in range(1,NSEED):
        _p=dict(PAR); _p.update(seed=_k*101,bagging_seed=_k*101+1,feature_fraction_seed=_k*101+2,data_random_seed=_k*101+3)
        BFS.append(lgb.train(_p,lgb.Dataset(XF,label=dF.y.to_numpy().astype(int)),nf))
    if NSEED>1: print(f"  ★최종 q 시드 앙상블 {NSEED}개",flush=True)
    dlh = dl[dl.season>=dl.season.max()-RECENT_HIST+1] if RECENT_HIST>0 else dl
    fR=fF=0.0
    if ENV_ADJ:
        dlh, fR, fF = env_center(dlh)
        print(f"  최종 환경 제거: 목표 R {fR:.3f} · F {fF:.3f} (차이 {fF-fR:+.3f})",flush=True)
    TF=build_tables(dlh,gm); TF["prior"]=pri; TF=mag_tables(dlh,TF); TF["lvl_diff"]=lvl_diff_of(dl)
    TF["env_R"]=fR; TF["env_F"]=fF
    if HAND_G: TF["hand_dev"]=hand_dev_of(dl)
    if EXP_G:
        ei,ev,em,es=exp_of(dl); TF["exp_idx"]=ei; TF["exp_val"]=ev; TF["exp_mu"]=em; TF["exp_sd"]=es
    if USE_G:
        _ss=[x for x in sorted(dl.season.unique()) if x>=int(dl.season.min())+2]
        ub,um,us=usage_fit(dl,_ss,dl); TF["use_b"]=ub; TF["use_mu"]=um; TF["use_sd"]=us
        print(f"  기용 계수(최종) 시즌 {_ss} · "+"  ".join(f"{k}={v:+.4f}" for k,v in zip(USE_KEYS,ub)),flush=True)
    if LVLN_G and not NPRI_G:
        _lz=np.log1p(TF["pit_n"].to_numpy(float)); TF["npri_mu"]=float(_lz.mean()); TF["npri_sd"]=max(float(_lz.std()),1e-9)
    if NPRI_G:
        _ss=[x for x in sorted(dl.season.unique()) if x>=int(dl.season.min())+1]
        nb,nm,ns=npri_fit(dl,TF["pit_n"],_ss); TF["npri_b"]=nb; TF["npri_mu"]=nm; TF["npri_sd"]=ns
        print(f"  ★이력크기 사전(최종): b={nb:+.4f} · log1p(n) 평균 {nm:.3f} sd {ns:.3f}",flush=True)
    if COMP_G:
        ci_,cv_=comp_table(d,int(d.season.max())); TF["comp_idx"]=ci_; TF["comp_val"]=cv_
    TF["tstar"]=str(d[d.game_type=="F"].pitcher_team_id.value_counts().index[0])
    if RGV:
        vi,vr,vg,vrm,vgm=rgv_of(d,dl,float(os.environ.get("A_T","1800")))
        TF["rgv_idx"]=vi; TF["rgv_r"]=vr; TF["rgv_g"]=vg; TF["rgv_rmu"]=vrm; TF["rgv_gmu"]=vgm
        print(f"  ★성분분리(최종): 투수 {len(vi)} · r 창 최근{RGV_H}시즌 · r 평균 {vrm:.4f} · g 평균 {vgm:.3f}",flush=True)
    if RGSPLIT:
        ri,gv,rv,gmu,rmu,rd=rg_of(dlh)
        TF["rg_idx"]=ri; TF["rg_g_val"]=gv; TF["rg_r_val"]=rv
        TF["rg_gmu"]=gmu; TF["rg_rmu"]=rmu; TF["rg_rdiff"]=rd
        print(f"  ★곱분해(최종): 투수 {len(ri)} · F−R 반대투구율차 {rd:+.4f}",flush=True)
    if TRG_G or TRF_G:
        ti,lv,f1,fa=trans_of(dl); TF["lastteam_idx"]=ti; TF["lastteam_val"]=lv
        TF["fs1_val"]=f1; TF["fsall_val"]=fa
        print(f"  ★전환표(최종): 투수 {len(ti)}",flush=True)
    if CELL_G:
        ci,cv,cm=pcell_of(dlh,pri); TF["pcell_idx"]=ci; TF["pcell_val"]=cv; TF["pcell_mu"]=cm
        print(f"  ★상황구성(최종): 투수 {len(ci)} · 투수간 sd {cv.std():.3f}",flush=True)
    if BAT_G:
        bi,bv=bdev_of(dlh); TF["bdev_idx"]=bi; TF["bdev_val"]=bv
        print(f"  ★타자 편차(최종): 타자 {len(bi)} · sd {bv.std():.3f}",flush=True)
    if LOCSD_G:
        _ss=[x for x in sorted(dl.season.unique()) if x>=int(dl.season.min())+2]
        li,lv,lm,lsd,lb=locsd_fit(dl,dlh,_ss); TF["locsd_idx"]=li; TF["locsd_val"]=lv
        TF["locsd_mu"]=lm; TF["locsd_sd"]=lsd; TF["locsd_b"]=lb
        print(f"  ★로케이션 산포(최종): 투수 {len(li)} · 계수 b={lb:+.4f} (적합 {_ss})",flush=True)
    if ENVAX: TF=envax_tables(dl,TF)
    if LVL_TEAM_G or HTM_G:
        ti,td,_=team_dev_of(dl); TF["team_idx"]=ti; TF["team_dev"]=td
    if SEA_ADJ: TF=sea_tables(dlh,TF,sea_only=os.environ.get("SEA_ONLY","0")=="1")
    if RECENT_HIST>0: print(f"  최종 투수 이력 표 = 최근 {RECENT_HIST}시즌 ({sorted(dlh.season.unique())}) · {len(dlh):,}행",flush=True)
    if USE_PROXY:
        TF["pgr_sum"]=gtab["sum"].to_numpy(float); TF["pgr_n"]=gtab["size"].to_numpy(float)
        TF["pgr_idx"]=gtab.index.to_numpy()
    os.makedirs(OUT,exist_ok=True)
    bf.save_model(f"{OUT}/main.txt",num_iteration=nf)
    for _k,_b in enumerate(BFS[1:],1): _b.save_model(f"{OUT}/main_{_k}.txt",num_iteration=nf)
    save_tables(TF,list(XF.columns),1.0,OUT)
    m=json.load(open(f"{OUT}/meta.json",encoding="utf-8")); m["mode"]="anchor"
    json.dump(m,open(f"{OUT}/meta.json","w",encoding="utf-8"),ensure_ascii=False)
    print(f"  저장 {OUT} (피처 {len(XF.columns)}열 · a_p={m['a_p']} beta={m['beta']} a_t={m.get('a_t')} beta_t={m.get('beta_t')})",flush=True)

if __name__=="__main__": main()
