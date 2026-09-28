# LG Aimers 9기 오프라인 해커톤 — KBO 투구 제구등급 예측 (2026-09-19 ~ 09-20)

> 문제정의 → EDA → 아키텍처 → 실험 → 트러블슈팅 → 결과 → **재현 가이드** → 레포 맵. 수치는 전부 LB 실측 또는 로그 인용.
> 팀 miniL 4인(dain·종민·seongmin·celine). 최종 채택 = **`submit_dFING2.zip` — Score 70.3140 / Pitch 70.1038 / Player 70.8044** (팀 최고, 마감 09-20 10:00).

> **레포 경로 안내** — 이 문서는 로컬 아카이브(`~/Documents/lg_aimers/archive/`) 기준으로 쓰였다. 이 레포에서는 `archive/dain_src/` → `src/`, `archive/dain_src/l236x/log/` → `logs/final_builds/`, `archive/dain_log/` → `logs/pipeline/`, `archive/dain_eda/` → `eda/`, `archive/dain_out/*.json` → `results/`, `archive/EDA_SUMMARY.md` → `docs/eda_summary.md` 로 옮겼다. 원본 데이터·팀원 코드(`jm_*`, `seongmin_*`, `celine_*`, `team_docs/`)·목표 번들·제출 zip 은 포함하지 않는다.

---

## 0. 요약 카드

| 항목 | 내용 |
|---|---|
| 과제 | 투구 직전 29열로 제구 등급 3-class(Shadow/Heart/Failure) 확률 예측 · 학습 2019~24 · 평가 2025 |
| 지표 | Score = 0.7·Pitch(Brier) + 0.3·Player(투수별 평균 예측점수의 Kendall τ_b, 50구+ 투수 321명) |
| 핵심 통찰 | Pitch 는 포화(70.1~70.25) · 전장은 Player · Player 는 "실력"이 아니라 "2025 평가행 구성에서의 투수 순위" · 2023 F 체제변화 · 오프라인 검증이 Player 를 못 맞춤(0/7) |
| 아키텍처 | anchor 분리: q(GBDT, Pitch) 와 t(투수 목표값, Player) 를 분리하고 proj 사영으로 결합 → q 를 바꿔도 Player 비트 불변 |
| 내 기여 | 하이브리드(팀 목표 + 내 q, Pitch +0.04), 볼카운트 행 항 d22(Player +1.08, 팀 표준 부품), 마지막 날 감사·변형 30여 종·후보정·최종 조립(dFING2) |
| 최종 | FIN_G 목표(종민) + q = 0.5·dain + 0.5·종민, 온도 T=0.9 → Pitch +0.010, Player 동일 |

---

## 1. 문제 정의

**데이터** (`archive/data/`, 명세 `data_description.md`)
- `train.csv` 1,474,868행 × 29열 (2019~2024) · `train_outcomes.csv` 라벨·위치·반대투구·스트/볼·결과코드·구종 (라벨 결측 17.16% = Unlabeled, Failure 아님) · `train_trackman.csv` 물리량 31열(부분 연결) · `test.csv`/`sample_submission.csv` 는 더미 5행(실제 평가 218,992행은 비공개).
- 29열: season, game_month/dayofweek, inning, top_bottom, game_type(R/F), balls/strikes/outs_before, run_top/bot/total_before, score_diff_home/pitcher_team, runner_on_1b/2b/3b, num_runners_on, base_state, home/away_win_expectancy, li, pitcher_id, batter_id, pitcher_hand, batter_hand, pitcher_team_id, batter_team_id.
- 라벨 규칙(100% 재현됨): Heart |x|≤0.263 & 0.537≤y≤0.963 · Shadow |x|≤0.373 & 0.427≤y≤1.073 (Heart 제외) · 나머지 또는 REVERSE_YN=T → Failure.

**평가식** — `Score = 0.7·Pitch + 0.3·Player`, Pitch = 100·(1 − Brier/2), Player = 50·(1 + τ_b), τ_b = 투수별 mean(100·p_S + 30·p_H) vs 실제 mean 의 Kendall τ_b (평가 50구+ 투수). 감도: Brier 0.01 = Score 0.35, τ 0.01 = Score 0.15. 제출 = 코드 zip(script.py·requirements.txt·model/), 채점 서버 6 vCPU·10분·인터넷 차단·**py3.11 / pandas 2.0.3 / numpy 1.26.4 / lightgbm 4.7.0**. 1일 100회, Public=Private(test 100%).

**규정(실격)** — 평가 행 독립: test 를 투수별로 집계·순위·분포보정 금지. 허용 = (a) 학습데이터로 만든 표를 자기 열로 조회, (b) 같은 행 안의 열 조합. 사후정보(위치·반대투구·결과·구종·Trackman)는 과거 이력 집계로만. 외부 데이터 금지. 결측 라벨 ≠ Failure.

---

## 2. EDA 핵심 (`archive/EDA_SUMMARY.md`, `archive/dain_eda/`, 팀 `team_docs/`)

1. **Pitch 포화**: 상수 prior 69.71 → 카운트 표 69.85 → 투수 EB 69.97 → LGB 70.0. 베이스라인 LB 70.11, 상위권 70.18~70.25. **점수는 τ 에서 난다** (τ 0→0.3 = +4.5).
2. **평가셋 = 2025 라벨 확정 투구만** (218,992 ≈ 2024 라벨행 218,836). 50구+ 투수 321명이 99.4% 를 던짐. 신인(이력 0) 60명(19%), 이력<1000 154명.
3. **2023 F 체제변화**: 퓨처스(F) 반대투구 기록률 2022 4.9% → 2023 30% → 2024 29% (같은 투수 70명 기준; 신인 유입 아님). F−R 점수차 2022 +6.0 → 2023 −2.5. LGB 에 `season` 열 하나로 τ 0.20→0.32 (+1.8 Score). 2025 는 전 구장 ABS+Trackman 첫 해.
4. **투수 실력의 정체**: Shadow 해간 지속성 0.34(참 sd≈0) — 실력은 Heart↔Failure 배분. 반대투구율 지속성 0.61~0.67(체제 안에서). 투수 간 참 sd 1.76 ≈ 투구 523개 평균의 SE 1.60 → **신호대잡음 1:1**, 완벽한 실력을 알아도 τ 0.59~0.65 상한.
5. **상황은 실력과 역상관**: 0-2 카운트 평균 23.3 vs 0-0 29.1 — 좋은 투수가 유리한 카운트를 만든다. 상황만으로 순위를 매기면 τ −0.23. 그래서 GBDT 단독은 Player 붕괴(d4 60.2).
6. **병목 군** (팀 114번, as-of 2024): 이력 0 τ +0.435(양호) · **1~300구 −0.048** · R 위주 혼합 +0.076 · **F 관여 투수 147명(46%) 군내 τ 0.05~0.08 = 무작위** · 3k+ +0.374. 소이력×대이력 교차쌍(44%) τ 0.137.
7. **오프라인 검증 무효**: 폴드·as-of 패널의 Player 부호가 LB 와 0/7 일치(SIT·SH·구성항·대리등급·역할·drift·타자). Pitch 만 이관율 0.5~1.0 로 신뢰.

---

## 3. 아키텍처와 용어

```
q  = LightGBM 3-class 5시드 (46 피처: 상황 15 + 투수/타자 EB(배치셀 prior) + 투수×카운트 잔차 + Trackman 프로필; LOSO 피처)   → Pitch
t  = 투수 목표 제구점수 (행마다 계산되지만 투수 평균이 Player 를 결정)                                                             → Player
   = pgr-EB : (Σ 이력 점수 + α_t·배치셀 prior 점수)/(n + α_t), α_t 2400, 이력 = 라벨행 + 대리등급(pkey) 채운 미라벨행, 시즌중심화(PGR_CTR), PGR2(seongmin, n≤500)
   + 행 항  : γ_F·lvl_diff·isF (−0.6·−2.22 = F 행 +1.33) · 볼카운트 0.5·w_p·Σβ·z(볼·스트·주자·득점·아웃) · 플래툰 0.6·hand_dev · 팀 −0.25·team_dev · 상황구성모델 cmp −0.15·(cmp(행) − μ)
   + poff   : 투수 오프셋 = 0.3·GEO + 직교잔차 {RK .25, SIT .25, SH .20, PLT .20, DIR .15, DSHs .12} (직교화 축 = L164 값·log n·이력 F비중, 총 sd ≈ 0.5)
   + r12    : 투수×카운트12×같은손 조건부 편차 (n/(n+600)·((셀−투수)−(리그셀−리그)), 시즌중심 점수)
   + r13    : 투수×역할 조건부 편차 (FIN_G, 종민, +0.035)
p  = anchor_proj(q, t) : p = q + (t − q·G)/(G·v)·v,  v = (170,−40,−130)/3 (제약 100p_S+30p_H = t, Σp=1 하의 L2 최소해) → clip 0%
```

| 용어 | 뜻 |
|---|---|
| EB | Empirical Bayes 수축 평균. 배치셀 prior = li3×inn4×sd5 60셀(감독의 기용이 실력을 드러냄, +0.66) |
| pgr / PGR2 / PGR3 | 대리등급 표: 미라벨행을 (STRK·BALL·RESULT·PITCH_KIND·rev·카운트) 조합 평균으로 채움. PGR2 = 문맥 조건 추가(+0.054). PGR3 = GBDT(R²↑) → LB −0.07~−0.14 |
| anchor(proj) | 행 기대점수를 t 로 고정. Player = τ(t) 정확 성립 → **q 변경은 Player 불변** (LB 3회 10자리 동일 확인) |
| LOSO | 학습 피처를 leave-one-season-out 으로 만들어 인샘플 오염 제거(+1.80) |
| GEO | 반대투구 태그를 무시한 위치 기하 재채점 EB − 등급 EB (0.3 배, +0.21) |
| RK | 투수-시즌 패널 lambdarank(이전 시즌만 → 다음 시즌 순위) 직교 잔차 (+0.236) |
| SIT / SH / DIR / PLT / DSHs | 상황보정 잔차 EB / Shadow·Heart 분리 EB / Dirichlet α 분리 / 같은손 노출 비율 / 같은손 상대 Shadow 비율 |
| cmp | 선수·팀 ID 없는 24파생으로 행 기대점수 예측(2023~24 학습, `comp.txt`). 음수(−0.15)가 정답 = 상황 이점을 뺀다 |
| d22 항 | dain `usage_fit.py`: 투수 잔차 ~ 행 상태 z 릿지, β balls −5.74 · strikes +2.51 · runners −1.43 · run_total −0.54 · outs +1.32, w_p = 2000/(n+2000), scale 0.5 |

**내 초기 파이프라인**(`archive/dain_src/pipeline.py·train.py·submit/script.py`): 62피처 LGB(lr .02, leaves 63, minleaf 1000, 677 rounds, 5시드) + EB anchor + 주입/skill prior 옵션. 최종에서는 이 q 가 종민 q 와 평균된다.

---

## 4. 실험

### 4-1. dain LB 제출 전부 (Score / Pitch / Player)

| 시각 | zip | 구성 | LB | 판정 |
|---|---|---|---|---|
| 09-19 낮 | d4 | LGB 69피처(season·game_type·이력·Trackman·위치이력) 5시드, gbdt 단독 | 67.198 / **70.206** / 60.179 | Pitch 최고급·Player 붕괴 → 분리 구조 필요 |
| | d5 | anchor: EB(α2000, 배치셀 prior) 목표 + noF GBDT 배분 | 68.627 / 70.166 / 65.036 | 구조 채택 |
| | d6 | d5 + 위치 이력 신호 주입 ε.1 (3폴드 ✓) | 68.582 / 70.167 / 64.881 | 폴드↔LB 역전 첫 관측 |
| | d7 | d5 + 빗나간 크기(far50·chase) 주입 ε.4 (3폴드 ✓3/3) | 68.455 / 70.174 / 64.444 | ✗ |
| 09-19 밤 | d20 | **하이브리드**: 종민 L87 목표 그대로 + dain 5시드 q (가산형 anchor) | 69.6032 / 70.1623 / 68.2985 | Player 가 L87 과 10자리 동일 · Pitch +0.044 |
| | d21 | d20 + **볼카운트 행 항** scale 1.0 | Player +0.54 | |
| | d22 | 〃 scale 0.5 | **69.8864 / 70.1034 / 69.3802** | **+1.08 Player, 당시 팀 최고** → 팀 표준 부품(L110→L164→…) |
| 09-20 04:16 | d28 | L236 + comparable prior 소이력 직교 sd.10 + ncut1000 | 70.1112 / 70.1243 / 70.0807 | Player −0.478 |
| 05:14 | d30b | O57 + drift-anchor κ.5 + 타자 질 −0.5 + dain q | 69.8941 / 70.1143 / 69.3802 | Player −1.34 (타자 양방향 사망) |
| 05:57 | d36e | TS + r12 3변형 평균 + soft 재라벨 c.2 | 70.2844 / 70.0956 / 70.7251 | −0.045 |
| 09:04 | dFIN | TS + r12 3변형 평균 · q = .5 dain + .5 jm · T .9 | 70.2792 / **70.1056** / 70.6844 | Pitch +0.012 · r12 평균 −0.085 ⇒ soft c.2 = +0.041 |
| 09:33 | dLAST2 | TS + soft c.25 + RK 재직교화 · 같은 q | 70.2959 / 70.1062 / 70.7386 | RK 재직교화 ≈ −0.07 |
| **09:42** | **dFING2** | **FIN_G 목표 그대로 + q .5/.5 · T .9** | **70.3140 / 70.1038 / 70.8044** | **최종** (Pitch +0.010, Player 비트 동일) |

(참고 팀 기준선: L236 70.2419/70.1062/70.5586 · O57 70.2797/70.0930/70.7154 · TS_P600G10 70.2965/70.0937/70.7696 · FIN_G 70.3072/70.0941/70.8044)

### 4-2. 빌드했지만 미제출 (전부 게이트 통과, 구성·수치는 `archive/dain_out/submissions.md`, `docs/notes/work_log_0920.md`)
skill prior Cf/Af(d10/d15, 폴드 τ +.06~.10) · soft 재라벨 d14(폴드 +.04τ) · CTR d19 · 이력구간 d21 · TabPFN d8/d12 · erank(비트동일 실패) · GEOF/SITb/CNTH/CPs 오프셋 · r12 등급별/2단풀링/최근/저랭크(d34*) · 대리등급 가중 w=0/.5 · soft 투여량 0.1~1.0 · 스택(d36*) · drift κ 스캔(d29*) · Player 불변 q 앙상블 스캔(d37*, d38tv/ts) · dFING3/4(soft .2/.6 + FIN_G).

### 4-3. 팀 전체 계보 (`team_docs/제출기록.md`, 170여 건)
L5 65.0 → γ_F −0.6 (+2.61) → α 4000→1800 (+0.31) → **d22 카운트 (+1.08)** → GEO .3 (+0.21) → 직교잔차 RK/SIT/SH (+0.47) → PLT (+0.28) → DIR (+0.18) → L236 70.56 → α2400 (+0.02) → cmp −0.15 (+0.08) → PGR2 O57 (+0.05) → r12 TS (+0.05) → r13 FIN_G (+0.035) → **dFING2 q 앙상블 (Pitch +0.010)**.

---

## 5. 트러블슈팅 (재발 방지)

1. **폴드/패널이 Player 를 못 맞춘다 (0/7)** — 2025 는 전 구장 ABS 첫 해, F 체제는 2023 에 바뀜 → 2024 폴드의 최적 보정이 2025 와 반대. 대응: Player 는 LB 1회로 판정, 후보는 LB 이웃이 양수인 가족에서만. Pitch 는 폴드×0.5~1.0.
2. **홀드아웃에 미래 학습 부품 혼입** — 2023~24 로 학습된 `comp.txt` 를 ≤2023 홀드아웃에 넣어 drift-anchor Pitch +0.087 이 누수(실제 +0.015). 홀드아웃에 들어가는 모든 부품의 학습 창을 확인.
3. **학습/추론 불일치 (팀 L8, Player τ −0.08)** — target_grade 가 prior 를 반환하는 분기. 대응: `make_bundle()` 단일화 + 학습 스크립트 내 "29열 가짜 test 재현" 게이트(차이 <1e-6). 내 빌더는 (a) 행 독립(부분집합·단일행·역순 <1e-10) (b) 확률 형식 (c) 2024 재현 (d) 투수 평균점수 비트 비교(7e-15) 자동화.
4. **평가서버 환경** — pickle 금지(numpy 2 피클 → 1.26 못 읽음). 표는 npz+json, 모델은 LightGBM txt. venv_eval(py3.11) 로 더미 실행 후 제출.
5. **병행 작업 충돌** — 같은 폴더의 두 작업이 `submit_dN` 을 덮어씀(d11 소실). 빌드 직전 `ls out/`, 접미사 이름.
6. **ssh 끊김** — 장시간 ssh 안의 빌드가 함께 죽음. 서버측 `chain_*.sh` 를 `setsid nohup` 으로, 완료 마커 폴링.
7. **τ_b 동일값 착시** — 다른 제출이 10자리까지 같은 Player(d30b=d22, T3_B10=L228). τ_b 는 정수 C−D 라 수백 회면 우연히 일치. 버그로 오판하지 말 것.
8. **anchor 의 Brier 비용** — 행마다 t 강제로 행 수준 상황 정보가 지워져 Pitch −0.14. 복원 시도(TQ·drift·ncut·cw)는 Player −0.5~−1.7. Pitch 회수는 **q 개선(앙상블·온도)** 만 안전.
9. **이력 정밀화는 전부 실패** — 대리등급 GBDT, 적률 α, 제구 지수, F 환산, 저랭크, 타자, 역할… 2025 로 지속되는 성분(sd≈0.9)이 시즌 내 분산(2.39)보다 훨씬 작아 강한 수축과 평가행 구성만 남는다.
10. **원칙 판단의 일관성** — 볼카운트 항을 "지표 조작"으로 폐기했던 것이 오판: γ_F·anchor 도 같은 절충. 규정 문구(자기 행 열 + 상수)로 판단해야 한다.

---

## 6. 결과와 원리

**최종** `submit_dFING2.zip` = 종민 T_FIN_G 목표(TS + r13) + dain q 앙상블(w .5, T .9). 70.3140 / 70.1038 / 70.8044. 후보정 T 는 ≤2023 홀드아웃 모델의 2024 표본외에서 anchored Brier 최소화로 적합(T .85~.9 → +0.016; 클래스 편향 항 +0.026 은 체제 의존이라 제외).

**통한 것** — 자기 행 열 × 학습 상수 행 항(F·카운트·손·팀·cmp 음수) · 강한 수축(α↑ 6/6) · 작은 직교 잔차(총 sd 0.5 정점) · 투수×상황 조건부 표(r12/r13) · **q 만 바꾸는 Pitch 개선(Player 무위험)**.
**안 통한 것** — 이력 정밀화 전부 · 소이력/F 군 겨냥 개입(양방향) · 타자 질 · 역할/구장/물리 · 차원축소 · 사후기대순위/Borda · 세기 재스캔(정점).
**남은 병목** — F 관여·소이력 147명의 군내 순위(τ 0.05 → 0.25 면 Player +2). 29열 밖의 정보가 없고 2025 체제를 오프라인에서 관측할 수 없다. 다음엔 첫날부터 (a) pitcher-season as-of 패널에서 **현재 목표값 잔차**를 목표로 한 순위모델, (b) LB 이웃 규칙, (c) Player 불변 Pitch 라인을 병행.

---

## 7. 재현 가이드 (로컬, 처음부터 끝까지)

### 7-0. 환경
```bash
# 학습/빌드용 (개발서버와 동일 계열): Python 3.12, pandas 3.x, numpy 2.x, lightgbm 4.7, scipy, scikit-learn
python3 -m venv ~/venv_lg && source ~/venv_lg/bin/activate
pip install "pandas>=2.2" numpy "lightgbm==4.7.0" scipy scikit-learn
# 평가서버 모사용 (제출물 검증): Python 3.11, pandas 2.0.3, numpy 1.26.4, lightgbm 4.7.0
python3.11 -m venv ~/venv_eval && ~/venv_eval/bin/pip install pandas==2.0.3 numpy==1.26.4 lightgbm==4.7.0
```
메모리 ~10GB(전체 데이터 로드), 12스레드 기준 시간은 각 단계에 표기. Apple Silicon 이면 lightgbm 은 `brew install libomp` 필요.

### 7-1. 서버 경로 재현 (모든 스크립트가 `/home/ubuntu/workspace/...` 절대경로를 하드코딩)
```bash
export WS=~/lg_workspace                       # 서버 레이아웃을 그대로 복원
mkdir -p $WS/산출물/{dain,종민,seongmin,celine} $WS/data
A=~/Documents/lg_aimers/archive
cp $A/data/* $WS/data/
cp -r $A/dain_src $WS/산출물/dain/src; cp -r $A/dain_eda $WS/산출물/dain/eda; mkdir -p $WS/산출물/dain/{out,log}; cp $A/dain_out/* $WS/산출물/dain/out/
cp -r $A/jm_src $WS/산출물/종민/src; cp -r $A/jm_eda $WS/산출물/종민/eda; mkdir -p $WS/산출물/종민/{out,log}
cp -r $A/final_bundles/TS_P600G10_dir $WS/산출물/종민/submit_TS_P600G10; cp -r $A/final_bundles/T_FIN_G_dir $WS/산출물/종민/submit_T_FIN_G
cp -r $A/seongmin_src $WS/산출물/seongmin/src
# 절대경로 치환 (.py .sh 전부)
grep -rl "/home/ubuntu/workspace" $WS --include=*.py --include=*.sh | xargs sed -i '' "s#/home/ubuntu/workspace#$WS#g"
grep -rl "~/workspace" $WS --include=*.sh | xargs sed -i '' "s#~/workspace#$WS#g"
sed -i '' "s#/usr/bin/python3#$HOME/venv_lg/bin/python#g; s#\$HOME/workspace/산출물/dain/venv_eval/bin/python#$HOME/venv_eval/bin/python#g; s#~/workspace/산출물/dain/venv_eval/bin/python#$HOME/venv_eval/bin/python#g" $WS/산출물/dain/src/*.sh
```
(리눅스면 `sed -i ''` → `sed -i`.)
```bash
# l236x/common.py 가 기대하는 보조 디렉토리 복원
mkdir -p $WS/산출물/dain/src/l236x/base && unzip -q $A/final_bundles/submit.L236.zip -d $WS/산출물/dain/src/l236x/base   # L236 script/model (import 기준)
mkdir -p $WS/산출물/dain/tmp_O57 && unzip -q $A/final_bundles/submit_O57.zip -d $WS/산출물/dain/tmp_O57                  # O57 기반 빌더(build_d29/var)용
cp $A/jm_out/*.csv $WS/산출물/종민/out/                                                                                  # rk_score.csv(순위모델 점수)·qp_score.csv — signals.py 가 읽음
```
이후 모든 명령은 `cd $WS/산출물/dain/src/l236x` 또는 해당 src 에서 실행.

### 7-2. 빠른 재현 — 최종 zip 재조립 (5분)
아카이브된 종민 목표 번들과 내 q 모델로 dFING2 를 다시 만든다.
```bash
cd $WS/산출물/dain/src
unzip -q ~/Documents/lg_aimers/submit_dFING2.zip -d /tmp/dfing2   # 내 q 모델(lgb_seed0~4 · tables · meta) 확보
mkdir -p submit_d20/model && cp /tmp/dfing2/model/lgb_seed*.txt /tmp/dfing2/model/meta.json submit_d20/model/ && cp -r /tmp/dfing2/model/tables submit_d20/model/
cd l236x
python build_hyb2.py dFING2_repro $WS/산출물/종민/submit_T_FIN_G 0.5 0.9
#  → 게이트(행 독립 1e-16 · 확률합) · 2024 재현 · "투수 평균점수 최대차 7e-15 (Player 불변 검증)" · out/submit_dFING2_repro.zip
```
`build_hyb2.py` 는 `common.load_d()` 로 학습 데이터를 한 번 읽어 `cache_d.pkl`(700MB) 을 만든다(첫 실행 ~3분). 생성된 zip 의 `model/jm_root` = T_FIN_G, `hyb.json` = {w_dain .5, q_T .9}. 원본과의 차이는 dain q 파일이 동일하므로 0 이어야 한다(`unzip -l` 로 크기 비교).

### 7-3. 전체 재현 — 학습부터
**(A) 종민 목표 번들 (TS_P600G10 까지, ~1시간)**
1. 기본 번들(L236 계열, 5시드 q + 표): `train_L7.py` 는 `SUBDIR/script.py` 를 import 하므로 `SUBDIR=$WS/산출물/종민/submit_TS_P600G10` (그 script.py 가 최신 추론 코드).
   ```bash
   cd $WS/산출물/종민/src
   SUBDIR=$WS/산출물/종민/submit_TS_P600G10 MODEL_OUT=$WS/산출물/종민/submit_BASE/model \
   NSEED=5 NTH=12 A_T=2400 BETA_T=0 A_P=2000 BETA=0 LVL_G=-0.6 LVL_TEAM_G=-0.25 HAND_G=0.6 \
   USE_G=0.5 USE_KEYS=balls_before,strikes_before,num_runners_on,run_total_before,outs_before USE_W=1 USE_AW=2000 \
   ANCHOR_MODE=proj PGR_CTR=1 USE_PROXY=1 HOLD_Y=2024 python -W ignore train_L7.py > ../log/base.log
   ```
   출력: 2024 홀드아웃 점수·게이트(학습 내부 vs 제출 경로 차이 0)·`submit_BASE/model/{main.txt, main_1~4.txt, tables.npz, meta.json}`. 배치셀 prior·pgr(대리등급, PGR_CTR)·hand_dev·team_dev·use_b(볼카운트 β — d22 계수와 동일 방식으로 재적합) 가 여기서 만들어진다. ※ 볼카운트 β 의 원본은 `archive/dain_out/usage_B.json`(dain `usage_fit.py` 산출) — 종민 번들은 이 값을 그대로 meta 에 넣었다(TS meta 의 use_b 와 일치 확인 가능).
2. 순위모델: `python factory_rank.py` → `out/rk_score.csv` (`BASE`=기본 번들, dain `usage_B.json` 읽음, 5분).
3. 투수 오프셋: `BASE=$WS/산출물/종민/submit_BASE python factory_orth.py "L236r|RK:0.25,SIT:0.25,SH:0.20,PLT:0.20,DIR:0.15,DSHs:0.12|0"` → `submit_L236r/`(poff 표 주입, 5분). GEO·SIT·SH·PLT·DIR·DSHs 정의는 이 파일 안에 있다.
4. 상황구성모델 `comp.txt` 와 PGR2·cmp_g: 종민 작업 중 인라인으로 만들어져 **학습 코드가 파일로 남아있지 않다.** 레시피(로그): 2023~24 라벨행 431,438 · 24파생(`CMP_COLS`, script.py `_cmp_feats`) · LightGBM 3-class 400트리 · `cmp_g=−0.15, cmp_mu=27.111`. PGR2 = `archive/seongmin_src/build_O55_O56_pgr2.py`(pkey + game_type|투수손|타자손|이닝군 조건 평균, pseudocount 80, n≤500). 재학습 대신 아카이브된 `submit_TS_P600G10/model/{comp.txt, tables.npz}` 를 쓰면 된다.
5. r12 표: `cd $WS/산출물/dain/src/l236x && python build_ts.py TSr base --nozip` → TS_P600G10 표와 **최대차 0** 재현(레시피: 시즌중심 점수, 키 `pid|cnt|samehand`, λ600). r13(역할 조건부, FIN_G)은 종민 인라인 코드라 표만 보존(`T_FIN_G_dir/model/tables.npz`, 조회 코드는 그 script.py `r13_*`).

**(B) dain q 모델 (30분)**
```bash
cd $WS/산출물/dain/src
python train.py final 0,1,2,3,4 677 --drop "b_,season,game_type,pgt_,p_fshare" --combine anchor --lam 1.0 --out submit_d20/model
```
`pipeline.py` 가 표(`model/tables/pitcher.csv, batter.csv`)를, `train.py` 가 `lgb_seed{0..4}.txt` 와 `meta.json`(features 62, params lr .02/leaves 63/minleaf 1000/λ2 1) 을 만든다. 검증: `python train.py val 0 --drop ... --combine anchor` → 2024 폴드 Total/Pitch/τ.

**(C) 최종 조립·후보정·검증**
```bash
cd $WS/산출물/dain/src/l236x
python fit_calib.py            # (선택) ≤2023 홀드아웃(train_hold.py, HOLD_OUT=hold24) 위에서 T 재적합 → calib.json (T≈0.85~0.99)
python build_hyb2.py dFINAL $WS/산출물/종민/submit_T_FIN_G 0.5 0.9
# 평가서버 모사: 더미 5행 + 2024 라벨행 218,836 → 2025 로 바꾼 가짜 test 로 타이밍(<10분)·형식
bash ../pack_and_test2.sh dFINAL_chk $WS/산출물/dain/src/submit_dFINAL/model $WS/산출물/dain/src/submit_dFINAL
```
`build_hyb2.py` 가 자동으로 (1) 30k 학습행 부분집합·단일행·역순 행 독립, (2) 확률합·범위, (3) 2024 라벨행 재현 점수, (4) 목표 번들 대비 투수 평균점수 최대차(Player 불변) 를 출력한다. 채점 서버와 같은 환경은 `~/venv_eval/bin/python script.py` 로 확인.

### 7-4. 실험 재실행 지도
| 무엇 | 스크립트 (`dain_src/l236x/`) | 비고 |
|---|---|---|
| L236 감사(poff 재구성·블록 평균·베테랑 편향) | `diag_poff.py`(tmp_L236 기준) | 결과 `docs/notes/work_log_0920.md` §1 |
| poff 신호 정의 + 새 신호(GEOF·SITb·CNTH·CPs) | `signals.py`, `build.py NAME "RK:0.25,..." [--ncut N]` | L236 기반 |
| r12 변형(등급별·2단·최근·저랭크) | `build_ts.py NAME dir|hier|rec|lowrank ...` | TS 기반 |
| F 행 재측정(soft 재라벨·기하) | `build_tbl.py NAME soft|gt23|gtall --c C` | pgr 델타 패치 |
| 대리등급 가중 / 타자 항 / drift | `build_var.py`, `build_d29.py`, `build_pgrw.py` | O57·하이브리드 기반 |
| Player 불변 q 앙상블·온도 | `build_hyb2.py`, `hyb2_script.py`, `fit_calib.py` | 최종 경로 |
| as-of 검증(≤2023→2024) | `train_hold.py`(HOLD_OUT), `asof.py`, `asof_var.py`, `pitchrec*.py`, `eval24.py` | comp.txt 누수 주의(§5-2) |
| 초기 파이프라인 실험 | `../{skill_prior,usage_fit*,build_pre23f,pfn_level,miss_magnitude,eb_resid,prior_reg,taulab}.py`, `run_*.sh` | 각 파일 docstring 참조 |

---

## 8. 레포 맵

| 경로 | 내용 |
|---|---|
| `final_submission/` | 최종 제출물 dFING2 의 코드 (script.py · dain_feats.py · model/{meta.json, hyb.json}). 모델 가중치·팀원 목표 번들(`model/jm_root`)은 제외 |
| `src/` | 초기 파이프라인(`pipeline.py`·`train.py`·`submit/script.py`(추론)·`skill_prior.py`·`usage_fit*.py`·`build_pre23f.py`·`verify*.py`·`make_submit.sh`·`pack_and_test2.sh`·`make_hybrid.sh`·`run_*.sh`), `script_hybrid/`(d20 형 하이브리드 추론) |
| `src/l236x/` | 마지막 날 빌더·진단·체인 스크립트 + `calib.json`·`cw_val.json` |
| `logs/final_builds/` · `logs/pipeline/` | 마지막 날 빌드 로그 88개 · 초기 파이프라인 학습 로그 |
| `eda/` · `docs/eda_summary.md` | EDA 10~25 (`_common.py` 에 공식 지표 `score()`, `_feat.py` 피처 빌더) |
| `results/` | usage β json(`usage_B.json` = d22 계수), val json |
| `docs/submissions.md` | dain 제출 전부·LB |
| `docs/notes/` | 당일 작성 문서 (작업 기록 · EDA 미시도 인사이트 · 방법론 조사 · 파생변수 전략 · 후처리 전략) |

## 9. 타임라인·역할

- 09-19 13:00~18:00 EDA(10~19) → season 열·체제변화 발견 → d1/d4 → anchor 구조(d5) → 주입 실험(d6/d7)
- 09-19 18:00~24:00 skill prior·soft 재라벨·CTR·이력구간·TabPFN(폴드 양수, LB 미제출) → **d20 하이브리드 → d22 볼카운트 항(팀 최고)**
- 09-20 03:20~06:00 L236 감사(찐빠 3건)·GEOF/CPs/drift/타자/soft 변형·d28·d30b·d36e
- 09-20 06:00~10:00 TS 파쿠리(r12 비트 재현)·후보정 적합·Player 불변 q 앙상블 → dFIN·dLAST2 → **dFING2 최종(09:42)** → 아카이브
- 팀: 종민(목표값 파이프라인·LB 판정 규율), seongmin(PGR2·S 시리즈), celine(Trackman·r12 주입), dain(하이브리드·행 항·감사·후보정·최종 조립)
