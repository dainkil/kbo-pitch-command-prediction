# dain 작업 기록 — 2026-09-20 03:20~06:15 (L236 고도화 → TS 파쿠리)

기준선 변천: L236 (70.2419 / 70.1062 / 70.5586) → O57 (70.2797 / 70.0930 / 70.7154) → **TS_P600G10 (70.2965 / 70.0937 / 70.7696, 팀 최고)**. 표기 = Score / Pitch / Player.

## 1. L236 찐빠 감사 (읽기 전용, 코드·표 재구성)
- 학습/추론 불일치·행 독립·룩업 버그 **없음** (poff 재구성 최대차 0.0, clip 0%, 무이력 투수 poff 0).
- 통계적 찐빠: ① DIR·DSHs 미시즌중심화(이력 평균시즌과 corr −0.66/−0.23) — 단 DIRc≈SH(0.80)·DSHsc≈DSHs(0.97) 라 고치면 LB 가 좋아한 성분을 지움 → 미적용. ② poff 블록 이동 미통제(이력 1~100구 +0.37). ③ 카운트 항 F 행 적용(ronly 미검증).
- 진단 스크립트: `산출물/dain/src/l236x/diag_poff.py`, `tmp_L236/diag.log`.

## 2. LB 실측 (dain 제출)
| zip | 구성 | Score | Pitch | Player | 판정 |
|---|---|---|---|---|---|
| d28 | L236 + CPs(comparable prior 소이력 직교 sd .10) + ncut1000 | 70.1112 | 70.1243 | 70.0807 | Pitch +0.018 · Player **−0.478** ✗ |
| d30b | O57 + drift-anchor κ0.5 + 타자 −0.5 + dain q | 69.8941 | 70.1143 | 69.3802 | Player **−1.34** ✗ (타자 축 양방향 사망, drift Pitch 이관 1/4) |
| d36e | TS + r12 3변형 평균 + soft 재라벨 c=0.2 | 70.2844 | 70.0956 | 70.7251 | Player −0.045 ✗ (F 재라벨 계열 7번째 실패) |

교훈: (1) 홀드아웃 as-of 에 종민 `comp.txt`(2023~24 학습) 를 넣어 drift Pitch +0.087 이 누수였음(실제 +0.015). (2) as-of Player 는 팀 규칙대로 예측력 0 — 타자 −0.5 as-of +1.28 → LB 큰 음수. (3) 이력 재가공(soft/gt/proxy/lowrank)은 전부 음수 또는 무효.

## 3. 빌드만 하고 미제출 (전부 게이트 통과 · 로컬 `~/Documents/lg_aimers/`)
| zip | 구성 | 비고 |
|---|---|---|
| d26 / d27a / d27b | L236 + GEOF(+DIRc·DSHsc / +CPs) + ncut1000 / ncut 만 | GEOF as-of −0.02τ |
| d29a / d29b | O57 + drift-anchor κ0.5 / 1.0 + dain q(proj) | drift 단독 미검증 |
| d29c | L236 + CPs (ncut 없음) | |
| d30p / d30h | d29a + 대리등급 w=0 / 0.5 | as-of −0.37 / −0.16 |
| d34dir / d34hier / d34rec / d34dh | TS r12 표 교체: 등급별 λ(S900/H500) / 2단 풀링 / 2023+ 만 / 결합 (sd 0.42) | TS 표 상관 .98/.88/.62/.87 |
| d35soft(01/02/025/05/045/1.0) · d35gt23 · d35gtall | TS + pgr F 행 재측정 (소프트 재라벨 c / 기하 재채점) | d36e 결과로 계열 사망 |
| d35lr2 / d35lr4 | r12 저랭크(요인분석) | 2024 준-표본외 −0.21/−0.09 · 제출 비추 |
| d36a / d36b / d36c / d36d / d36f | 스택: soft c(.25/.5/0/.1/.45) + r12 3변형 평균 | d36c = r12 평균만(순위이동 0.6/321) |
| **d37w05** | **Player 불변 하이브리드**: t = TS 그대로 · q = 0.5·dain 5시드 + 0.5·jm 5시드 · proj anchor | 투수 평균점수 최대차 7e-15 (Player 비트 동일 검증) · 2024 인샘플 Pitch +0.043 (dain q 인샘플 과장 ~10배, LB 기대 +0.01~0.02) |
| **d38tv / d38ts** | TV_F600 / TS 목표 그대로 + q 0.5·dain+0.5·jm, **T=0.9** | 서버 out/ 에 zip 있음(로컬 미복사) · 각 기반의 Player 비트 보존 · 인샘플 Pitch +0.049 |
| (d37w05t / d37w10 / d37w00t) | 〃 T=0.9 / dain q 100% / jm q + T=0.9 — nozip | 인샘플 Pitch +0.049 / +0.065 / +0.010 |

## 4. 인프라 (`산출물/dain/src/l236x/`)
`common.py`(로더·공식 지표) · `signals.py`(factory_orth 이식 + GEOF/SITb/CNTH/CPs) · `build.py`(L236 poff 교체 + 게이트 + zip) · `build_d29.py`/`build_var.py`(O57 하이브리드·표 패치) · `build_ts.py`(TS r12 변형: base/dir/hier/rec/lowrank) · `build_tbl.py`(pgr F 행 델타 패치) · `build_stack.py`(r12 평균) · `build_hyb2.py`+`hyb2_script.py`(Player 불변 q 앙상블) · `eval24.py`/`pitchrec*.py`/`asof*.py`(진단) · `train_hold.py`+`hold24/`(≤2023 홀드아웃 번들). 체인 실행은 `chain_*.sh` (ssh 끊김 대비 setsid).
레시피 확인: O57 pgr = PGR_CTR=1·프록시 w=1 (n>500 최대차 0) · TS r12 = 시즌중심 pt, λ600, Z=n/(n+λ)·((셀−투수)−(리그셀−리그)).

## 5. 남은 결론
- Player 하방을 지키며 올릴 수 있는 확정 수단은 **q 변경(Player 수학적 불변)** 뿐: d37w05 (+ T=0.9) → Pitch +0.01~0.02 기대.
- TV_F600(카운트 항 → 전체 조건부 표 sd 1.79, use_g=0) 전략은 LB 수치 미확인. 그 위에 q 앙상블을 얹는 `build_hyb2.py NAME <TV_F600 dir> 0.5 0.9` 로 즉시 빌드 가능(d38tv 예정).
- 다른 작업 산출물 `submit_d30/d31/d33.zip` 은 dain 폴더에 있지만 이 작업 것이 아님.
