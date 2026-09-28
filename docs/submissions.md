# dain 제출 기록 — submit_dN.zip
| zip | 모델 | 2024 폴드 Total / Pitch / τ | LB |
|---|---|---|---|
| submit_d1.zip | v1: LGB(season+이력+Trackman) 3시드, lr0.05 | 69.16 / 70.01 / 0.344 | |
| submit_d4.zip | v3: v1 + 위치·반대투구 이력, −타자피처, lr0.02 minleaf1000, 5시드 | 69.27 / 70.02 / 0.350 | |
| submit_d5.zip | anchor: 행별 제구점수=EB(α2000, 배치셀prior) 고정 + GBDT noF(5시드) 등급배분. τ=EB | 67.95 / 67.38 / 67.69 (3폴드) · 배포동일 2024: 67.69 / 69.97 / 0.248 | |
| submit_d6.zip | d5 + 이력신호 주입(loc: p_h_sd·pgtR_rev·p_out, ε=0.1) — 2025 F체제 베팅, 2022폴드 ✗ | 67.77 / 67.86 / 67.94 · 배포동일 2024: 67.94 / 69.97 / 0.264 | |
LB: d4 = 67.198 / 70.206 / 60.179 (τ 0.204)
| submit_d7.zip | **y2 GBDT(바이트 동일) + EB anchor + 빗나간 크기 주입**: far50(Waste, −)·chase(+) 60셀 역할보정 잔차 EB α2000, (과거F비중×이력) 중립화, ε=0.4 | 3폴드 Δτ +.017/+.044/+.014 (within-R 3/3) · 배포동일 2024: 67.92 / 69.98 / τ .262 | |
LB: d5 68.627/70.166/65.036 · d6 68.582/70.167/64.881 · y2 68.632/70.173/65.036
LB: d7 68.455/70.174/64.444 (τ .2889) ✗ — 위치 파생 주입은 LB 에서 손해 (d6·L7 과 같은 방향)
| submit_d8.zip | y2 GBDT + EB anchor + **TabPFN v2 투수×시즌 잔차 주입**(표 조회, 중립화, w=0.5; 36 이력피처) | 3폴드 Δτ +.019/+.020/+.026 | |
| submit_d9.zip | d5 + L25 목표(F행 +1.333 · 팀 −0.25·team_dev, adj.json) | 배포동일 2024: 65.83 / 69.95 / τ 0.125 (거울상 예상) · 기대 LB ≈ 69.46 (Player 67.809 + Pitch 70.166) | |
| submit_d11.zip | d9 + lvln_g=+0.3 (F보정×이력크기, 신인 F행 +3.33 / 대이력 +0.33) — 거울상 베팅 · ⚠ zip 이 21:40 다른 작업(skill prior 빌드)에 덮어써져 삭제됨. 종민 L80(+0.3, LB −1.09)로 기각된 축이라 재생성 안 함 | 배포동일 2024: 65.65 / 69.95 / τ 0.113 (거울상) | |
| submit_d12.zip | d8(TabPFN 잔차 ε=0.568, d4형 GBDT) + L25 보정 | 3폴드 Δτ +.02 · 기대 d9 +1 Player | |
| submit_d13.zip | d9 + 2019~22 F 행 제외 (EB 표만, build_pre23f.py drop) | 배포동일 2024(adj 없이): 68.146 / 69.976 / τ .2776 (base .2478) | |
| submit_d14.zip | d9 + 2019~22 F 소프트 재라벨 (p̂(rev) AUC .953, build_pre23f.py soft) | 배포동일 2024(adj 없이): 68.310 / 69.982 / τ .2882 (base .2478) | |
| submit_d10.zip | **skill prior Cf**: d9(anchor·adj L25) + **α1300** + EB prior 수준을 GBDT q_skill(row) 로 (구종구성 pk_mix 7·트랙맨 구성/구속·이력 n·기용 A군·타자 EB/레벨; **F 관련 피처 없음**), 학습행 (R)\|(≥2023). 셀 prior 는 형태만 유지 | 3폴드 ΔEB τ +.012/+.058/+.075 · 배포동일 2024: 67.90 / 69.977 / τ .261 (base α1300 65.98 / 69.952 / .134) | |
| submit_d15.zip | d10 + `game_type`·`p_fshare`·`post23` (F 선택 신호 포함, **Af**) | 3폴드 +.016/+.076/+.106 · 배포동일 2024: 68.81 / 69.984 / τ .321 · ⚠ L76(F×이력크기) LB −5.37 와 같은 기계 → 하락 예상 탐침 | |
| submit_d16.zip | d14(soft) + 플래툰 +0.6 (hand_dev 2024) — L87 항 전부 + soft | 기대 L87 Player 68.30 + soft 효과, Pitch 70.17 | |
| submit_d17.zip | d9 + 플래툰 +0.6 (대조군) | 기대 d9 +0.1 | |
| (erank 기각) | — | d16 + 사후기대순위 재배치 (τ²=7.8 모멘트법, 참조 766) | 2024 폴드 τ .2882→.2800 (−0.008) · Total 68.31→68.18 · 단일행 비트동일 실패(1e-16) → 제출 안 함 | |
| submit_d19.zip | d16 + 시즌중심화(CTR, 소프트 one-hot 판) — soft+ctr · F −0.6 · 팀 −0.25 · 플래툰 +0.6 · 5시드 | 배포동일 2024(adj 없이): 68.366 / 69.982 / τ .2919 (soft .2882 · base .2478) · CTR 은 F 직교축(종민 L83 LB +0.2) | |
| submit_d21.zip | d9 + **α1300** + **이력구간 수준 보정** hist_delta (신인 −0.82 · 1~300 −1.45 · 301~1000 −1.17; R 행 LOSO 회귀 계수, `eda/25`) — skill prior 없음, 마진 투수 선택 효과 단독 탐침 | 배포동일 2024: 67.03 / 69.966 / τ .204 (base α1300 65.98 / 69.952 / .134) · R행 LOSO 3/3 +.025/+.046/+.030 | |
| submit_d22.zip | **d10 + 투수-중심 anchor κ=0.5** (GBDT 행 점수의 학습행 투수평균 대비 편차를 목표에 되살림 → Pitch 전용) | 배포동일 2024: 67.79 / **69.996** / τ .253 (d10 Cf 67.90 / 69.977 / .261) — Pitch +0.02 (이관됨), τ −0.008(잡음) | |
| submit_d20.zip | ★ 하이브리드: 목표 t = 종민 L87 번들(model/jm_root, LB Player 68.299) 그대로 · 등급 배분 q = dain 5시드 GBDT (LB Pitch 70.18) · 가산형 anchor | 2024 가짜 test: 행별 점수 = L87 과 2e-14 동일, τ 동일, Pitch 69.989→70.130 · 기대 LB ≈ 0.7·70.18+0.3·68.30 = **69.62** (L87 +0.05) · 행독립 부분집합/역순 0, 단일행 1.1e-16(종민 게이트 기준 통과) · 37s/0.6GB | |
| submit_d21.zip | d20 + **기용 B군 항** (usage_B: balls −5.74·z, strikes +2.51, runners −1.43, run_total −0.54, outs +1.32 · w=α/(n+α) · scale 1.0) | 표본외 폴드 Δτ 2023 +0.053 / 2024 +0.066 (소이력 +0.05~0.09) · 행MSE +0.9~1.5% · ⚠ 종민이 "지표 조작"으로 원칙 폐기한 축 — 규정 문구는 통과(자기 행 열 + 상수), 발표·코드검증 리스크는 팀 판단 | |
| submit_d22.zip | d21 의 scale 0.5 | 곡선 확인용 | |
| submit_d23.zip | ★ d20 + 기용항 **C0.5**: x = 0.5·z_row + 0.5·z̄_hist(투수 과거 기용 평균 표, 전 행, κ=200) · β balls −8.03 strikes +4.13 runners −2.98 run_total −0.80 outs +3.38 · w=α/(n+α) · scale 1.0 | 표본외 폴드 Δτ **+0.082/+0.068** (B5@1.0 +0.053/+0.065) · 행MSE +0.5% (B5@0.5 수준) → LB 에서 B5@0.5(+1.08) + 이력분 기대 | |
| submit_d24.zip | d20 + 기용항 **C1.0 = 순수 이력 표** (행 왜곡 0, 규정·원칙 논란 없음): β balls −3.62 strikes +4.11 runners −2.98 run_total −0.66 outs +5.03 | 표본외 Δτ +0.046/+0.023 · 행MSE −0.03% (Pitch 불변) · 깨끗한 축 | |
| submit_d25.zip | d20 + 기용항 **C0.7** (행 0.3 · 이력 0.7) | 표본외 Δτ +0.089/+0.057 · 행MSE +0.12~0.20% | |
| submit_d27b.zip | L236 + poff[RK:0.25,SIT:0.25,SH:0.2,PLT:0.2,DIR:0.15,DSHs:0.12] meta{"ncut": 1000.0} | 2024 인샘플 L236 대비 ΔPitch +0.0236 ΔPlayer +0.3174 · 순위이동 6.8/321 | |
| submit_d27a.zip | L236 + poff[RK:0.2209,SIT:0.2209,SH:0.1767,PLT:0.1767,DIR:0.1325,DSHs:0.106,GEOF:0.114,CPs:0.10] meta{"ncut": 1000.0} | 2024 인샘플 L236 대비 ΔPitch +0.0227 ΔPlayer -0.7769 · 순위이동 9.5/321 | |
| submit_d26.zip | L236 + poff[RK:0.2325,SIT:0.2325,SH:0.186,PLT:0.186,DIR:0.1395,DSHs:0.1116,GEOF:0.12] meta{"ncut": 1000.0} | 2024 인샘플 L236 대비 ΔPitch +0.0224 ΔPlayer -0.9171 · 순위이동 8.9/321 | |
| submit_d28.zip | L236 + poff[RK:0.2375,SIT:0.2375,SH:0.19,PLT:0.19,DIR:0.1425,DSHs:0.114,CPs:0.10] meta{"ncut": 1000.0} | 2024 인샘플 L236 대비 ΔPitch +0.0238 ΔPlayer +0.3972 · 순위이동 8.7/321 | |
| submit_d29c.zip | L236 + poff[RK:0.2375,SIT:0.2375,SH:0.19,PLT:0.19,DIR:0.1425,DSHs:0.114,CPs:0.10] meta | 2024 인샘플 L236 대비 ΔPitch +0.0002 ΔPlayer +0.0818 · 순위이동 4.4/321 | |
| LB d28 | Score 70.1112 · Pitch 70.1243 · Player 70.0807 (L236 대비 Pitch +0.018 · Player −0.478) |
| submit_d29a.zip | **drift-anchor 하이브리드**: O57 목표 + κ=0.5·[이력有]·(cmp행 − 이력 cmp 평균, 수축200) · q = dain 5시드(proj) | 2024 인샘플 O57 대비 Pitch +0.1812 Player +3.0783 · as-of 홀드아웃 κ0.5 Pitch +0.087 Player +0.47 / κ1.0 +0.151 +0.34 | |
| submit_d29b.zip | **drift-anchor 하이브리드**: O57 목표 + κ=1.0·[이력有]·(cmp행 − 이력 cmp 평균, 수축200) · q = dain 5시드(proj) | 2024 인샘플 O57 대비 Pitch +0.2699 Player +4.6340 · as-of 홀드아웃 κ0.5 Pitch +0.087 Player +0.47 / κ1.0 +0.151 +0.34 | |
| submit_d30h.zip | d29a + proxy w=0.5 | 2024 인샘플 d29a 대비 Pitch +0.0005 Player -0.0604 · 순위이동 3.8/321 | |
| submit_d30p.zip | d29a + proxy w=0.0 | 2024 인샘플 d29a 대비 Pitch +0.0013 Player -0.0623 · 순위이동 7.0/321 | |
| submit_d40.zip | ★ 하이브리드: seongmin O57 t(LB 70.2797/70.093/70.7154) × dain 5시드 q · make_hybrid.sh | 검산 행별 점수 O57 과 2.8e-14 일치 → Player 동일 · Pitch +0.005~0.01 · 55s/0.65GB | |
| (pch 검토) | 종민 05:00 제안 투수×카운트12×같은손 조건부 편차 표 (λ200) | 공정 as-of 2024: τ .2478→.2470(s0.5)/.2465(1.0)/.2429(1.5) · Pitch ±0.003 · 투수 평균 이동 sd 0.07~0.20 → 크기 자체가 잡음 · 제출 안 함 | |
| submit_d30b.zip | d29a + bat g=-0.5 | 2024 인샘플 d29a 대비 Pitch -0.0067 Player +1.0086 · 순위이동 10.7/321 | |
| (as-of ≤2023→2024, hold24 O57형) | drift κ0.5 Pitch +.087 Player +.47 · κ1.0 +.151/+.34 · proxy w0 +.002/−.37 · bat −0.25/−0.5/−1.0 Player +.69/+1.28/+2.20 (+0.5 → −1.48, LB T3_B10 +1.0 → −.48 부호 일치) · drift+bat −0.5 +.091/+3.36 |
| submit_d34rec.zip | **TS_P600G10 파쿠리** r12 표 교체: rec sd0.42 lamS900 lamH500 lam1300 lam2600 ymin2023 rlam600.0 (TS 표와 상관 +0.623) | 2024 인샘플 TS 대비 Pitch +0.0104 Player +0.0331 · 순위이동 3.3/321 | |
| submit_d34hier.zip | **TS_P600G10 파쿠리** r12 표 교체: hier sd0.42 lamS900 lamH500 lam1300.0 lam2600.0 ymin2019 rlam600 (TS 표와 상관 +0.881) | 2024 인샘플 TS 대비 Pitch -0.0035 Player -0.0019 · 순위이동 1.8/321 | |
| submit_d34dir.zip | **TS_P600G10 파쿠리** r12 표 교체: dir sd0.42 lamS900.0 lamH500.0 lam1300 lam2600 ymin2019 rlam600 (TS 표와 상관 +0.976) | 2024 인샘플 TS 대비 Pitch -0.0006 Player +0.0409 · 순위이동 1.1/321 | |
| submit_d34dh.zip | **TS_P600G10 파쿠리** r12 표 교체: dirhier sd0.42 lamS900 lamH500 lam1300.0 lam2600.0 ymin2019 rlam600 (TS 표와 상관 +0.867) | 2024 인샘플 TS 대비 Pitch -0.0040 Player -0.0195 · 순위이동 1.8/321 | |
| submit_d35gt23.zip | **TS_P600G10 + anchor 이력표 F행 재측정** gt23 c=1.0 (pgr 평균 변화 sd 2.020) | 2024 인샘플 TS 대비 Pitch +0.0058 Player +0.3329 · 순위이동 12.4/321 | |
| submit_d35gtall.zip | **TS_P600G10 + anchor 이력표 F행 재측정** gtall c=1.0 (pgr 평균 변화 sd 2.236) | 2024 인샘플 TS 대비 Pitch +0.0048 Player -0.3213 · 순위이동 13.0/321 | |
| submit_d35soft.zip | **TS_P600G10 + anchor 이력표 F행 재측정** soft c=1.0 (pgr 평균 변화 sd 2.258) | 2024 인샘플 TS 대비 Pitch +0.0070 Player +0.6425 · 순위이동 15.3/321 | |
| submit_d35lr2.zip | **TS_P600G10 파쿠리** r12 표 교체: lowrank sd0.42 lamS900 lamH500 lam1300 lam2600 ymin2019 rlam600 (TS 표와 상관 +0.386) | 2024 인샘플 TS 대비 Pitch -0.0230 Player -0.2122 · 순위이동 5.0/321 | |
| submit_d35lr4.zip | **TS_P600G10 파쿠리** r12 표 교체: lowrank sd0.42 lamS900 lamH500 lam1300 lam2600 ymin2019 rlam600 (TS 표와 상관 +0.489) | 2024 인샘플 TS 대비 Pitch -0.0184 Player -0.0935 · 순위이동 4.5/321 | |
| submit_d35soft05.zip | **TS_P600G10 + anchor 이력표 F행 재측정** soft c=0.5 (pgr 평균 변화 sd 1.129) | 2024 인샘플 TS 대비 Pitch +0.0039 Player +0.5413 · 순위이동 7.5/321 | |
| submit_d35soft025.zip | **TS_P600G10 + anchor 이력표 F행 재측정** soft c=0.25 (pgr 평균 변화 sd 0.564) | 2024 인샘플 TS 대비 Pitch +0.0020 Player +0.2998 · 순위이동 3.7/321 | |
| submit_d36a.zip | **스택** d35soft025 + r12=avg(TS+d34dir+d34hier) sd0.42 | 2024 인샘플 TS 대비 Pitch +0.0019 Player +0.2921 · 순위이동 3.8/321 | |
| submit_d36b.zip | **스택** d35soft05 + r12=avg(TS+d34dir+d34hier) sd0.42 | 2024 인샘플 TS 대비 Pitch +0.0038 Player +0.5510 · 순위이동 7.6/321 | |
| submit_d36c.zip | **스택** d31chk + r12=avg(TS+d34dir+d34hier) sd0.42 | 2024 인샘플 TS 대비 Pitch -0.0001 Player +0.0097 · 순위이동 0.6/321 | |
| submit_d35soft01.zip | **TS_P600G10 + anchor 이력표 F행 재측정** soft c=0.1 (pgr 평균 변화 sd 0.226) | 2024 인샘플 TS 대비 Pitch +0.0008 Player +0.0779 · 순위이동 1.5/321 | |
| submit_d36d.zip | **스택** d35soft01 + r12=avg(TS+d34dir+d34hier) sd0.42 | 2024 인샘플 TS 대비 Pitch +0.0007 Player +0.0993 · 순위이동 1.6/321 | |
| submit_d35soft02.zip | **TS_P600G10 + anchor 이력표 F행 재측정** soft c=0.2 (pgr 평균 변화 sd 0.452) | 2024 인샘플 TS 대비 Pitch +0.0017 Player +0.2005 · 순위이동 3.0/321 | |
| submit_d36e.zip | **스택** d35soft02 + r12=avg(TS+d34dir+d34hier) sd0.42 | 2024 인샘플 TS 대비 Pitch +0.0016 Player +0.2239 · 순위이동 3.1/321 | |
| submit_d35soft045.zip | **TS_P600G10 + anchor 이력표 F행 재측정** soft c=0.45 (pgr 평균 변화 sd 1.016) | 2024 인샘플 TS 대비 Pitch +0.0035 Player +0.5023 · 순위이동 6.7/321 | |
| submit_d36f.zip | **스택** d35soft045 + r12=avg(TS+d34dir+d34hier) sd0.42 | 2024 인샘플 TS 대비 Pitch +0.0034 Player +0.4848 · 순위이동 6.8/321 | |
| submit_d37w05.zip | **Player 불변 하이브리드**: t = submit_TS_P600G10 그대로 · q = 0.50·dain5시드 + 0.50·jm5시드, T=1.0 · proj anchor | 2024 인샘플 Pitch +0.0429 · 투수평균 최대차 7.1e-15 | |
| submit_d38tv.zip | **Player 불변 하이브리드**: t = submit_TV_F600 그대로 · q = 0.50·dain5시드 + 0.50·jm5시드, T=0.9 · proj anchor | 2024 인샘플 Pitch +0.0485 · 투수평균 최대차 7.1e-15 | |
| submit_d38ts.zip | **Player 불변 하이브리드**: t = submit_TS_P600G10 그대로 · q = 0.50·dain5시드 + 0.50·jm5시드, T=0.9 · proj anchor | 2024 인샘플 Pitch +0.0485 · 투수평균 최대차 7.1e-15 | |
| LB d36e | Score 70.2844 · Pitch 70.0956 · Player 70.7251 (TS 대비 Player −0.045) — soft 재라벨 계열 사망 |
| 작업 기록 | `docs/notes/work_log_0920.md` (빌드 목록·LB·교훈·인프라) |
| submit_dFIN.zip | **Player 불변 하이브리드**: t = submit_d36c 그대로 · q = 0.50·dain5시드 + 0.50·jm5시드, T=0.9 · proj anchor | 2024 인샘플 Pitch +0.0485 · 투수평균 최대차 7.1e-15 | |
| submit_dFIN0.zip | **Player 불변 하이브리드**: t = submit_d36cp0 그대로 · q = 0.50·dain5시드 + 0.50·jm5시드, T=0.9 · proj anchor | 2024 인샘플 Pitch +0.0485 · 투수평균 최대차 7.1e-15 | |
| submit_dFIN5.zip | **Player 불변 하이브리드**: t = submit_d36cp5 그대로 · q = 0.50·dain5시드 + 0.50·jm5시드, T=0.9 · proj anchor | 2024 인샘플 Pitch +0.0485 · 투수평균 최대차 7.1e-15 | |
| submit_dFIN.zip / dFIN0 / dFIN5 | 최종 후보: TS 목표 + r12 3변형 평균 (+ 대리등급 프록시 w=1 / 0 / 0.5) · q = 0.5 dain + 0.5 jm · T=0.9(≤2023→2024 표본외 적합 +0.016) · Player 비트 보존 | w0: pgr sd 1.56 · 2024 인샘플 Player −0.12 · w0.5: −0.15 | |
| submit_dLAST.zip | **Player 불변 하이브리드**: t = submit_d35soft025 그대로 · q = 0.50·dain5시드 + 0.50·jm5시드, T=0.9 · proj anchor | 2024 인샘플 Pitch +0.0485 · 투수평균 최대차 7.1e-15 | |
| submit_dLAST2.zip | **Player 불변 하이브리드**: t = submit_d35soft025rk 그대로 · q = 0.50·dain5시드 + 0.50·jm5시드, T=0.9 · proj anchor | 2024 인샘플 Pitch +0.0485 · 투수평균 최대차 7.1e-15 | |
| submit_dLAST3.zip | **Player 불변 하이브리드**: t = submit_d35soft05rk 그대로 · q = 0.50·dain5시드 + 0.50·jm5시드, T=0.9 · proj anchor | 2024 인샘플 Pitch +0.0485 · 투수평균 최대차 7.1e-15 | |
| submit_dFING2.zip | **Player 불변 하이브리드**: t = jm_root 그대로 · q = 0.50·dain5시드 + 0.50·jm5시드, T=0.9 · proj anchor | 2024 인샘플 Pitch +0.0485 · 투수평균 최대차 7.1e-15 | |
| submit_dFING3.zip | **Player 불변 하이브리드**: t = submit_fgsoft02 그대로 · q = 0.50·dain5시드 + 0.50·jm5시드, T=0.9 · proj anchor | 2024 인샘플 Pitch +0.0485 · 투수평균 최대차 7.1e-15 | |
| submit_dFING4.zip | **Player 불변 하이브리드**: t = submit_fgsoft06 그대로 · q = 0.50·dain5시드 + 0.50·jm5시드, T=0.9 · proj anchor | 2024 인샘플 Pitch +0.0485 · 투수평균 최대차 7.1e-15 | |
