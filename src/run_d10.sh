#!/bin/bash
# d10/d11 = d9 (anchor · adj L25 목표) + EB α=1300 + skill prior → 4단 검산 → zip
# 태그 = <variant><flags>:  A/B/C = 피처 세트 (C = 체제노출 피처 제외) · r = --resid (셀 prior 잔차) · f = --rfilter ((R)|(≥2023) 행만)
# 사용: bash run_d10.sh parity "Ar Cf"      (2024 홀드아웃: submit_val_d9 GBDT 재사용 + α1300 + skill → 채점)
#       bash run_d10.sh build  "Ar Cf" 10   (배포: skill 2020..2024 학습 → submit_d10.zip, submit_d11.zip …)
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
cd "$(dirname "$0")"; L=../log; O=../out; PY="/usr/bin/python3 -W ignore"; DROP="b_,season,game_type,pgt_,p_fshare"
STEP=${1:-parity}; TAGS=${2:-"Ar"}; N0=${3:-10}
flags() { local t=$1; local f="--variant ${t:0:1}"; [[ $t == *r* ]] && f="$f --resid"; [[ $t == *f* ]] && f="$f --rfilter"; [[ $t == *g* ]] && f="$f --gate 50"; echo $f; }

if [ "$STEP" = "parity" ]; then
  echo "=== [P1] skill prior 2024 홀드아웃 학습 (표 <2024)"
  for t in $TAGS; do [ -f $O/skill_${t}_2024/skill_meta.json ] || $PY skill_prior.py $(flags $t) --upto 2024 --out $O/skill_${t}_2024 --eval > $L/skill_${t}_2024.log 2>&1 & done; wait
  echo "=== [P2] 배포동일 검산: submit_val_d9 GBDT(<2024) 재사용 + α1300 [+ skill] → 2024 가짜 test 채점"
  for V in base $TAGS; do
    rm -rf submit_val_d10$V; mkdir -p submit_val_d10$V; cp -r submit_val_d9/model submit_val_d10$V/model
    SK=""; [ "$V" != "base" ] && SK="--skill_dir $O/skill_${V}_2024"
    $PY train.py final 0 --drop $DROP --combine anchor --lam 1.0 --eb_alpha 1300 --skip_gbdt --upto 2024 --out submit_val_d10$V/model $SK > $L/parity_d10$V.log 2>&1
    echo "--- d10$V"; $PY verify.py submit_val_d10$V/model $O/fake_test_2024.csv $O/fake_labels_2024.csv 2>&1 | grep -E "①|②|④|τ "
  done
  echo PARITY_DONE
fi

if [ "$STEP" = "build" ]; then
  echo "=== [B1] skill prior 배포용 (2020..2024 학습, 표 <2025)"
  for t in $TAGS; do $PY skill_prior.py $(flags $t) --upto 2025 --out $O/skill_${t}_2025 > $L/skill_${t}_2025.log 2>&1 & done; wait
  for t in $TAGS; do tail -1 $L/skill_${t}_2025.log; done
  echo "=== [B2] 배포 모델: d9 GBDT(5시드) 재사용 + α1300 + skill → zip + 평가서버 모사 + 행 독립"
  N=$N0
  for t in $TAGS; do
    rm -rf submit_d$N; mkdir -p submit_d$N; cp -r submit_d9/model submit_d$N/model
    $PY train.py final 0,1,2,3,4 --drop $DROP --combine anchor --lam 1.0 --eb_alpha 1300 --skip_gbdt --upto 2025 --out submit_d$N/model --skill_dir $O/skill_${t}_2025 > $L/train_d$N.log 2>&1
    tail -2 $L/train_d$N.log
    bash make_submit.sh d$N $PWD/submit_d$N/model 2>&1 | grep -E "zip →|saved|elapsed"
    $PY verify.py submit_d$N/model $O/fake_test_2024.csv 2>&1 | grep -E "①|④"
    md5sum submit_d$N/model/lgb_seed0.txt submit_d9/model/lgb_seed0.txt | awk '{print $1}' | uniq | wc -l | sed 's/^1$/GBDT 바이트 동일 ✓/'
    echo "d$N = $t"; N=$((N+1))
  done
  echo BUILD_DONE
fi
