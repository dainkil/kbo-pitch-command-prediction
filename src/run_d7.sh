#!/bin/bash
# d7 = y2 (d4 GBDT 5시드, 바이트 동일) + EB anchor + miss 주입(far50−, chase+, 중립, ε=0.4)
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
cd "$(dirname "$0")"; L=../log; O=../out; PY="/usr/bin/python3 -W ignore"
rm -rf submit_d7 submit_val_d7; mkdir -p submit_d7; cp -r submit_d4/model submit_d7/model
echo "=== [1] d7 표·EB·주입 (GBDT 재사용)"
$PY train.py final 0,1,2,3,4 --drop b_ --combine anchor --lam 1.0 --inject miss --eps 0.4 --neutral --skip_gbdt --out submit_d7/model > $L/train_d7.log 2>&1; tail -4 $L/train_d7.log
echo "=== [2] 배포동일 검산 (--upto 2024: d4 설계 GBDT 1시드 + EB + 주입 → 2024 채점; 기대 τ ≈ .2478+.0141=.262)"
$PY train.py final 0 560 --drop b_ --combine anchor --lam 1.0 --inject miss --eps 0.4 --neutral --upto 2024 --out submit_val_d7/model > $L/parity_d7.log 2>&1; tail -2 $L/parity_d7.log
$PY verify.py submit_val_d7/model $O/fake_test_2024.csv $O/fake_labels_2024.csv 2>&1 | tail -3
echo "=== [3] zip + 평가서버 모사 + 행 독립"
bash make_submit.sh d7 $PWD/submit_d7/model 2>&1 | grep -E "zip →|saved|elapsed"
$PY verify.py submit_d7/model $O/fake_test_2024.csv 2>&1 | grep -E "①|④"
md5sum submit_d7/model/lgb_seed0.txt submit_d4/model/lgb_seed0.txt | awk '{print $1}' | uniq | wc -l | sed 's/^1$/GBDT 바이트 동일 ✓/'
echo D7_DONE
