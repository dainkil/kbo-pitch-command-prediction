#!/bin/bash
# d5 = anchor(EB α2000, GBDT noF 5시드) · d6 = d5 + loc 주입 ε0.1  → 4단 검산 → zip
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
cd "$(dirname "$0")"; L=../log; O=../out; PY="/usr/bin/python3 -W ignore"; DROP="b_,season,game_type,pgt_,p_fshare"
echo "=== [0] d4 배포동일 검산 (v3 설계, --upto 2024, script.py 경로)"
$PY train.py final 0 560 --upto 2024 --out submit_val_d4/model > $L/parity_d4_train.log 2>&1; tail -2 $L/parity_d4_train.log
$PY verify.py submit_val_d4/model $O/fake_test_2024.csv $O/fake_labels_2024.csv 2>&1 | tail -4
echo "=== [1] d5 학습: noF GBDT 5시드 (2020–2024) + EB + anchor"
$PY train.py final 0,1,2,3,4 --drop $DROP --combine anchor --lam 1.0 --out submit_d5/model > $L/train_d5.log 2>&1; tail -4 $L/train_d5.log
echo "=== [2] d6: d5 GBDT 재사용 + loc 주입 ε=0.1"
rm -rf submit_d6; mkdir -p submit_d6; cp -r submit_d5/model submit_d6/model
$PY train.py final 0,1,2,3,4 --drop $DROP --combine anchor --lam 1.0 --inject loc --eps 0.1 --skip_gbdt --out submit_d6/model > $L/train_d6.log 2>&1; tail -3 $L/train_d6.log
echo "=== [3] 배포동일 검산 (anchor 설계, --upto 2024 → 2024 채점; 기대 ≈ final_eval anchor1.0 2024: Total 67.69 / τ 0.248)"
$PY train.py final 0 --drop $DROP --combine anchor --lam 1.0 --upto 2024 --rounds 560 --out submit_val_d5/model > $L/parity_d5_train.log 2>&1
$PY verify.py submit_val_d5/model $O/fake_test_2024.csv $O/fake_labels_2024.csv 2>&1 | tail -3
$PY train.py final 0 --drop $DROP --combine anchor --lam 1.0 --inject loc --eps 0.1 --skip_gbdt --upto 2024 --out submit_val_d5/model > $L/parity_d6_train.log 2>&1
$PY verify.py submit_val_d5/model $O/fake_test_2024.csv $O/fake_labels_2024.csv 2>&1 | tail -3
echo "=== [4] zip + 평가서버 모사 (더미 5행 · 218,992행 타이밍)"
bash make_submit.sh d5 submit_d5/model 2>&1 | grep -E "zip →|saved|elapsed|PASS|FAIL|Error|error" ; bash make_submit.sh d6 submit_d6/model 2>&1 | grep -E "zip →|saved|elapsed|PASS|FAIL|Error|error"
echo "=== [5] 행 독립 검산 (zip 의 model)"
$PY verify.py submit_d5/model $O/fake_test_2024.csv 2>&1 | grep -E "①|④"; $PY verify.py submit_d6/model $O/fake_test_2024.csv 2>&1 | grep -E "①|④"
echo FINAL_DONE
