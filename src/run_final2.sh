#!/bin/bash
# d5 = anchor(EB α2000, GBDT noF 5시드) · d6 = d5 + loc 주입 ε0.1  → 4단 검산 → zip
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
cd "$(dirname "$0")"; L=../log; O=../out; PY="/usr/bin/python3 -W ignore"; DROP="b_,season,game_type,pgt_,p_fshare"
echo "=== [2] (완료) d6"
echo "=== [3] 배포동일 검산 (anchor 설계, --upto 2024 → 2024 채점; 기대 ≈ final_eval anchor1.0 2024: Total 67.69 / τ 0.248)"
$PY train.py final 0 --drop $DROP --combine anchor --lam 1.0 560 --upto 2024 --out submit_val_d5/model > $L/parity_d5_train.log 2>&1
$PY verify.py submit_val_d5/model $O/fake_test_2024.csv $O/fake_labels_2024.csv 2>&1 | tail -3
$PY train.py final 0 --drop $DROP --combine anchor --lam 1.0 --inject loc --eps 0.1 --skip_gbdt --upto 2024 --out submit_val_d5/model > $L/parity_d6_train.log 2>&1
$PY verify.py submit_val_d5/model $O/fake_test_2024.csv $O/fake_labels_2024.csv 2>&1 | tail -3
echo "=== [4] zip + 평가서버 모사 (더미 5행 · 218,992행 타이밍)"
bash make_submit.sh d5 submit_d5/model 2>&1 | grep -vE "^model/|^script|^requirements|^TEST_|^TRAIN_|^row_id"; bash make_submit.sh d6 submit_d6/model 2>&1 | grep -vE "^model/|^script|^requirements|^TEST_|^TRAIN_|^row_id"
echo "=== [5] 행 독립 검산 (zip 의 model)"
$PY verify.py submit_d5/model $O/fake_test_2024.csv 2>&1 | grep -E "①|④"; $PY verify.py submit_d6/model $O/fake_test_2024.csv 2>&1 | grep -E "①|④"
echo FINAL_DONE
