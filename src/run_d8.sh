#!/bin/bash
# d8 = y2 (d4 GBDT) + EB anchor + TabPFN 투수 잔차 주입(중립, w=0.5) — 표 조회 방식
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
cd "$(dirname "$0")"; L=../log; O=../out; PY="/usr/bin/python3 -W ignore"; PFN="../venv_pfn/bin/python -W ignore"
echo "=== [0] TabPFN 투수 z 표 (2025 제출용, 2024 검산용)"
$PFN pfn_signal.py 2025 2024 2>&1 | grep -v "Warning\|warn" | grep "U="
rm -rf submit_d8 submit_val_d8; mkdir -p submit_d8; cp -r submit_d4/model submit_d8/model
echo "=== [1] d8 표·EB·주입 (GBDT 재사용)"
$PY train.py final 0,1,2,3,4 --drop b_ --combine anchor --lam 1.0 --inject_file $O/pfn_z_2025.csv --eps 0.5 --skip_gbdt --out submit_d8/model > $L/train_d8.log 2>&1; tail -3 $L/train_d8.log
echo "=== [2] 배포동일 검산 (--upto 2024; 기대 τ ≈ .2735, Total ≈ 68.09)"
$PY train.py final 0 560 --drop b_ --combine anchor --lam 1.0 --inject_file $O/pfn_z_2024.csv --eps 0.5 --upto 2024 --out submit_val_d8/model > $L/parity_d8.log 2>&1; tail -1 $L/parity_d8.log
$PY verify.py submit_val_d8/model $O/fake_test_2024.csv $O/fake_labels_2024.csv 2>&1 | tail -3
echo "=== [3] zip + 모사 + 행 독립"
bash make_submit.sh d8 $PWD/submit_d8/model 2>&1 | grep -E "zip →|saved|elapsed"
$PY verify.py submit_d8/model $O/fake_test_2024.csv 2>&1 | grep -E "①|④"
md5sum submit_d8/model/lgb_seed0.txt submit_d4/model/lgb_seed0.txt | awk '{print $1}' | uniq | wc -l | sed 's/^1$/GBDT 바이트 동일 ✓/'
echo D8_DONE
