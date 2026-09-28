#!/bin/bash
# [C] 끝난 뒤 배포동일 검산 (12스레드 예산 준수)
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
cd "$(dirname "$0")"; L=../log; O=../out
while pgrep -f "bash run_diag.sh" >/dev/null; do sleep 20; done
echo "=== [A] final 0 560 --upto 2024 → script.py 경로로 2024 채점"
/usr/bin/python3 -W ignore train.py final 0 560 --upto 2024 --out submit_val/model > $L/diag_final2024.log 2>&1; tail -3 $L/diag_final2024.log
/usr/bin/python3 -W ignore verify.py submit_val/model $O/fake_test_2024.csv $O/fake_labels_2024.csv > $L/diag_verify.log 2>&1; cat $L/diag_verify.log
echo PARITY_DONE
