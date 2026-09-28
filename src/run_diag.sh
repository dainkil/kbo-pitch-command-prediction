#!/bin/bash
# 진단 묶음 (12스레드 순차): 배포동일+행독립 검산 → 3폴드 × 3설계
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
cd "$(dirname "$0")"; L=../log; O=../out
/usr/bin/python3 -W ignore - <<'PY'
import pandas as pd
tr = pd.read_csv("/home/ubuntu/workspace/data/train.csv", dtype={"row_id": str}); oc = pd.read_csv("/home/ubuntu/workspace/data/train_outcomes.csv", dtype={"row_id": str}, usecols=["row_id", "control_grade"])
d = tr.merge(oc, on="row_id"); d = d[(d.season == 2024) & d.control_grade.notna()]
d.drop(columns=["control_grade"]).to_csv("../out/fake_test_2024.csv", index=False); d[["row_id", "control_grade"]].to_csv("../out/fake_labels_2024.csv", index=False); print("fake test 2024:", len(d), "rows, 29 cols")
PY
echo "=== [A] final --upto 2024 (학습 2020–2023) → script.py 경로로 2024 채점"
/usr/bin/python3 -W ignore train.py final 0 --upto 2024 --out submit_val/model --rounds 560 > $L/diag_final2024.log 2>&1; tail -3 $L/diag_final2024.log
/usr/bin/python3 -W ignore verify.py submit_val/model $O/fake_test_2024.csv $O/fake_labels_2024.csv > $L/diag_verify.log 2>&1; cat $L/diag_verify.log
echo "=== [B] 제출본(d4) 행 독립 검산 (더미 test 대신 fake 2024, season 그대로)"
/usr/bin/python3 -W ignore verify.py submit/model $O/fake_test_2024.csv > $L/diag_verify_d4.log 2>&1; cat $L/diag_verify_d4.log
echo "=== [C] 3폴드 × 3설계 (seed 0)"
for Y in 2022 2023 2024; do
  /usr/bin/python3 -W ignore train.py val 0 --year $Y --tag v3 > $L/fold_v3_$Y.log 2>&1
  /usr/bin/python3 -W ignore train.py val 0 --year $Y --tag noseason --drop b_,season > $L/fold_noseason_$Y.log 2>&1
  /usr/bin/python3 -W ignore train.py val 0 --year $Y --tag noF --drop b_,season,game_type,pgt_,p_fshare > $L/fold_noF_$Y.log 2>&1
  for t in v3 noseason noF; do echo "[$Y $t] $(grep '★ 앙상블' $L/fold_${t}_$Y.log)"; done
done
echo DIAG_DONE
