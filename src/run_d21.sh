#!/bin/bash
# d21 = d9 (anchor · adj L25) + α1300 + 이력구간 수준 보정 hist_delta (R 행 LOSO 회귀: 신인 −0.82 · 1~300 −1.45 · 301~1000 −1.17). skill prior 없음.
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
cd "$(dirname "$0")"; L=../log; O=../out; PY="/usr/bin/python3 -W ignore"; DROP="b_,season,game_type,pgt_,p_fshare"
HD='{"d0": -0.82, "d1": -1.45, "d2": -1.17}'
addhd() { /usr/bin/python3 - "$1" "$HD" <<'PY'
import json, sys; p = sys.argv[1] + "/adj.json"; A = json.load(open(p, encoding="utf-8")); A["hist_delta"] = json.loads(sys.argv[2]); json.dump(A, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("adj.json + hist_delta", A["hist_delta"])
PY
}
echo "=== [1] 2024 배포동일: submit_val_d10base (α1300, GBDT<2024) 에 hist_delta 만 추가"
rm -rf submit_val_d21; cp -r submit_val_d10base submit_val_d21; addhd submit_val_d21/model
$PY verify.py submit_val_d21/model $O/fake_test_2024.csv $O/fake_labels_2024.csv 2>&1 | grep -E "①|②|④|τ "
echo "=== [2] 배포: d9 GBDT 5시드 + α1300 + hist_delta"
rm -rf submit_d21; mkdir -p submit_d21; cp -r submit_d9/model submit_d21/model
$PY train.py final 0,1,2,3,4 --drop $DROP --combine anchor --lam 1.0 --eb_alpha 1300 --skip_gbdt --upto 2025 --out submit_d21/model > $L/train_d21.log 2>&1; tail -1 $L/train_d21.log
addhd submit_d21/model
bash make_submit.sh d21 $PWD/submit_d21/model 2>&1 | grep -E "zip →|saved|elapsed"
$PY verify.py submit_d21/model $O/fake_test_2024.csv 2>&1 | grep -E "①|④"
echo D21_DONE
