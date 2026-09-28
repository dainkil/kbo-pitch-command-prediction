#!/bin/bash
# 투수-중심 anchor κ 폴드 테스트 (2024, GBDT<2024 1시드, α1300, adj L25): κ=0.5 / 1.0 → Pitch(판정) · Player(참고)
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
cd "$(dirname "$0")"; L=../log; O=../out; PY="/usr/bin/python3 -W ignore"; DROP="b_,season,game_type,pgt_,p_fshare"
for K in 0.5 1.0; do
  D=submit_val_c$K; rm -rf $D; mkdir -p $D; cp -r submit_val_d9/model $D/model
  $PY train.py final 0 --drop $DROP --combine anchor --lam 1.0 --eb_alpha 1300 --skip_gbdt --upto 2024 --out $D/model --center_k $K > $L/parity_c$K.log 2>&1; grep "s_hat" $L/parity_c$K.log
  echo "--- κ=$K"; $PY verify.py $D/model $O/fake_test_2024.csv $O/fake_labels_2024.csv 2>&1 | grep -E "①|②|τ 전체"
done
echo CENTER_DONE
