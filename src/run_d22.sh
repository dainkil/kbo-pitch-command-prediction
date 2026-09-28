#!/bin/bash
# d22 = d10 (Cf skill prior · α1300 · adj L25) + 투수-중심 anchor κ=0.5 (Pitch +0.03 폴드)
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
cd "$(dirname "$0")"; L=../log; O=../out; PY="/usr/bin/python3 -W ignore"; DROP="b_,season,game_type,pgt_,p_fshare"
echo "=== [1] 2024 배포동일: Cf + center 0.5"
D=submit_val_d22; rm -rf $D; mkdir -p $D; cp -r submit_val_d9/model $D/model
$PY train.py final 0 --drop $DROP --combine anchor --lam 1.0 --eb_alpha 1300 --skip_gbdt --upto 2024 --out $D/model --skill_dir $O/skill_Cf_2024 --center_k 0.5 > $L/parity_d22.log 2>&1
$PY verify.py $D/model $O/fake_test_2024.csv $O/fake_labels_2024.csv 2>&1 | grep -E "①|②|④|τ "
echo "=== [2] 배포"
rm -rf submit_d22; mkdir -p submit_d22; cp -r submit_d9/model submit_d22/model
$PY train.py final 0,1,2,3,4 --drop $DROP --combine anchor --lam 1.0 --eb_alpha 1300 --skip_gbdt --upto 2025 --out submit_d22/model --skill_dir $O/skill_Cf_2025 --center_k 0.5 > $L/train_d22.log 2>&1; grep -E "s_hat|skill prior|final done" $L/train_d22.log
bash make_submit.sh d22 $PWD/submit_d22/model 2>&1 | grep -E "zip →|saved|elapsed"
$PY verify.py submit_d22/model $O/fake_test_2024.csv 2>&1 | grep -E "①|④"
md5sum submit_d22/model/lgb_seed0.txt submit_d9/model/lgb_seed0.txt | awk '{print $1}' | uniq | wc -l | sed 's/^1$/GBDT 바이트 동일 ✓/'
echo D22_DONE
