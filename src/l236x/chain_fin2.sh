#!/bin/bash
cd ~/workspace/산출물/dain/src/l236x
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=10
/usr/bin/python3 -W ignore build_pgrw.py d36c d36cp0 0.0 > log/build_d36cp0.log 2>&1
/usr/bin/python3 -W ignore build_hyb2.py dFIN0 ~/workspace/산출물/dain/src/submit_d36cp0 0.5 0.9 > log/build_dFIN0.log 2>&1
/usr/bin/python3 -W ignore build_pgrw.py d36c d36cp5 0.5 > log/build_d36cp5.log 2>&1
/usr/bin/python3 -W ignore build_hyb2.py dFIN5 ~/workspace/산출물/dain/src/submit_d36cp5 0.5 0.9 > log/build_dFIN5.log 2>&1
echo DONE > log/chain_fin2.done
