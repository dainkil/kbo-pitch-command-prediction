#!/bin/bash
cd ~/workspace/산출물/dain/src/l236x
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8
/usr/bin/python3 -W ignore build_hyb2.py dFIN ~/workspace/산출물/dain/src/submit_d36c 0.5 0.9 > log/build_dFIN.log 2>&1
echo DONE > log/chain_fin.done
