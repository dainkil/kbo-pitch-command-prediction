#!/bin/bash
cd ~/workspace/산출물/dain/src/l236x
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6
/usr/bin/python3 -W ignore build_hyb2.py d38tv ~/workspace/산출물/종민/submit_TV_F600 0.5 0.9 > log/build_d38tv.log 2>&1
/usr/bin/python3 -W ignore build_hyb2.py d38ts ~/workspace/산출물/종민/submit_TS_P600G10 0.5 0.9 > log/build_d38ts.log 2>&1
echo DONE > log/chain_hyb2.done
