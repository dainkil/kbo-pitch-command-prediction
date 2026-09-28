#!/bin/bash
cd ~/workspace/산출물/dain/src/l236x
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6
J=~/workspace/산출물/종민/submit_TS_P600G10
/usr/bin/python3 -W ignore build_hyb2.py d37w05 $J 0.5 1.0 > log/build_d37w05.log 2>&1
/usr/bin/python3 -W ignore build_hyb2.py d37w05t $J 0.5 0.9 --nozip > log/build_d37w05t.log 2>&1
/usr/bin/python3 -W ignore build_hyb2.py d37w10 $J 1.0 1.0 --nozip > log/build_d37w10.log 2>&1
/usr/bin/python3 -W ignore build_hyb2.py d37w00t $J 0.0 0.9 --nozip > log/build_d37w00t.log 2>&1
echo DONE > log/chain_hyb.done
