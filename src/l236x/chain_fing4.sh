#!/bin/bash
cd ~/workspace/산출물/dain/src/l236x
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12
/usr/bin/python3 -W ignore build_tbl_fg.py fgsoft06 soft --c 0.6 --nozip > log/build_fgsoft06.log 2>&1
/usr/bin/python3 -W ignore build_hyb2.py dFING4 ~/workspace/산출물/dain/src/submit_fgsoft06 0.5 0.9 > log/build_dFING4.log 2>&1
echo DONE > log/chain_fing4.done
