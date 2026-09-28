#!/bin/bash
cd ~/workspace/산출물/dain/src/l236x
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6
/usr/bin/python3 -W ignore build_tbl.py d35soft02 soft --c 0.2 > log/build_d35soft02.log 2>&1
/usr/bin/python3 -W ignore build_stack.py d35soft02 d36e "TS,d34dir,d34hier" > log/build_d36e.log 2>&1
/usr/bin/python3 -W ignore build_tbl.py d35soft045 soft --c 0.45 > log/build_d35soft045.log 2>&1
/usr/bin/python3 -W ignore build_stack.py d35soft045 d36f "TS,d34dir,d34hier" > log/build_d36f.log 2>&1
echo CHAIN_DONE > log/chain_d36ef.done
