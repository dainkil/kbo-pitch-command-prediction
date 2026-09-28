#!/bin/bash
cd ~/workspace/산출물/dain/src/l236x
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12
/usr/bin/python3 -W ignore build_rk2.py d35soft025 d35soft025rk > log/build_rk2.log 2>&1
/usr/bin/python3 -W ignore build_hyb2.py dLAST2 ~/workspace/산출물/dain/src/submit_d35soft025rk 0.5 0.9 > log/build_dLAST2.log 2>&1
T=/tmp/dain_last2_test; rm -rf $T; mkdir -p $T/data $T/output; /usr/bin/python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" ../../out/submit_dLAST2.zip $T; cp ~/workspace/data/test.csv ~/workspace/data/sample_submission.csv $T/data/; (cd $T && ~/workspace/산출물/dain/venv_eval/bin/python script.py > ~/workspace/산출물/dain/src/l236x/log/dLAST2_dummy.log 2>&1)
echo DONE > log/chain_rk2.done
