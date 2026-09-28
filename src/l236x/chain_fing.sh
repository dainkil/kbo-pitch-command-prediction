#!/bin/bash
cd ~/workspace/산출물/dain/src/l236x
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12
J=$(ls -d ~/workspace/산출물/종민/submit_FIN_G 2>/dev/null || ls -d ~/workspace/산출물/*/submit_FIN_G | head -1)
echo "JMDIR=$J" > log/build_dFING.log
/usr/bin/python3 -W ignore build_hyb2.py dFING $J 0.5 0.9 >> log/build_dFING.log 2>&1
T=/tmp/dain_fing_test; rm -rf $T; mkdir -p $T/data $T/output; /usr/bin/python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" ../../out/submit_dFING.zip $T; cp ~/workspace/data/test.csv ~/workspace/data/sample_submission.csv $T/data/; (cd $T && ~/workspace/산출물/dain/venv_eval/bin/python script.py > ~/workspace/산출물/dain/src/l236x/log/dFING_dummy.log 2>&1)
echo DONE > log/chain_fing.done
