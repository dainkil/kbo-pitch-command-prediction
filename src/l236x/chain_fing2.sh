#!/bin/bash
cd ~/workspace/산출물/dain/src/l236x
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12
/usr/bin/python3 -W ignore build_hyb2.py dFING2 ~/workspace/산출물/종민/submit_FIN_G/model/jm_root 0.5 0.9 > log/build_dFING2.log 2>&1
T=/tmp/dain_fing2_test; rm -rf $T; mkdir -p $T/data $T/output; /usr/bin/python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" ../../out/submit_dFING2.zip $T; cp ~/workspace/data/test.csv ~/workspace/data/sample_submission.csv $T/data/; (cd $T && ~/workspace/산출물/dain/venv_eval/bin/python script.py > ~/workspace/산출물/dain/src/l236x/log/dFING2_dummy.log 2>&1)
echo DONE > log/chain_fing2.done
