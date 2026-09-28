#!/bin/bash
cd ~/workspace/산출물/dain/src/l236x
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=12
sed 's#BASE_TS = JM + "/submit_TS_P600G10"#BASE_TS = JM + "/submit_FIN_G/model/jm_root"#' build_tbl.py > build_tbl_fg.py
/usr/bin/python3 -W ignore build_tbl_fg.py fgsoft02 soft --c 0.2 --nozip > log/build_fgsoft02.log 2>&1
/usr/bin/python3 -W ignore build_hyb2.py dFING3 ~/workspace/산출물/dain/src/submit_fgsoft02 0.5 0.9 > log/build_dFING3.log 2>&1
T=/tmp/dain_fing3_test; rm -rf $T; mkdir -p $T/data $T/output; /usr/bin/python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" ../../out/submit_dFING3.zip $T; cp ~/workspace/data/test.csv ~/workspace/data/sample_submission.csv $T/data/; (cd $T && ~/workspace/산출물/dain/venv_eval/bin/python script.py > ~/workspace/산출물/dain/src/l236x/log/dFING3_dummy.log 2>&1)
echo DONE > log/chain_fing3.done
