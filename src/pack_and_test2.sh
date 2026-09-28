#!/bin/bash
# 사용: pack_and_test2.sh <TAG> <MODEL_DIR abs> <SCRIPT_DIR abs (script.py+requirements.txt)> — submit/ 스테이징 미사용
set -e; TAG=$1; MDIR=$2; SDIR=$3; OUT=~/workspace/산출물/dain/out; ZIP=$OUT/submit_$TAG.zip; PY=~/workspace/산출물/dain/venv_eval/bin/python
S=/tmp/stage_$TAG; rm -rf $S; mkdir -p $S; cp $SDIR/script.py $SDIR/requirements.txt $S/; cp -r "$MDIR" $S/model; find $S -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
cd $S; /usr/bin/python3 - "$ZIP" <<'PY'
import sys, zipfile, os
z = zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED)
for f in ("script.py", "requirements.txt"): z.write(f)
for root, _, files in os.walk("model"):
    for f in files: z.write(os.path.join(root, f))
z.close(); print("zip →", sys.argv[1], f"{os.path.getsize(sys.argv[1])/1e6:.1f} MB", len(zipfile.ZipFile(sys.argv[1]).namelist()), "files")
PY
T=/tmp/dain_submit_test_$TAG; rm -rf $T; mkdir -p $T/data $T/output
/usr/bin/python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" $T
cp ~/workspace/data/test.csv ~/workspace/data/sample_submission.csv $T/data/
cd $T; echo "--- [1] 더미 5행:"; $PY script.py
if [ -f /tmp/dain_synth_test.csv ]; then cp /tmp/dain_synth_test.csv data/test.csv; cp /tmp/dain_synth_sample.csv data/sample_submission.csv; echo "synthetic test rows: (cached)"; else
$PY - <<'PY'
import pandas as pd
tr = pd.read_csv("/home/ubuntu/workspace/data/train.csv", dtype={"row_id": str}); oc = pd.read_csv("/home/ubuntu/workspace/data/train_outcomes.csv", dtype={"row_id": str}, usecols=["row_id","control_grade"])
d = tr.merge(oc, on="row_id"); d = d[(d.season == 2024) & d.control_grade.notna()].head(218992).copy(); d["season"] = 2025
d.drop(columns=["control_grade"]).to_csv("data/test.csv", index=False)
pd.DataFrame({"row_id": d.row_id, "prob_shadow": 1/3, "prob_heart": 1/3, "prob_failure": 1/3}).to_csv("data/sample_submission.csv", index=False)
print("synthetic test rows:", len(d))
PY
cp data/test.csv /tmp/dain_synth_test.csv; cp data/sample_submission.csv /tmp/dain_synth_sample.csv; fi
echo "--- [2] 218,836행 타이밍 (taskset 0-5):"; /usr/bin/time -f "  elapsed %es · maxRSS %MKB" taskset -c 0-5 $PY script.py
