#!/bin/bash
# submit.zip 생성 + 평가서버 모사 테스트 (venv_eval: py3.11 / pandas 2.0.3 / numpy 1.26.4 / lightgbm 4.7.0)
set -e
cd "$(dirname "$0")/submit"; find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
MDIR=${2:-model}; [ "$MDIR" != "model" ] && { rm -rf model; cp -r "$MDIR" model; }
if [ -n "$1" ]; then TAG=$1; else N=$(ls ../../out/submit_d*.zip 2>/dev/null | sed "s/.*submit_d\([0-9]*\).zip/\1/" | sort -n | tail -1); TAG=d$(( ${N:-0} + 1 )); fi
ZIP=$(cd ../../out && pwd)/submit_$TAG.zip; PY=~/workspace/산출물/dain/venv_eval/bin/python
/usr/bin/python3 - "$ZIP" <<'PY'
import sys, zipfile, os
z = zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED)
for f in ("script.py", "requirements.txt"): z.write(f)
for root, _, files in os.walk("model"):
    for f in files: z.write(os.path.join(root, f))
z.close(); print("zip →", sys.argv[1], f"{os.path.getsize(sys.argv[1])/1e6:.1f} MB"); print("\n".join(sorted(zipfile.ZipFile(sys.argv[1]).namelist())))
PY
# 1) 공개 더미 5행으로 구조 테스트
T=/tmp/dain_submit_test; rm -rf $T; mkdir -p $T/data $T/output
/usr/bin/python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" $T
cp ~/workspace/data/test.csv ~/workspace/data/sample_submission.csv $T/data/
cd $T; echo "--- [1] 더미 5행:"; $PY script.py; cat output/submission.csv
# 2) 2024 시즌 라벨 행을 2025 로 바꿔 218,992행 규모 타이밍 테스트
$PY - <<'PY'
import pandas as pd
tr = pd.read_csv("/home/ubuntu/workspace/data/train.csv", dtype={"row_id": str}); oc = pd.read_csv("/home/ubuntu/workspace/data/train_outcomes.csv", dtype={"row_id": str}, usecols=["row_id","control_grade"])
d = tr.merge(oc, on="row_id"); d = d[(d.season == 2024) & d.control_grade.notna()].head(218992).copy(); d["season"] = 2025
d.drop(columns=["control_grade"]).to_csv("data/test.csv", index=False)
pd.DataFrame({"row_id": d.row_id, "prob_shadow": 1/3, "prob_heart": 1/3, "prob_failure": 1/3}).to_csv("data/sample_submission.csv", index=False)
print("synthetic test rows:", len(d))
PY
echo "--- [2] 218,992행 타이밍 (6 vCPU 모사: taskset 0-5):"; /usr/bin/time -f "  elapsed %es · maxRSS %MKB" taskset -c 0-5 $PY script.py
head -3 output/submission.csv
