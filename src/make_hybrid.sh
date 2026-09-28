#!/bin/bash
# 사용: make_hybrid.sh <TAG> <TEAM_T_ZIP abs> [Q_MODEL_DIR abs, 기본 submit_d20/model]
# = 팀 최고 t(그 zip 의 script+model 을 jm_root 에 동봉) × dain q → zip + 타이밍 + 행독립 + 동일성 검산 (2024 가짜 test)
set -e; TAG=$1; TZIP=$2; Q=${3:-$HOME/workspace/산출물/dain/src/submit_d20/model}; cd ~/workspace/산출물/dain/src
rm -rf submit_$TAG /tmp/tzip_$TAG && mkdir -p submit_$TAG/model/jm_root /tmp/tzip_$TAG && (cd /tmp/tzip_$TAG && unzip -q "$TZIP")
cp $Q/meta.json submit_$TAG/model/ && cp $Q/lgb_seed*.txt submit_$TAG/model/ && cp -r $Q/tables submit_$TAG/model/tables && cp /tmp/tzip_$TAG/script.py submit_$TAG/model/jm_root/ && cp -r /tmp/tzip_$TAG/model submit_$TAG/model/jm_root/model
/usr/bin/python3 - "$TAG" "$TZIP" <<'PY'
import json, sys; p = f"submit_{sys.argv[1]}/model/meta.json"; m = json.load(open(p)); m["mode"] = "jm_anchor"; m["jm_source"] = sys.argv[2]; m["inject_eps"] = 0; m.pop("inject", None); json.dump(m, open(p, "w"), ensure_ascii=False, indent=1); print("meta ok ·", sys.argv[2])
PY
bash pack_and_test2.sh $TAG "$PWD/submit_$TAG/model" "$PWD/script_hybrid" 2>&1 | grep -E "zip →|saved output.*218|elapsed|Error|error|Traceback"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 taskset -c 12-17 /usr/bin/python3 -W ignore verify_with.py "$PWD/script_hybrid" submit_$TAG/model ../out/fake_test_2024.csv 2>&1 | grep -E "행 독립|형식"
