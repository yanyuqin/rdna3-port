#!/bin/bash
# run_all.sh <port> <tag>
#   eval100 全量(100) -> RULER qa(12) -> polyglot100 全量(100), 全程采样 7900 温度
# 输出: /home/yanyq/port/logs/<tag>/{eval100,ruler_qa,polyglot100}.jsonl + *.log + temp.csv + phases.txt
set -u
PORT=${1:-8093}
TAG=${2:-full-$(date +%m%d-%H%M)}
E=/home/yanyq/eval100-kit
OUT=/home/yanyq/port/logs/$TAG
mkdir -p "$OUT"

curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "port $PORT not answering"; exit 1; }

phase() { echo "$(date +%s) $(date +%H:%M:%S) $1" >> "$OUT/phases.txt"; }
phase "start port=$PORT"

python3 /home/yanyq/bench/temp_sample.py --out "$OUT/temp.csv" --tag "$TAG" > "$OUT/temp.log" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null' EXIT

# 1) eval100 全量：MMLU 60 + DROP 40，thinking OFF
phase "eval100 begin"
(cd "$E" && timeout 14400 python3 eval100.py --port "$PORT" --tag "${TAG}e100" \
   --data eval100.json --out "$OUT/eval100.jsonl") 2>&1 | tee "$OUT/eval100.log" | tail -3
phase "eval100 end"

# 2) RULER qa 部分：qa_1 x6 + qa_2 x6，65k 上下文，thinking OFF
phase "ruler_qa begin"
(cd "$E" && timeout 7200 python3 ruler100.py --port "$PORT" --tag "${TAG}rqa" \
   --data ruler_qa.json --out "$OUT/ruler_qa.jsonl") 2>&1 | tee "$OUT/ruler_qa.log" | tail -3
phase "ruler_qa end"

# 3) polyglot100 全量：thinking ON + effort medium，官方测试判分
phase "polyglot begin"
export PATH=/home/yanyq/venv-strata/bin:/home/yanyq/go-toolchain/go/bin:/usr/bin:/bin
export GOCACHE=/home/yanyq/.gocache
(cd /home/yanyq/polyglot-benchmark-b && timeout 43200 /home/yanyq/venv-strata/bin/python "$E/polyglot100.py" \
   --port "$PORT" --tag "${TAG}poly" --data "$E/polyglot100.json" --out "$OUT/polyglot100.jsonl") 2>&1 \
   | tee "$OUT/polyglot100.log" | tail -3
phase "polyglot end"

kill $SP 2>/dev/null
phase "all done"
echo "=== phases ==="
cat "$OUT/phases.txt"
echo "=== temp rows: $(wc -l < "$OUT/temp.csv") ==="
echo "OUTDIR=$OUT"
