#!/bin/bash
# run_prod.sh [port] [affinity]
# Production launcher for the RX 7900 XTX: the fork binary (nicejunjie/Strata-rdna3, 7df6538d)
# plus its 46 STRATA_* flags, with exactly the config and the HTTPS front end that ran before.
#   port 8093 (default)  -> 0.0.0.0:8093, TLS 9443 via tls_proxy.py
#   any other port       -> test instance, no TLS helper touched
set -u
PORT=${1:-8093}
AFF=${2:-8-15,24-31}
CFG=/home/yanyq/bench/cfg-fork-prod.json
ENVF=/home/yanyq/bench/prod-rdna3.env
EXE=/home/yanyq/port/fork/build-1100/strata
TOK=/data480/models/strata-pack-ud-iq3xxs/tokenizer
SLOG=/home/yanyq/bench/serve-prod.log
ELOG=/home/yanyq/bench/engine-fork-prod.log

[ -x "$EXE" ] || { echo "missing engine: $EXE"; exit 1; }
[ -r "$CFG" ] || { echo "missing config: $CFG"; exit 1; }
[ -r "$ENVF" ] || { echo "missing env file: $ENVF"; exit 1; }

# free the port
fuser -k -TERM "$PORT"/tcp 2>/dev/null
sleep 3
fuser -k -KILL "$PORT"/tcp 2>/dev/null
for i in $(seq 1 20); do
  ss -ltn 2>/dev/null | grep -q ":$PORT " || break
  sleep 2
done
ss -ltn 2>/dev/null | grep -q ":$PORT " && { echo "port $PORT still busy"; exit 1; }
rm -f "$SLOG" "$ELOG"

cd /home/yanyq/strata || exit 1
set -a; . "$ENVF"; set +a
setsid nohup taskset -c "$AFF" /home/yanyq/venv-strata/bin/python -m serve.server --engine strata \
  --config "$CFG" --port "$PORT" --host 0.0.0.0 --tokenizer "$TOK" \
  < /dev/null > "$SLOG" 2>&1 &

for i in $(seq 1 120); do
  grep -q "^ready: " "$SLOG" 2>/dev/null && { echo "ready after $((i*5))s"; break; }
  pgrep -f "serve.server --engine strata.*--port $PORT" > /dev/null || {
    echo "server died"; tail -30 "$SLOG"; exit 1; }
  sleep 5
done
grep -E "^ready:|expert cache" "$SLOG" | head -3

# HTTPS for the production port only, same front end as before: 9443 -> 127.0.0.1:8093
if [ "$PORT" = "8093" ]; then
  if ! ss -ltn 2>/dev/null | grep -q ":9443 "; then
    setsid nohup /home/yanyq/venv-strata/bin/python /home/yanyq/bench/tls_proxy.py \
      --listen 9443 --target 127.0.0.1:8093 \
      --cert /home/yanyq/bench/tls/cert.pem --key /home/yanyq/bench/tls/key.pem \
      < /dev/null > /home/yanyq/bench/tls-9443.log 2>&1 &
    sleep 2
  fi
  echo -n "https 9443 -> "; curl -sk -m 8 https://127.0.0.1:9443/v1/models | head -c 90; echo
fi

echo -n "http $PORT -> "; curl -s -m 8 "http://127.0.0.1:$PORT/v1/models" | head -c 90; echo
echo "env in effect: $(tr '\0' '\n' < /proc/$(pgrep -f "serve.server --engine strata.*--port $PORT" | head -1)/environ | grep -c '^STRATA_') STRATA_* vars"
