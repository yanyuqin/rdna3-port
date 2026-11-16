#!/bin/bash
# run_prod_0214.sh [port] [affinity]
# Production launcher for the RX 7900 XTX on the upstream v0.2.14.1 tree with the rdna3-port patches
# (branch rdna3-port-v0214: kfdq power parking, the gfx1100 build and this launcher), on the tuned b9 line.
#   port 8094 (default) -> 0.0.0.0:8094, TLS 9444 via tls_proxy.py
#   any other port      -> test instance, no TLS helper touched
# Only the RX 7900 XTX is visible to HIP (UUID, not an index), so the two MI50s stay idle.
set -u
PORT=${1:-8094}
AFF=${2:-8-15,24-31}
CFG=${CFG:-/home/yanyq/bench/cfg-a0214-mv.json}
ENVF=${ENVF:-/home/yanyq/port/env-up-v0214-mv.sh}
EXE=${EXE:-/home/yanyq/port/up0214/build-1100-mv/strata}
SRVROOT=${SRVROOT:-/home/yanyq/port/up0214}
TOK=${TOK:-/data480/models/strata-pack-ud31-iq3xxs/tokenizer}
SLOG=${SLOG:-/home/yanyq/bench/serve-prod-0214.log}
ELOG=${ELOG:-/home/yanyq/port/logs/engine-prod-0214.log}
TLS_PORT=${TLS_PORT:-9444}

unset HSA_OVERRIDE_GFX_VERSION ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=${VIS_7900XTX:-GPU-b8626a2a8e10f3cc}

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
mkdir -p "$(dirname "$ELOG")"
rm -f "$SLOG" "$ELOG"

cd "$SRVROOT" || exit 1
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
grep -E "^ready:|expert cache" "$SLOG" "$ELOG" 2>/dev/null | head -3

if [ "$PORT" = "8094" ]; then
  if ! ss -ltn 2>/dev/null | grep -q ":$TLS_PORT "; then
    setsid nohup /home/yanyq/venv-strata/bin/python /home/yanyq/bench/tls_proxy.py \
      --listen "$TLS_PORT" --target 127.0.0.1:8094 \
      --cert /home/yanyq/bench/tls/cert.pem --key /home/yanyq/bench/tls/key.pem \
      < /dev/null > "/home/yanyq/bench/tls-$TLS_PORT.log" 2>&1 &
    sleep 2
  fi
  echo -n "https $TLS_PORT -> "; curl -sk -m 8 "https://127.0.0.1:$TLS_PORT/v1/models" | head -c 90; echo
fi

echo -n "http $PORT -> "; curl -s -m 8 "http://127.0.0.1:$PORT/v1/models" | head -c 90; echo
echo "env in effect: $(tr '\0' '\n' < /proc/$(pgrep -f "serve.server --engine strata.*--port $PORT" | head -1)/environ | grep -c '^STRATA_') STRATA_* vars"
