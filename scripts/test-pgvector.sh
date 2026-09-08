#!/usr/bin/env bash
# Uses a disposable container; never targets an existing PostgreSQL installation.
set -euo pipefail
cd "$(dirname "$0")/.."
image="${HYPATIA_PGVECTOR_TEST_IMAGE:-pgvector/pgvector:pg17-trixie}"
container="hypatia-pgvector-test-$$"
cleanup() { docker rm -f "$container" >/dev/null 2>&1 || true; }
trap cleanup EXIT
image_id="$(docker image inspect "$image" --format '{{.Id}}')"
printf 'Testing image %s (%s)\n' "$image" "$image_id"
docker run -d --name "$container" -e POSTGRES_PASSWORD=hypatia-test-only -e POSTGRES_DB=hypatia_test -p 127.0.0.1:0:5432 "$image_id" >/dev/null
ready=false
for attempt in $(seq 1 30); do
    if docker exec "$container" pg_isready -h 127.0.0.1 -U postgres -d hypatia_test >/dev/null 2>&1; then ready=true; break; fi
    sleep 1
done
if [ "$ready" != true ]; then docker logs "$container"; exit 1; fi
port="$(docker port "$container" 5432/tcp | awk -F: '{print $NF}')"
docker exec "$container" psql -v ON_ERROR_STOP=1 -U postgres -d hypatia_test -c "CREATE EXTENSION vector; SELECT version(); SELECT extversion FROM pg_extension WHERE extname = 'vector';"
export HYPATIA_TEST_POSTGRES_URL="postgres://postgres:hypatia-test-only@127.0.0.1:$port/hypatia_test?sslmode=disable"
cargo test --locked --lib --test backend_contract
cargo test --locked --features postgres-backend --lib --test backend_contract --test postgres_store -- --include-ignored
