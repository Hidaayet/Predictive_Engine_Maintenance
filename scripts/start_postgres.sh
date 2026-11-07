#!/usr/bin/env bash
# Starts a throw-away local PostgreSQL cluster on port 5433 (data in ./.pgdata) unless one is already reachable.
# Skip this script if you already have a PostgreSQL server: just set DATABASE_URL.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PGBIN="${PGBIN:-$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V | tail -1)}"
PGDATA="$ROOT/.pgdata"; PORT="${PGPORT:-5433}"
if "$PGBIN/pg_isready" -q -h localhost -p "$PORT"; then echo "postgres already up on $PORT"; exit 0; fi
run() { if [ "$(id -u)" = 0 ]; then runuser -u pgrun -- "$@"; else "$@"; fi; }
if [ "$(id -u)" = 0 ]; then id pgrun >/dev/null 2>&1 || useradd -m pgrun; chown -R pgrun "$ROOT/.pgdata" 2>/dev/null || true; mkdir -p "$PGDATA"; chown -R pgrun "$PGDATA"; chmod o+x "$(dirname "$ROOT")" "$ROOT" || true; fi
[ -f "$PGDATA/PG_VERSION" ] || run "$PGBIN/initdb" -D "$PGDATA" -A trust -U postgres >/dev/null
run "$PGBIN/pg_ctl" -D "$PGDATA" -o "-p $PORT -k $PGDATA" -l "$PGDATA/server.log" -w start
"$PGBIN/psql" -h localhost -p "$PORT" -U postgres -tc "SELECT 1 FROM pg_database WHERE datname='cmapss'" | grep -q 1 || "$PGBIN/createdb" -h localhost -p "$PORT" -U postgres cmapss
echo "postgres ready: postgresql://postgres@localhost:$PORT/cmapss"
