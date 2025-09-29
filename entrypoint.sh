#!/bin/sh
set -e

# Export or rewrite variables for async drivers as before
if [ -z "$DATABASE_URL" ]; then
  echo "DATABASE_URL not set"
  exit 1
fi

if echo "$DATABASE_URL" | grep -q "^sqlite://"; then
  export DATABASE_URL=$(echo "$DATABASE_URL" | sed 's#sqlite:///#sqlite+aiosqlite:///#!')
fi

# Corrected a typo here from "$DATABASE_L" to "$DATABASE_URL"
if echo "$DATABASE_URL" | grep -q "^postgresql://"; then
  export DATABASE_URL=$(echo "$DATABASE_URL" | sed 's#postgresql:#postgresql+asyncpg:#')
fi

echo "Starting Alembic migrations"
alembic upgrade head

echo "Starting uvicorn"
echo "Port is set to: $PORT"

# The --forwarded-allow-ips='*' flag is added to fully trust the proxy,
# ensuring the application generates the correct https:// callback URL.
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'

