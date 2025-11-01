#!/bin/sh
set -e

echo "========================================"
echo "EvolveFit AI Backend Starting..."
echo "========================================"

# ===== 1. DATABASE URL VALIDATION =====
if [ -z "$DATABASE_URL" ]; then
  echo "❌ ERROR: DATABASE_URL not set"
  exit 1
fi

echo "✅ DATABASE_URL is set"

# ===== 2. CONVERT DATABASE URL FOR ASYNC DRIVERS =====
# Save original for Alembic (sync)
export ALEMBIC_DATABASE_URL="$DATABASE_URL"

# SQLite conversion
if echo "$DATABASE_URL" | grep -q "^sqlite://"; then
  export DATABASE_URL=$(echo "$DATABASE_URL" | sed 's#sqlite:///#sqlite+aiosqlite:///#')
  echo "✅ Converted SQLite URL for async driver"
fi

# PostgreSQL conversion (for async runtime)
if echo "$DATABASE_URL" | grep -q "^postgresql://"; then
  export DATABASE_URL=$(echo "$DATABASE_URL" | sed 's#postgresql://#postgresql+asyncpg://#')
  echo "✅ Converted PostgreSQL URL for async driver"
fi

# ===== 3. SKIP MIGRATIONS ON RAILWAY (OPTIONAL) =====
if [ "$SKIP_MIGRATIONS" = "true" ]; then
  echo "⏭️  Skipping migrations (SKIP_MIGRATIONS=true)"
else
  echo ""
  echo "========================================"
  echo "Running Database Migrations..."
  echo "========================================"
  
  # Try migrations with timeout and retry
  MIGRATION_TIMEOUT=30
  MIGRATION_RETRIES=3
  RETRY=0
  
  while [ $RETRY -lt $MIGRATION_RETRIES ]; do
    if timeout $MIGRATION_TIMEOUT alembic upgrade head 2>&1; then
      echo "✅ Migrations completed successfully"
      break
    else
      RETRY=$((RETRY + 1))
      if [ $RETRY -lt $MIGRATION_RETRIES ]; then
        echo "⚠️  Migration attempt $RETRY failed, retrying in 5s..."
        sleep 5
      else
        echo "❌ Migration failed after $MIGRATION_RETRIES attempts"
        echo ""
        echo "🔧 TROUBLESHOOTING:"
        echo "  1. Check DATABASE_URL is correct in Railway"
        echo "  2. Ensure PostgreSQL service is running"
        echo "  3. Try setting SKIP_MIGRATIONS=true to start without migrations"
        echo "  4. You can run migrations manually later with:"
        echo "     railway run alembic upgrade head"
        echo ""
        echo "⚠️  Starting server anyway (migrations will retry on next restart)..."
      fi
    fi
  done
fi

# ===== 4. START UVICORN SERVER =====
echo ""
echo "========================================"
echo "Starting Uvicorn Server..."
echo "========================================"
echo "Port: ${PORT:-8000}"
echo "Environment: ${RAILWAY_ENVIRONMENT:-development}"
echo "========================================"

# Start the application with Railway-specific settings
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port ${PORT:-8000} \
  --proxy-headers \
  --forwarded-allow-ips='*' \
  --log-level info \
  --timeout-keep-alive 65