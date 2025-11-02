#!/bin/sh
set -e

echo ""
echo "========================================"
echo "🚀 EVOLVEFITAI BACKEND STARTING"
echo "========================================"
echo ""

# ============================================================================
# STEP 1: VALIDATE ENVIRONMENT VARIABLES
# ============================================================================
echo "📋 Step 1: Validating environment variables..."

if [ -z "$DATABASE_URL" ]; then
  echo "❌ FATAL ERROR: DATABASE_URL is not set"
  echo ""
  echo "🔧 How to fix:"
  echo "   1. Go to Railway dashboard"
  echo "   2. Make sure PostgreSQL service is added"
  echo "   3. Check Variables tab for DATABASE_URL"
  echo ""
  exit 1
fi

echo "✅ DATABASE_URL is set"

if [ -z "$SECRET_KEY" ]; then
  echo "⚠️  WARNING: SECRET_KEY not set (required for production)"
fi

if [ -z "$QR_ENCRYPTION_KEY" ]; then
  echo "⚠️  WARNING: QR_ENCRYPTION_KEY not set (QR features disabled)"
fi

echo ""

# ============================================================================
# STEP 2: TEST DATABASE CONNECTIVITY
# ============================================================================
echo "📡 Step 2: Testing database connectivity..."

# Extract database host and port from URL
DB_HOST=$(echo "$DATABASE_URL" | sed -n 's#.*@\([^:/?]*\).*#\1#p')
DB_PORT=$(echo "$DATABASE_URL" | sed -n 's#.*:\([0-9]*\)/.*#\1#p')

# Default port if not found
if [ -z "$DB_PORT" ]; then
  DB_PORT=5432
fi

if [ -n "$DB_HOST" ]; then
  echo "   Testing connection to: $DB_HOST:$DB_PORT"
  
  # Wait for database (max 60 seconds)
  MAX_WAIT=60
  ELAPSED=0
  CONNECTED=0
  
  while [ $ELAPSED -lt $MAX_WAIT ]; do
    if nc -z "$DB_HOST" "$DB_PORT" 2>/dev/null; then
      echo "✅ Database is reachable at $DB_HOST:$DB_PORT"
      CONNECTED=1
      break
    fi
    
    if [ $((ELAPSED % 10)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
      echo "   Still waiting... ($ELAPSED/${MAX_WAIT}s)"
    fi
    
    sleep 2
    ELAPSED=$((ELAPSED + 2))
  done
  
  if [ $CONNECTED -eq 0 ]; then
    echo "❌ ERROR: Cannot connect to database after ${MAX_WAIT} seconds"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Check if PostgreSQL service is running in Railway"
    echo "   2. Verify DATABASE_URL format"
    echo "   3. Check Railway service logs for database issues"
    echo ""
    exit 1
  fi
else
  echo "⚠️  WARNING: Could not parse database host from URL"
  echo "   Proceeding anyway..."
fi

echo ""

# ============================================================================
# STEP 3: PREPARE DATABASE URLs FOR DIFFERENT DRIVERS
# ============================================================================
echo "🔄 Step 3: Converting database URLs..."

# Save original for Alembic (sync driver)
export ALEMBIC_DATABASE_URL="$DATABASE_URL"

# Convert for async runtime
if echo "$DATABASE_URL" | grep -q "^sqlite://"; then
  export DATABASE_URL=$(echo "$DATABASE_URL" | sed 's#sqlite:///#sqlite+aiosqlite:///#')
  echo "✅ Converted SQLite URL for async driver (aiosqlite)"
elif echo "$DATABASE_URL" | grep -q "^postgresql://"; then
  export DATABASE_URL=$(echo "$DATABASE_URL" | sed 's#postgresql://#postgresql+asyncpg://#')
  echo "✅ Converted PostgreSQL URL for async driver (asyncpg)"
else
  echo "ℹ️  Database URL format: $(echo $DATABASE_URL | sed 's#://.*@# → #')"
fi

echo ""

# ============================================================================
# STEP 4: RUN DATABASE MIGRATIONS
# ============================================================================
if [ "$SKIP_MIGRATIONS" = "true" ]; then
  echo "⏭️  Step 4: SKIPPED (SKIP_MIGRATIONS=true)"
else
  echo "🗃️  Step 4: Running database migrations..."
  echo ""
  
  # Show current migration state
  echo "   Checking current migration status..."
  if alembic current 2>/dev/null; then
    echo ""
  else
    echo "   (No migrations applied yet)"
    echo ""
  fi
  
  # Run migrations with detailed output
  echo "   Applying pending migrations..."
  if alembic upgrade head 2>&1; then
    echo ""
    echo "✅ Migrations completed successfully"
    echo ""
    echo "   Final migration state:"
    alembic current 2>/dev/null || echo "   (Could not retrieve status)"
  else
    echo ""
    echo "❌ MIGRATION FAILED!"
    echo ""
    echo "🔧 Troubleshooting steps:"
    echo "   1. Check the error message above"
    echo "   2. Verify DATABASE_URL is correct"
    echo "   3. Ensure PostgreSQL service is healthy"
    echo "   4. Check if tables already exist (conflict)"
    echo "   5. Try manually: railway run alembic upgrade head"
    echo ""
    echo "Common issues:"
    echo "   • 'relation already exists' → Database has old schema"
    echo "   • 'connection refused' → Database not ready"
    echo "   • 'authentication failed' → Wrong credentials"
    echo ""
    exit 1
  fi
fi

echo ""

# ============================================================================
# STEP 5: VERIFY APPLICATION FILES
# ============================================================================
echo "📂 Step 5: Verifying application files..."

if [ ! -f "app/main.py" ]; then
  echo "❌ ERROR: app/main.py not found!"
  echo "   Current directory: $(pwd)"
  echo "   Files present:"
  ls -la
  exit 1
fi

echo "✅ Application files found"
echo ""

# ============================================================================
# STEP 6: START UVICORN SERVER
# ============================================================================
echo "========================================"
echo "🎯 STARTING UVICORN SERVER"
echo "========================================"
echo "   Host: 0.0.0.0"
echo "   Port: ${PORT:-8000}"
echo "   Environment: ${RAILWAY_ENVIRONMENT:-development}"
echo "   Workers: 1"
echo "========================================"
echo ""

# Start the application
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --proxy-headers \
  --forwarded-allow-ips='*' \
  --log-level info \
  --timeout-keep-alive 65 \
  --no-access-log