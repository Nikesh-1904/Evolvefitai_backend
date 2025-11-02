#!/bin/sh
set -e

echo ""
echo "========================================"
echo "🔍 DEBUG ENTRYPOINT - VERBOSE MODE"
echo "========================================"
echo ""

# Add debug trap to see which commands are executed
set -x

# Step 1: Check environment
echo "📋 Step 1: Environment check"
echo "DATABASE_URL length: ${#DATABASE_URL}"
echo "SECRET_KEY set: $([ -n "$SECRET_KEY" ] && echo "YES" || echo "NO")"
echo "QR_ENCRYPTION_KEY set: $([ -n "$QR_ENCRYPTION_KEY" ] && echo "YES" || echo "NO")"
echo ""

# Step 2: Database connectivity
echo "📡 Step 2: Testing database..."
DB_HOST=$(echo "$DATABASE_URL" | sed -n 's#.*@\([^:/?]*\).*#\1#p')
DB_PORT=$(echo "$DATABASE_URL" | sed -n 's#.*:\([0-9]*\)/.*#\1#p')
[ -z "$DB_PORT" ] && DB_PORT=5432

if nc -z "$DB_HOST" "$DB_PORT"; then
  echo "✅ Database reachable"
else
  echo "❌ Database unreachable"
  exit 1
fi
echo ""

# Step 3: Run migrations
echo "🗃️  Step 3: Running migrations..."
if alembic upgrade head; then
  echo "✅ Migrations completed"
else
  echo "❌ Migrations failed"
  exit 1
fi
echo ""

# Step 4: Test Python import
echo "🐍 Step 4: Testing Python imports..."
if python -c "import app.main; print('✅ Import successful')"; then
  echo "✅ Python imports work"
else
  echo "❌ Python import failed - THIS IS THE ISSUE"
  exit 1
fi
echo ""

# Step 5: Show what we're about to run
echo "========================================"
echo "🚀 Step 5: STARTING UVICORN"
echo "========================================"
echo "Command: uvicorn app.main:app"
echo "Host: 0.0.0.0"
echo "Port: ${PORT:-8000}"
echo "========================================"
echo ""

# Step 6: Actually start the server
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --log-level debug \
  --timeout-keep-alive 65

echo "❌ This line should NEVER print (exec replaces the process)"