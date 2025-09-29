#!/bin/bash
# Wait for Postgres to be ready (Railway sets DATABASE_URL)
if [ -z "$DATABASE_URL" ]; then
    echo "DATABASE_URL is not set! Please set it in Railway or your env."
    exit 1
fi

# If using SQLite in dev, make sure to force aiosqlite
if [[ "$DATABASE_URL" == sqlite:///* ]]; then
    export DATABASE_URL=${DATABASE_URL/sqlite:\/\//sqlite+aiosqlite:\/\/}
fi
if [[ "$DATABASE_URL" == postgresql://* ]]; then
    export DATABASE_URL=${DATABASE_URL/postgresql:/postgresql+asyncpg:}
fi

# Wait for Postgres if needed (uncomment if you want to pause until DB is up)
# until pg_isready -h $PGHOST -p $PGPORT -U $PGUSER; do
#   echo "Waiting for database..."
#   sleep 2
# done

echo "Running Alembic migrations..."
alembic upgrade head

echo "Starting FastAPI app..."
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
