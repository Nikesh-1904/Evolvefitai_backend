import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models import Base  # Adjust import if needed

# This is the Alembic Config object, which provides access to the values within
# the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
fileConfig(config.config_file_name)

from app.core.config import settings
import os

# Get the database URL - try multiple sources
sync_database_url = os.getenv("ALEMBIC_DATABASE_URL") or settings.DATABASE_URL

# CRITICAL: Alembic needs a SYNC database driver
# Railway/runtime uses async drivers, so we must convert

if sync_database_url:
    # Convert async PostgreSQL URL to sync
    if "postgresql+asyncpg://" in sync_database_url:
        sync_database_url = sync_database_url.replace("postgresql+asyncpg://", "postgresql://")
        print(f"✅ Converted asyncpg → psycopg2 for Alembic")
    
    # Convert async SQLite URL to sync
    elif "sqlite+aiosqlite://" in sync_database_url:
        sync_database_url = sync_database_url.replace("sqlite+aiosqlite://", "sqlite:///")
        print(f"✅ Converted aiosqlite → sqlite3 for Alembic")
    
    # If already sync, just confirm
    elif sync_database_url.startswith("postgresql://"):
        print(f"✅ Using sync PostgreSQL URL for Alembic")
    elif sync_database_url.startswith("sqlite:///"):
        print(f"✅ Using sync SQLite URL for Alembic")
else:
    print("❌ ERROR: No DATABASE_URL found!")
    raise ValueError("DATABASE_URL must be set for migrations")

# Set the sqlalchemy.url in Alembic's config object dynamically
config.set_main_option("sqlalchemy.url", sync_database_url)
print(f"📊 Alembic using: {sync_database_url.split('@')[0]}@***")

# Alembic needs a non-async (sync) database driver.
# We must replace 'postgresql+asyncpg' with 'postgresql'
if sync_database_url and sync_database_url.startswith("postgresql+asyncpg"):
    sync_database_url = sync_database_url.replace("postgresql+asyncpg", "postgresql")

# Set the sqlalchemy.url in Alembic's config object dynamically
config.set_main_option("sqlalchemy.url", sync_database_url)

# target_metadata used for 'autogenerate'
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
