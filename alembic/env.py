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

# Get the database URL from our app's settings
# (which loads from .env or environment variables)
sync_database_url = settings.DATABASE_URL

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
