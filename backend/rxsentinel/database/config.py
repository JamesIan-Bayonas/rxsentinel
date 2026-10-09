import os
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool


def database_url() -> str | None:
    # Read the local config without mutating process environment or logging credentials.
    value = os.environ.get("RXSENTINEL_DATABASE_URL")
    if value is None:
        value = dotenv_values(Path(".env")).get("RXSENTINEL_DATABASE_URL")
    return value or None


def engine_for(url: str) -> Engine:
    kwargs = {"pool_pre_ping": True, "hide_parameters": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url.endswith(":memory:"):
            kwargs["poolclass"] = StaticPool
    elif url.startswith(("mysql", "mariadb")):
        kwargs["connect_args"] = {"connect_timeout": 10, "read_timeout": 30, "write_timeout": 30}
        kwargs["pool_recycle"] = 1800
    else:
        raise ValueError("Supported databases are MariaDB/MySQL and SQLite")
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def sqlite_constraints(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")

    return engine
