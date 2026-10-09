from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.engine import Engine


def upgrade_schema(engine: Engine) -> None:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    with engine.begin() as connection:
        if "products" in inspect(connection).get_table_names() and (
            "alembic_version" not in inspect(connection).get_table_names()
        ):
            raise ValueError(
                "Refusing to migrate an existing unversioned database. "
                "Use a new project database and migrate-sqlite."
            )
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
