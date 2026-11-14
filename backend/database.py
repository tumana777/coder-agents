import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from backend.agents import build_graph

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

@asynccontextmanager
async def lifespan(app):
    database_url = f"postgres://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    if not database_url:
        raise ValueError("DATABASE_URL is not set")

    pool = ConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=10,
        kwargs={
            "row_factory": dict_row,
            "autocommit": True
        },
        open=False
    )

    try:
        pool.open(wait=True)

        checkpointer = PostgresSaver(pool)
        checkpointer.setup()

        app.state.graph = build_graph(checkpointer)

        yield

    finally:
        pool.close()






















