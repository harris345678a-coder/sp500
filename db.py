import os, json, psycopg2, psycopg2.extras
from typing import Any, Dict

DDL_SQL = """
CREATE TABLE IF NOT EXISTS daily_rank_runs (
  id SERIAL PRIMARY KEY,
  as_of DATE NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  top3 JSONB NOT NULL,
  top50 JSONB NOT NULL
);
"""

def maybe_init_db():
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    conn = psycopg2.connect(url)
    with conn, conn.cursor() as cur:
        cur.execute(DDL_SQL)
    return conn

def save_run(conn, payload: Dict[str, Any]):
    with conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO daily_rank_runs(as_of, top3, top50) VALUES (%s, %s, %s)",
            (payload["as_of"], json.dumps(payload["top3"]), json.dumps(payload["top50"]))
        )

def latest_run(conn):
    with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM daily_rank_runs ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
        return dict(row) if row else None
