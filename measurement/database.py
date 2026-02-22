"""
database.py
SQLite logging layer. Every hypothesis, reward function, episode result,
and novelty score is persisted here so you can query and plot later.
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional


DB_PATH = Path(__file__).parent.parent / "logs" / "experiment.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables. Safe to call multiple times (IF NOT EXISTS)."""
    conn = get_connection()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL,
            agent       TEXT NOT NULL,
            world       TEXT NOT NULL,
            started_at  REAL NOT NULL,
            config      TEXT          -- JSON blob of hyperparams
        );

        CREATE TABLE IF NOT EXISTS episodes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            agent           TEXT NOT NULL,
            world           TEXT NOT NULL,
            episode_num     INTEGER NOT NULL,
            total_reward    REAL NOT NULL,
            steps           INTEGER NOT NULL,
            terminated      INTEGER NOT NULL,  -- 1=fell, 0=truncated (survived)
            timestamp       REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS hypotheses (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT NOT NULL,
            agent               TEXT NOT NULL,
            world               TEXT NOT NULL,
            episode_num         INTEGER NOT NULL,
            hypothesis_text     TEXT NOT NULL,   -- LLM's theory of the physics
            reward_fn_code      TEXT,            -- generated Python code
            boden_type          TEXT,            -- combinational/exploratory/transformational
            boden_justification TEXT,
            embedding           TEXT,            -- JSON list of floats
            embedding_distance  REAL,            -- cosine dist from previous hypothesis
            timestamp           REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS novelty_scores (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            agent           TEXT NOT NULL,
            world           TEXT NOT NULL,
            episode_num     INTEGER NOT NULL,
            linguistic_surprise  REAL,   -- embedding distance from prev hypothesis
            strategy_label  TEXT,        -- LLM-assigned label e.g. "Jitter Strategy"
            behavioral_hash TEXT,        -- hash of state-action trajectory cluster
            timestamp       REAL NOT NULL
        );
    """)

    conn.commit()
    conn.close()


def log_experiment(run_id: str, agent: str, world: str, config: dict = None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO experiments (run_id, agent, world, started_at, config) VALUES (?,?,?,?,?)",
        (run_id, agent, world, time.time(), json.dumps(config or {}))
    )
    conn.commit()
    conn.close()


def log_episode(run_id: str, agent: str, world: str, episode_num: int,
                total_reward: float, steps: int, terminated: bool):
    conn = get_connection()
    # Convert numpy types to native Python types to avoid BLOB storage
    conn.execute(
        """INSERT INTO episodes
           (run_id, agent, world, episode_num, total_reward, steps, terminated, timestamp)
           VALUES (?,?,?,?,?,?,?,?)""",
        (run_id, agent, world, int(episode_num), float(total_reward), int(steps), int(terminated), time.time())
    )
    conn.commit()
    conn.close()


def log_hypothesis(run_id: str, agent: str, world: str, episode_num: int,
                   hypothesis_text: str, reward_fn_code: Optional[str] = None,
                   boden_type: Optional[str] = None,
                   boden_justification: Optional[str] = None,
                   embedding: Optional[list] = None,
                   embedding_distance: Optional[float] = None):
    conn = get_connection()
    # Convert numpy types to native Python types to avoid BLOB storage
    conn.execute(
        """INSERT INTO hypotheses
           (run_id, agent, world, episode_num, hypothesis_text, reward_fn_code,
            boden_type, boden_justification, embedding, embedding_distance, timestamp)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, agent, world, int(episode_num), hypothesis_text, reward_fn_code,
         boden_type, boden_justification,
         json.dumps(embedding) if embedding else None,
         float(embedding_distance) if embedding_distance is not None else None, 
         time.time())
    )
    conn.commit()
    conn.close()


def log_novelty_score(run_id: str, agent: str, world: str, episode_num: int,
                      linguistic_surprise: float, strategy_label: str,
                      behavioral_hash: Optional[str] = None):
    conn = get_connection()
    # Convert numpy types to native Python types to avoid BLOB storage
    conn.execute(
        """INSERT INTO novelty_scores
           (run_id, agent, world, episode_num, linguistic_surprise,
            strategy_label, behavioral_hash, timestamp)
           VALUES (?,?,?,?,?,?,?,?)""",
        (run_id, agent, world, int(episode_num), float(linguistic_surprise),
         strategy_label, behavioral_hash, time.time())
    )
    conn.commit()
    conn.close()


# ── Query helpers for plotting ─────────────────────────────────────────────

def get_episodes_df(run_id: Optional[str] = None):
    """Return all episodes as a pandas DataFrame."""
    import pandas as pd
    # Create connection without row_factory for pandas compatibility
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    
    if run_id:
        query = "SELECT * FROM episodes WHERE run_id = ?"
        df = pd.read_sql_query(query, conn, params=[run_id])
    else:
        query = "SELECT * FROM episodes"
        df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Ensure proper data types for columns that exist
    if not df.empty:
        if 'episode_num' in df.columns:
            df['episode_num'] = df['episode_num'].astype('int64')
        if 'total_reward' in df.columns:
            df['total_reward'] = pd.to_numeric(df['total_reward'], errors='coerce')
        if 'steps' in df.columns:
            df['steps'] = df['steps'].astype('int64')
        if 'terminated' in df.columns:
            df['terminated'] = df['terminated'].astype('int64')
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype('float64')
    
    return df


def get_hypotheses_df(run_id: Optional[str] = None):
    import pandas as pd
    # Create connection without row_factory for pandas compatibility
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    
    if run_id:
        query = "SELECT * FROM hypotheses WHERE run_id = ?"
        df = pd.read_sql_query(query, conn, params=[run_id])
    else:
        query = "SELECT * FROM hypotheses"
        df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Ensure proper data types for columns that exist
    if not df.empty:
        if 'episode_num' in df.columns:
            df['episode_num'] = df['episode_num'].astype('int64')
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype('float64')
        if 'embedding_distance' in df.columns:
            # Handle None/NULL values
            df['embedding_distance'] = pd.to_numeric(df['embedding_distance'], errors='coerce')
    
    return df


def get_novelty_df(run_id: Optional[str] = None):
    import pandas as pd
    # Create connection without row_factory for pandas compatibility
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    
    if run_id:
        query = "SELECT * FROM novelty_scores WHERE run_id = ?"
        df = pd.read_sql_query(query, conn, params=[run_id])
    else:
        query = "SELECT * FROM novelty_scores"
        df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Ensure proper data types for columns that exist
    if not df.empty:
        if 'episode_num' in df.columns:
            df['episode_num'] = df['episode_num'].astype('int64')
        if 'linguistic_surprise' in df.columns:
            df['linguistic_surprise'] = pd.to_numeric(df['linguistic_surprise'], errors='coerce')
        if 'novelty_score' in df.columns:  # Legacy column name
            df['novelty_score'] = pd.to_numeric(df['novelty_score'], errors='coerce')
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype('float64')
    
    return df


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at: {DB_PATH}")
