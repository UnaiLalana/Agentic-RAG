import os
import sqlite3

from config import Config


def get_db():
    """Get a SQLite connection for document metadata."""
    os.makedirs(os.path.dirname(Config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the metadata database."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            file_size INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
