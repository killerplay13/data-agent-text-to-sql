from pathlib import Path
import sqlite3


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data-agent.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS branches (
        branch_id INTEGER PRIMARY KEY,
        branch_name TEXT NOT NULL,
        city TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS relationship_managers (
        rm_id INTEGER PRIMARY KEY,
        rm_name TEXT NOT NULL,
        branch_id INTEGER NOT NULL,
        FOREIGN KEY (branch_id) REFERENCES branches(branch_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,
        customer_name TEXT NOT NULL,
        branch_id INTEGER NOT NULL,
        rm_id INTEGER NOT NULL,
        FOREIGN KEY (branch_id) REFERENCES branches(branch_id),
        FOREIGN KEY (rm_id) REFERENCES relationship_managers(rm_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS deposits (
        deposit_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        deposit_amount REAL NOT NULL,
        deposit_type TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    """)

    conn.commit()
    conn.close()

    print(f"Database initialized successfully at: {DB_PATH}")


if __name__ == "__main__":
    init_db()