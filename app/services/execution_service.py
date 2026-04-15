from pathlib import Path
import sqlite3


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data-agent.db"


class ExecutionService:
    def __init__(self):
        self.db_path = DB_PATH

    def is_safe_select_query(self, sql: str) -> bool:
        sql_clean = sql.strip().lower()
        return sql_clean.startswith("select") and all(
            keyword not in sql_clean
            for keyword in ["insert", "update", "delete", "drop", "alter", "truncate"]
        )

    def execute_query(self, sql: str):
        if not self.is_safe_select_query(sql):
            raise ValueError("Only safe SELECT queries are allowed.")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(sql)
        rows = cursor.fetchall()

        results = [dict(row) for row in rows]

        conn.close()
        return results