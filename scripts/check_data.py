from pathlib import Path
import sqlite3


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data-agent.db"


def check_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """
    SELECT c.customer_name, d.deposit_amount
    FROM customers c
    JOIN deposits d ON c.customer_id = d.customer_id
    ORDER BY d.deposit_amount DESC
    LIMIT 5
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    print("Top 5 customers by deposit:")
    for row in rows:
        print(row)

    conn.close()


if __name__ == "__main__":
    check_data()