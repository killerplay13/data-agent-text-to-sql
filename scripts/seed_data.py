from pathlib import Path
import sqlite3
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data-agent.db"


def seed_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM deposits")
    cursor.execute("DELETE FROM customers")
    cursor.execute("DELETE FROM relationship_managers")
    cursor.execute("DELETE FROM branches")

    branches = [
        (1, "Taipei Branch", "Taipei"),
        (2, "Taichung Branch", "Taichung"),
        (3, "Kaohsiung Branch", "Kaohsiung"),
    ]

    relationship_managers = [
        (1, "Alice Chen", 1),
        (2, "Brian Lin", 1),
        (3, "Cindy Wang", 2),
        (4, "David Liu", 3),
        (5, "Eva Tsai", 3),
    ]

    customers = [
        (1, "王小明", 1, 1),
        (2, "李小華", 1, 1),
        (3, "陳美玲", 1, 2),
        (4, "林建宏", 2, 3),
        (5, "張雅婷", 2, 3),
        (6, "劉志強", 3, 4),
        (7, "蔡佩珊", 3, 4),
        (8, "吳冠廷", 3, 5),
        (9, "鄭怡君", 1, 2),
        (10, "黃俊傑", 2, 3),
    ]

    now = datetime.utcnow().isoformat()

    deposits = [
        (1, 1, 8000000, "time_deposit", now),
        (2, 2, 2500000, "savings", now),
        (3, 3, 6200000, "time_deposit", now),
        (4, 4, 3100000, "savings", now),
        (5, 5, 1500000, "checking", now),
        (6, 6, 9200000, "time_deposit", now),
        (7, 7, 4700000, "savings", now),
        (8, 8, 2000000, "checking", now),
        (9, 9, 5600000, "time_deposit", now),
        (10, 10, 3900000, "savings", now),
    ]

    cursor.executemany(
        "INSERT INTO branches (branch_id, branch_name, city) VALUES (?, ?, ?)",
        branches
    )

    cursor.executemany(
        "INSERT INTO relationship_managers (rm_id, rm_name, branch_id) VALUES (?, ?, ?)",
        relationship_managers
    )

    cursor.executemany(
        "INSERT INTO customers (customer_id, customer_name, branch_id, rm_id) VALUES (?, ?, ?, ?)",
        customers
    )

    cursor.executemany(
        "INSERT INTO deposits (deposit_id, customer_id, deposit_amount, deposit_type, updated_at) VALUES (?, ?, ?, ?, ?)",
        deposits
    )

    conn.commit()
    conn.close()

    print("Seed data inserted successfully.")


if __name__ == "__main__":
    seed_data()