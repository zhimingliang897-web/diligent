import sqlite3
import os
import argparse
from pprint import pprint

def inspect_db(db_path):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    print(f"Inspecting database: {db_path}")
    print("-" * 50)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("No tables found in the database.")
            return

        print(f"Found {len(tables)} tables:")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  - {table_name}: {count} rows")

        print("-" * 50)
        
        # Optional: Dump checkpoints for inspection
        # Checkpoints in LangGraph are usually storing state in a blob column
        # This is a basic inspection
        if input("\nDo you want to see the last 5 checkpoints info? (y/n): ").lower() == 'y':
            cursor.execute("SELECT * FROM checkpoints ORDER BY thread_id DESC, checkpoint_id DESC LIMIT 5")
            rows = cursor.fetchall()
            # Get column names
            col_names = [description[0] for description in cursor.description]
            print(f"\nColumns: {col_names}")
            for row in rows:
                print(row)

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Agent Memory SQLite DB")
    # 使用正斜杠避免转义问题，或从脚本所在目录向上查找
    default_path = os.path.join(os.path.dirname(__file__), "..", "data", "db", "agent_memory.db")
    default_path = os.path.normpath(default_path)  # 规范化路径
    parser.add_argument("db_path", nargs="?", default=default_path, help="Path to the SQLite database file")
    args = parser.parse_args()
    
    inspect_db(args.db_path)
