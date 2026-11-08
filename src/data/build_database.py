"""src/data/build_database.py -- Store Clean Data into SQLite Database (Part A Task A4)

1. Copies data/raw/products.csv to data/clean/products.csv verbatim.
2. Loads clean customers, products, and orders datasets.
3. Builds SQLite database data/instamart.db containing tables:
   - customers
   - products
   - orders
4. Stores order_ts formatted as YYYY-MM-DD HH:MM:SS text.
5. Adds primary key / lookup indices on tables for efficient querying.
"""

from pathlib import Path
import shutil
import sqlite3
import pandas as pd


def build_database(
    project_root: Path,
    db_path: Path | None = None,
) -> dict[str, int]:
    raw_dir = project_root / "data" / "raw"
    clean_dir = project_root / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    if db_path is None:
        db_path = project_root / "data" / "instamart.db"

    # Step 1: Copy products.csv to data/clean/ without altering contents
    raw_products = raw_dir / "products.csv"
    clean_products = clean_dir / "products.csv"
    shutil.copy2(raw_products, clean_products)

    # Step 2: Load the three clean CSV datasets
    df_customers = pd.read_csv(clean_dir / "customers.csv")
    df_products = pd.read_csv(clean_products)
    df_orders = pd.read_csv(clean_dir / "orders.csv")

    # Step 3: Ensure timestamp and date fields are stored in standard ISO string format
    # orders.order_ts -> YYYY-MM-DD HH:MM:SS
    df_orders["order_ts"] = pd.to_datetime(df_orders["order_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    # customers.signup_date -> YYYY-MM-DD
    df_customers["signup_date"] = pd.to_datetime(df_customers["signup_date"]).dt.strftime("%Y-%m-%d")

    # Step 4: Write tables to SQLite database
    conn = sqlite3.connect(db_path)
    try:
        # Write clean dataframes to sqlite tables
        df_customers.to_sql("customers", conn, if_exists="replace", index=False)
        df_products.to_sql("products", conn, if_exists="replace", index=False)
        df_orders.to_sql("orders", conn, if_exists="replace", index=False)

        # Create indices for fast joins and lookups
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cust_id ON customers(customer_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prod_id ON products(product_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ord_id ON orders(order_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ord_cust ON orders(customer_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ord_prod ON orders(product_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ord_status ON orders(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ord_ts ON orders(order_ts);")
        conn.commit()

        # Step 5: Verify table names and counts
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

        counts = {}
        for t in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {t};")
            counts[t] = cursor.fetchone()[0]

        return counts
    finally:
        conn.close()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    db_file = root / "data" / "instamart.db"

    table_counts = build_database(project_root=root, db_path=db_file)

    print("=" * 70)
    print("INSTAMART COPILOT: SQLITE DATABASE BUILD COMPLETE")
    print("=" * 70)
    print(f"Database File: {db_file}")
    for table, count in table_counts.items():
        print(f"  - Table '{table}': {count:,} rows")
    print("=" * 70)

