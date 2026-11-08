"""src/data/validate_part_a.py -- Verification & Checkpoint Assertions for Part A

Asserts all reference checkpoints from Tasks 3, 4, and 5 of the project brief:
- Exactly 1,200 customers
- Exactly 40 products
- Exactly 12,309 orders
- Unique order_id count == 12,309
- Exactly 11,110 completed orders
- Exactly 1,199 cancelled orders
- Exactly 0 pending orders
- Exactly 0 missing order_amount
- Exactly 0 missing delivery_minutes for completed orders
- Exactly 1,199 missing delivery_minutes for cancelled orders
- Completed order_amount sum equals INR 2,998,668.80
- Customer city distribution (Pune: 227, Delhi: 210, Bengaluru: 197, Chennai: 191, Hyderabad: 191, Mumbai: 184)
- Customer gender distribution (female: 610, male: 560, unknown: 30)
- Zero missing ages, all ages between 18 and 60
- order_ts string format is YYYY-MM-DD HH:MM:SS

Prints exactly 'ALL CHECKPOINTS PASSED' when all assertions succeed.
"""

from pathlib import Path
import re
import sqlite3
import pandas as pd


def validate_database_and_clean_data(project_root: Path) -> bool:
    db_path = project_root / "data" / "instamart.db"
    assert db_path.exists(), f"Database not found at {db_path}"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check 1: Tables existence
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [r[0] for r in cursor.fetchall() if not r[0].startswith("sqlite_")]
        assert set(tables) >= {"customers", "orders", "products"}, f"Missing tables: {tables}"

        # Check 2: 1,200 customers
        cursor.execute("SELECT COUNT(*) FROM customers;")
        cust_count = cursor.fetchone()[0]
        assert cust_count == 1200, f"Expected 1,200 customers, got {cust_count}"

        # Check 3: 40 products
        cursor.execute("SELECT COUNT(*) FROM products;")
        prod_count = cursor.fetchone()[0]
        assert prod_count == 40, f"Expected 40 products, got {prod_count}"

        # Check 4: 12,309 orders
        cursor.execute("SELECT COUNT(*) FROM orders;")
        ord_count = cursor.fetchone()[0]
        assert ord_count == 12309, f"Expected 12,309 orders, got {ord_count}"

        # Check 5: Unique order_ids
        cursor.execute("SELECT COUNT(DISTINCT order_id) FROM orders;")
        dist_ord_count = cursor.fetchone()[0]
        assert dist_ord_count == 12309, f"Expected 12,309 distinct order_ids, got {dist_ord_count}"

        # Check 6: 11,110 completed orders
        cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'completed';")
        comp_count = cursor.fetchone()[0]
        assert comp_count == 11110, f"Expected 11,110 completed orders, got {comp_count}"

        # Check 7: 1,199 cancelled orders
        cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'cancelled';")
        canc_count = cursor.fetchone()[0]
        assert canc_count == 1199, f"Expected 1,199 cancelled orders, got {canc_count}"

        # Check 8: 0 pending orders
        cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'pending';")
        pend_count = cursor.fetchone()[0]
        assert pend_count == 0, f"Expected 0 pending orders, got {pend_count}"

        # Check 9: 0 missing order_amount
        cursor.execute("SELECT COUNT(*) FROM orders WHERE order_amount IS NULL OR order_amount = '';")
        miss_amt = cursor.fetchone()[0]
        assert miss_amt == 0, f"Expected 0 missing order amounts, got {miss_amt}"

        # Check 10: 0 missing delivery_minutes for completed orders
        cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'completed' AND (delivery_minutes IS NULL OR delivery_minutes = '');")
        miss_deliv_comp = cursor.fetchone()[0]
        assert miss_deliv_comp == 0, f"Expected 0 missing delivery_minutes for completed, got {miss_deliv_comp}"

        # Check 11: 1,199 missing delivery_minutes for cancelled orders
        cursor.execute("SELECT COUNT(*) FROM orders WHERE status = 'cancelled' AND (delivery_minutes IS NULL OR delivery_minutes = '');")
        miss_deliv_canc = cursor.fetchone()[0]
        assert miss_deliv_canc == 1199, f"Expected 1,199 missing delivery_minutes for cancelled, got {miss_deliv_canc}"

        # Check 12: Completed revenue sum == INR 2,998,668.80
        cursor.execute("SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status = 'completed';")
        rev_sum = cursor.fetchone()[0]
        assert abs(rev_sum - 2998668.80) < 1e-2, f"Expected revenue INR 2,998,668.80, got {rev_sum}"

        # Check 13: Customer city counts
        cursor.execute("SELECT city, COUNT(*) FROM customers GROUP BY city;")
        city_counts = dict(cursor.fetchall())
        expected_cities = {
            "Pune": 227,
            "Delhi": 210,
            "Bengaluru": 197,
            "Chennai": 191,
            "Hyderabad": 191,
            "Mumbai": 184,
        }
        assert city_counts == expected_cities, f"City distribution mismatch: {city_counts} vs {expected_cities}"

        # Check 14: Customer gender counts
        cursor.execute("SELECT gender, COUNT(*) FROM customers GROUP BY gender;")
        gender_counts = dict(cursor.fetchall())
        expected_genders = {"female": 610, "male": 560, "unknown": 30}
        assert gender_counts == expected_genders, f"Gender distribution mismatch: {gender_counts} vs {expected_genders}"

        # Check 15: Ages valid (0 missing, between 18 and 60)
        cursor.execute("SELECT COUNT(*), MIN(age), MAX(age) FROM customers WHERE age IS NULL OR age < 18 OR age > 60;")
        invalid_ages = cursor.fetchone()[0]
        assert invalid_ages == 0, f"Found {invalid_ages} invalid or out-of-range customer ages"

        # Check 16: order_ts formatted as YYYY-MM-DD HH:MM:SS text
        cursor.execute("SELECT order_ts FROM orders LIMIT 100;")
        ts_sample = [r[0] for r in cursor.fetchall()]
        ts_regex = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
        for ts in ts_sample:
            assert ts_regex.match(ts), f"order_ts '{ts}' does not match YYYY-MM-DD HH:MM:SS"

        return True

    finally:
        conn.close()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    all_ok = validate_database_and_clean_data(project_root)
    if all_ok:
        print("ALL CHECKPOINTS PASSED")

