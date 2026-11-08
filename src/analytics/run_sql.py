"""src/analytics/run_sql.py -- Execute and Validate 7 SQL Queries on data/instamart.db (Part A Task A5)

Reads queries from sql/queries.sql, executes each against data/instamart.db using
pd.read_sql_query, validates outputs against project brief reference checkpoints,
prints formatted results, and saves the full report to outputs/sql_analysis_results.txt.
"""

from pathlib import Path
import re
import sqlite3
import pandas as pd


def parse_queries(sql_file_path: Path) -> dict[str, str]:
    """Parse sql/queries.sql into individual named queries (Q1..Q7)."""
    with open(sql_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on comments matching '-- Q1:', '-- Q2:', etc.
    pattern = r"--\s*(Q[1-7])[:\s—\-]+([^\n]*)"
    parts = re.split(pattern, content)
    # re.split produces: [preamble, q_num1, q_title1, query_body1, q_num2, q_title2, query_body2, ...]

    queries = {}
    i = 1
    while i < len(parts):
        q_id = parts[i].strip()
        q_title = parts[i + 1].strip()
        q_body = parts[i + 2].strip()
        queries[q_id] = (q_title, q_body)
        i += 3

    return queries


def execute_and_validate_sql(db_path: Path, sql_file_path: Path, output_file: Path | None = None) -> str:
    conn = sqlite3.connect(db_path)
    queries = parse_queries(sql_file_path)

    lines = []
    lines.append("=" * 85)
    lines.append("INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT")
    lines.append("TASK A5: PART 1 & 2 - SQL ANALYSIS & VALIDATION REPORT")
    lines.append("=" * 85)
    lines.append(f"Database: {db_path}")
    lines.append(f"Queries File: {sql_file_path}\n")

    validations = []

    # ----------------------------------------------------
    # Q1: WHERE then HAVING
    # ----------------------------------------------------
    title_q1, sql_q1 = queries["Q1"]
    df_q1 = pd.read_sql_query(sql_q1, conn)
    q1_count = len(df_q1)
    q1_pass = (q1_count == 30)
    validations.append(("Q1 (WHERE & HAVING)", "Customer count == 30", q1_count, 30, q1_pass))

    lines.append("-" * 85)
    lines.append(f"Q1: {title_q1}")
    lines.append("-" * 85)
    lines.append(f"Result count: {q1_count} customers (Reference: 30)")
    lines.append("Top 5 customers with spend > 8000 on completed orders > 500:")
    lines.append(df_q1.head().to_string(index=False))
    lines.append("")

    # ----------------------------------------------------
    # Q2: Three-table JOIN
    # ----------------------------------------------------
    title_q2, sql_q2 = queries["Q2"]
    df_q2 = pd.read_sql_query(sql_q2, conn)
    q2_rows = len(df_q2)
    q2_total_rev = round(float(df_q2["revenue"].sum()), 2)
    max_idx = df_q2["revenue"].idxmax()
    max_row = df_q2.loc[max_idx]
    largest_cell = (max_row["city"], max_row["category"], round(float(max_row["revenue"]), 2))

    q2_rows_pass = (q2_rows == 30)
    q2_rev_pass = (abs(q2_total_rev - 2998668.80) < 0.01)
    q2_cell_pass = (largest_cell == ("Bengaluru", "Electronics", 233909.45))

    validations.append(("Q2 (Three-table JOIN)", "Row count == 30", q2_rows, 30, q2_rows_pass))
    validations.append(("Q2 (Three-table JOIN)", "Total Revenue == INR 2,998,668.80", f"INR {q2_total_rev:,.2f}", "INR 2,998,668.80", q2_rev_pass))
    validations.append(("Q2 (Three-table JOIN)", "Largest Cell == BLR / Electronics (233,909.45)", f"{largest_cell[0]} / {largest_cell[1]} ({largest_cell[2]:,.2f})", "Bengaluru / Electronics (233,909.45)", q2_cell_pass))

    lines.append("-" * 85)
    lines.append(f"Q2: {title_q2}")
    lines.append("-" * 85)
    lines.append(f"Result count: {q2_rows} rows | Grand Total: INR {q2_total_rev:,.2f}")
    lines.append(f"Largest Cell: {largest_cell[0]} / {largest_cell[1]} = INR {largest_cell[2]:,.2f}")
    lines.append("\nRevenue by City & Category (First 10 rows):")
    lines.append(df_q2.head(10).to_string(index=False))
    lines.append("")

    # ----------------------------------------------------
    # Q3: Scalar Subquery
    # ----------------------------------------------------
    title_q3, sql_q3 = queries["Q3"]
    df_q3 = pd.read_sql_query(sql_q3, conn)
    q3_count = len(df_q3)
    q3_pass = (q3_count == 11)
    validations.append(("Q3 (Scalar subquery)", "Above-average products == 11", q3_count, 11, q3_pass))

    lines.append("-" * 85)
    lines.append(f"Q3: {title_q3}")
    lines.append("-" * 85)
    lines.append(f"Result count: {q3_count} products (Reference: 11)")
    lines.append("Products priced above catalogue average:")
    lines.append(df_q3.to_string(index=False))
    lines.append("")

    # ----------------------------------------------------
    # Q4: LEFT JOIN
    # ----------------------------------------------------
    title_q4, sql_q4 = queries["Q4"]
    df_q4 = pd.read_sql_query(sql_q4, conn)
    q4_count = len(df_q4)
    q4_pass = (q4_count == 25)
    validations.append(("Q4 (LEFT JOIN)", "Zero-completed-order customers == 25", q4_count, 25, q4_pass))

    lines.append("-" * 85)
    lines.append(f"Q4: {title_q4}")
    lines.append("-" * 85)
    lines.append(f"Result count: {q4_count} customers (Reference: 25)")
    lines.append("First 10 customers with no completed order:")
    lines.append(df_q4.head(10).to_string(index=False))
    lines.append("")

    # ----------------------------------------------------
    # Q5: Common Table Expression (CTE)
    # ----------------------------------------------------
    title_q5, sql_q5 = queries["Q5"]
    df_q5 = pd.read_sql_query(sql_q5, conn)
    q5_count = len(df_q5)
    q5_avg_val = round(float(df_q5["benchmark_avg"].iloc[0]), 2) if q5_count > 0 else 0.0

    q5_count_pass = (q5_count == 550)
    q5_avg_pass = (abs(q5_avg_val - 2552.06) < 0.02)
    validations.append(("Q5 (CTE Benchmark)", "Above-average customers == 550", q5_count, 550, q5_count_pass))
    validations.append(("Q5 (CTE Benchmark)", "Average customer spend == INR 2,552.06", f"INR {q5_avg_val:,.2f}", "INR 2,552.06", q5_avg_pass))

    lines.append("-" * 85)
    lines.append(f"Q5: {title_q5}")
    lines.append("-" * 85)
    lines.append(f"Result count: {q5_count} customers (Reference: 550)")
    lines.append(f"Average Benchmark Completed Spend: INR {q5_avg_val:,.2f} (Reference: INR 2,552.06)")
    lines.append("Top 5 above-average spenders:")
    lines.append(df_q5.head().to_string(index=False))
    lines.append("")

    # ----------------------------------------------------
    # Q6: Ranking Window Functions
    # ----------------------------------------------------
    title_q6, sql_q6 = queries["Q6"]
    df_q6 = pd.read_sql_query(sql_q6, conn)
    top_per_city = (
        df_q6[df_q6["row_num"] == 1]
        .sort_values(by="city")[["city", "customer_id", "total_spend"]]
        .to_dict(orient="records")
    )
    expected_top = [
        {"city": "Bengaluru", "customer_id": "C0082", "total_spend": 12856.20},
        {"city": "Chennai", "customer_id": "C0930", "total_spend": 10151.40},
        {"city": "Delhi", "customer_id": "C0291", "total_spend": 13742.60},
        {"city": "Hyderabad", "customer_id": "C0743", "total_spend": 12730.05},
        {"city": "Mumbai", "customer_id": "C0693", "total_spend": 14766.55},
        {"city": "Pune", "customer_id": "C0426", "total_spend": 10377.10},
    ]

    top_match = True
    for act, exp in zip(top_per_city, expected_top):
        if act["city"] != exp["city"] or act["customer_id"] != exp["customer_id"] or abs(act["total_spend"] - exp["total_spend"]) > 0.01:
            top_match = False

    diff_rank_count = int((df_q6["rnk"] != df_q6["row_num"]).sum())
    diff_pass = (diff_rank_count == 8)

    validations.append(("Q6 (Ranking Window)", "Top customer per city matches reference", top_match, True, top_match))
    validations.append(("Q6 (Ranking Window)", "Rows where RANK != ROW_NUMBER == 8", diff_rank_count, 8, diff_pass))

    lines.append("-" * 85)
    lines.append(f"Q6: {title_q6}")
    lines.append("-" * 85)
    lines.append("Top customer per city:")
    for r in top_per_city:
        lines.append(f"  - {r['city']:<12}: Customer {r['customer_id']} with spend INR {r['total_spend']:,.2f}")
    lines.append(f"\nRows where RANK() != ROW_NUMBER(): {diff_rank_count} (Reference: 8)")
    lines.append("Reason for difference: Ties in total_spend cause RANK() to produce identical rank numbers and skip subsequent ranks, whereas ROW_NUMBER() arbitrarily breaks ties to assign distinct sequential integers.")
    lines.append("")

    # ----------------------------------------------------
    # Q7: Running Total
    # ----------------------------------------------------
    title_q7, sql_q7 = queries["Q7"]
    df_q7 = pd.read_sql_query(sql_q7, conn)
    q7_days = len(df_q7)
    q7_final_cum = round(float(df_q7["cumulative_revenue"].iloc[-1]), 2)

    q7_days_pass = (q7_days == 183)
    q7_rev_pass = (abs(q7_final_cum - 2998668.80) < 0.01)

    validations.append(("Q7 (Running Total)", "Daily rows == 183", q7_days, 183, q7_days_pass))
    validations.append(("Q7 (Running Total)", "Final cumulative revenue == INR 2,998,668.80", f"INR {q7_final_cum:,.2f}", "INR 2,998,668.80", q7_rev_pass))

    lines.append("-" * 85)
    lines.append(f"Q7: {title_q7}")
    lines.append("-" * 85)
    lines.append(f"Total Daily Rows: {q7_days} (2025-04-01 to 2025-09-30)")
    lines.append(f"Final Cumulative Revenue: INR {q7_final_cum:,.2f} (Reference: INR 2,998,668.80)")
    lines.append("\nFirst 5 days running total:")
    lines.append(df_q7.head().to_string(index=False))
    lines.append("\nLast 5 days running total:")
    lines.append(df_q7.tail().to_string(index=False))
    lines.append("")

    # ----------------------------------------------------
    # Summary of Validations
    # ----------------------------------------------------
    lines.append("=" * 85)
    lines.append("SQL CHECKPOINT VALIDATION SUMMARY")
    lines.append("=" * 85)
    lines.append(f"{'Query & Category':<25} {'Checkpoint Description':<38} {'Status':<10}")
    lines.append("-" * 85)
    all_passed = True
    for q_cat, desc, val, ref, passed in validations:
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        lines.append(f"{q_cat:<25} {desc:<38} {status_str:<10}")

    lines.append("-" * 85)
    if all_passed:
        lines.append("OVERALL STATUS: ALL 10 SQL CHECKPOINTS PASSED PERFECTLY")
    else:
        lines.append("OVERALL STATUS: VALIDATION FAILED")
    lines.append("=" * 85)

    conn.close()

    report = "\n".join(lines) + "\n"

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)

    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    db = root / "data" / "instamart.db"
    sql_file = root / "sql" / "queries.sql"
    out_file = root / "outputs" / "sql_analysis_results.txt"

    report = execute_and_validate_sql(db_path=db, sql_file_path=sql_file, output_file=out_file)
    print(report)

