"""src/data/audit.py -- Raw Data Audit for Instamart Copilot (Part A Task A1)

Loads raw customers, products, and orders datasets, computes dataset profiles,
missing values, data types, and specific quality checkpoints, and compares
against reference values from the project brief.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def run_audit(data_dir: Path, output_file: Path | None = None) -> str:
    lines = []
    lines.append("=" * 85)
    lines.append("INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT")
    lines.append("TASK A1: RAW DATA AUDIT REPORT")
    lines.append("=" * 85)
    lines.append("")

    # Load raw files
    # keep_default_na=False allows exact inspection of literal blanks and string flags
    cust_raw = pd.read_csv(data_dir / "customers.csv", keep_default_na=False)
    prod_raw = pd.read_csv(data_dir / "products.csv", keep_default_na=False)
    orders_raw = pd.read_csv(data_dir / "orders.csv", keep_default_na=False)

    datasets = {
        "customers.csv": cust_raw,
        "products.csv": prod_raw,
        "orders.csv": orders_raw,
    }

    # 1. Dataset Profiles (Shape, Dtypes, Missing values)
    lines.append("1. DATASET PROFILES (SHAPE, DATA TYPES & MISSING/BLANK VALUES)")
    lines.append("-" * 85)

    for name, df in datasets.items():
        lines.append(f"\n--- {name} (Shape: {df.shape[0]} rows, {df.shape[1]} columns) ---")
        lines.append(f"{'Column':<20} {'Raw Dtype':<15} {'Missing/Blank':<15} {'Missing %':<12} {'Notes'}")
        lines.append("-" * 85)
        for col in df.columns:
            s = df[col]
            blank_cnt = (s == "").sum() + s.isna().sum()
            blank_pct = (blank_cnt / len(df)) * 100
            note = ""
            if name == "orders.csv" and col == "order_amount":
                non_num = pd.to_numeric(s, errors="coerce").isna().sum()
                note = f"{non_num} non-numeric values ('N/A' or 'error')"
            elif name == "orders.csv" and col == "delivery_minutes":
                neg_cnt = (pd.to_numeric(s, errors="coerce") < 0).sum()
                note = f"{neg_cnt} negative values (< 0)"
            elif name == "customers.csv" and col == "age":
                num_age = pd.to_numeric(s, errors="coerce")
                out_bounds = ((num_age < 10) | (num_age > 100)).sum()
                note = f"{out_bounds} out-of-bounds (10-100)"
            lines.append(f"{col:<20} {str(s.dtype):<15} {blank_cnt:<15} {blank_pct:<12.2f}% {note}")

    lines.append("")
    lines.append("=" * 85)
    lines.append("2. SPECIFIC AUDIT METRICS & REFERENCE CHECKPOINT COMPARISON")
    lines.append("=" * 85)

    # Orders audit metrics
    orders_total_rows = len(orders_raw)
    orders_exact_dups = orders_raw.duplicated().sum()
    orders_pending_rows = (orders_raw["status"] == "pending").sum()
    orders_distinct_id = orders_raw["order_id"].nunique()

    # order_amount non-numeric
    numeric_amount = pd.to_numeric(orders_raw["order_amount"], errors="coerce")
    orders_non_num_amount = numeric_amount.isna().sum()

    # delivery_minutes
    numeric_deliv = pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce")
    orders_neg_deliv = (numeric_deliv < 0).sum()
    orders_blank_deliv = (orders_raw["delivery_minutes"] == "").sum() + orders_raw["delivery_minutes"].isna().sum()

    # Customers audit metrics
    cust_blank_age = (cust_raw["age"] == "").sum() + cust_raw["age"].isna().sum()
    numeric_age = pd.to_numeric(cust_raw["age"], errors="coerce")
    cust_age_outside = ((numeric_age < 10) | (numeric_age > 100)).sum()
    cust_city_spellings = cust_raw["city"].nunique()
    non_blank_gender_series = cust_raw[cust_raw["gender"] != ""]["gender"]
    cust_gender_spellings = non_blank_gender_series.nunique()
    cust_blank_gender = (cust_raw["gender"] == "").sum() + cust_raw["gender"].isna().sum()

    metrics = [
        # (Table, Metric description, Computed value, Reference checkpoint)
        ("ORDERS", "Raw orders.csv rows", orders_total_rows, 13084),
        ("ORDERS", "Exact duplicate rows", orders_exact_dups, 119),
        ("ORDERS", "Rows with pending status", orders_pending_rows, 656),
        ("ORDERS", "Distinct order_id count", orders_distinct_id, 12309),
        ("ORDERS", "Non-numeric order_amount entries", orders_non_num_amount, 120),
        ("ORDERS", "Negative delivery_minutes (< 0)", orders_neg_deliv, 106),
        ("ORDERS", "Blank delivery_minutes", orders_blank_deliv, 2541),
        ("CUSTOMERS", "Blank age values", cust_blank_age, 61),
        ("CUSTOMERS", "Age values outside 10-100", cust_age_outside, 6),
        ("CUSTOMERS", "Distinct raw city spellings", cust_city_spellings, 24),
        ("CUSTOMERS", "Distinct non-blank raw gender spellings", cust_gender_spellings, 10),
        ("CUSTOMERS", "Blank gender values", cust_blank_gender, 30),
    ]

    lines.append(f"{'Category':<12} {'Metric Description':<40} {'Computed':<10} {'Reference':<10} {'Status':<10}")
    lines.append("-" * 85)

    all_match = True
    for cat, desc, val, ref in metrics:
        match = (val == ref)
        if not match:
            all_match = False
        status_str = "PASS" if match else "MISMATCH"
        lines.append(f"{cat:<12} {desc:<40} {val:<10} {ref:<10} {status_str:<10}")

    lines.append("-" * 85)
    if all_match:
        lines.append("OVERALL AUDIT STATUS: ALL 12 REFERENCE CHECKPOINTS PASSED PERFECTLY")
    else:
        lines.append("OVERALL AUDIT STATUS: WARNING - MISMATCH DETECTED")

    # Additional contextual details
    lines.append("\n" + "=" * 85)
    lines.append("3. DETAILED ANOMALY BREAKDOWN")
    lines.append("=" * 85)
    lines.append("Customers City Spellings Found (24 total variations across 6 canonical metro cities):")
    cities_found = sorted(cust_raw["city"].unique().tolist())
    lines.append("  " + ", ".join(repr(c) for c in cities_found))

    lines.append("\nCustomers Gender Spellings Found (10 non-blank variations):")
    genders_found = sorted(non_blank_gender_series.unique().tolist())
    lines.append("  " + ", ".join(repr(g) for g in genders_found))

    lines.append("\nCustomers Out-of-Bounds Age Values (6 planted impossible ages):")
    bad_ages = cust_raw.loc[(numeric_age < 10) | (numeric_age > 100), ["customer_id", "age"]].to_dict(orient="records")
    lines.append("  " + str(bad_ages))

    lines.append("\nOrders Planted Non-Numeric Amount Types:")
    non_num_vals = orders_raw.loc[numeric_amount.isna(), "order_amount"].value_counts().to_dict()
    lines.append("  " + str(non_num_vals))

    report = "\n".join(lines) + "\n"

    # Save to outputs
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)

    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "raw"
    output_path = project_root / "outputs" / "raw_data_audit.txt"

    report = run_audit(data_dir=data_dir, output_file=output_path)
    print(report)
