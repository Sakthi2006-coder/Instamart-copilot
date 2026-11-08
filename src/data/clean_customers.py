"""src/data/clean_customers.py -- Customer Data Cleaning Pipeline (Part A Task A2)

Cleans and standardizes raw customer records:
- Standardizes city names (stripping whitespace, title case)
- Maps gender variations to canonical categories: 'male', 'female', 'unknown'
- Cleans and imputes missing / out-of-bounds ages using city median
- Converts signup_date to datetime
- Validates clean data against reference checkpoints
- Writes cleaned dataset to data/clean/customers.csv
"""

import hashlib
from pathlib import Path
import pandas as pd
import numpy as np


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash to verify file immutability."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_customers(raw_path: Path, clean_path: Path | None = None) -> pd.DataFrame:
    """Load raw customers.csv, apply standardization and imputation rules, and optionally save."""
    # Compute initial hash of raw file to guarantee immutability
    raw_hash_before = compute_file_hash(raw_path)

    # 1. Load raw dataset without modifying the source
    df_raw = pd.read_csv(raw_path, keep_default_na=False)
    df = df_raw.copy()

    # 2. Standardize city: strip whitespace, title case
    df["city"] = df["city"].astype(str).str.strip().str.title()

    # 3. Standardize gender: male / female / unknown
    def map_gender(val: str) -> str:
        s = str(val).strip().lower()
        if s in ("m", "male"):
            return "male"
        elif s in ("f", "female"):
            return "female"
        return "unknown"

    df["gender"] = df["gender"].apply(map_gender)

    # 4. Clean age: numeric conversion, out-of-bounds [10, 100] to NaN
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < 10) | (df["age"] > 100), "age"] = np.nan

    # 5. Impute missing ages using city median via groupby + transform
    city_medians = df.groupby("city")["age"].transform("median")
    df["age"] = df["age"].fillna(city_medians)

    # 6. Convert signup_date to datetime
    df["signup_date"] = pd.to_datetime(df["signup_date"])

    # Verify raw file was not modified
    raw_hash_after = compute_file_hash(raw_path)
    assert raw_hash_before == raw_hash_after, "FATAL: Raw customers.csv was modified during processing!"

    # 7. Save cleaned dataset
    if clean_path:
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        # Write clean CSV; format signup_date as YYYY-MM-DD
        df.to_csv(clean_path, index=False, date_format="%Y-%m-%d")

    return df


def validate_cleaned_customers(df: pd.DataFrame, raw_path: Path) -> dict[str, bool]:
    """Validate all Task A2 acceptance criteria and reference checkpoints."""
    results = {}

    # Check 1: Exactly 1,200 rows
    results["Exact 1,200 rows"] = (len(df) == 1200)

    # Check 2: Exactly six standardized cities
    canonical_cities = {"Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune"}
    cities_in_df = set(df["city"].unique())
    results["Exactly 6 standardized cities"] = (cities_in_df == canonical_cities)

    # Check 3: City counts match reference
    expected_city_counts = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    }
    actual_city_counts = df["city"].value_counts().to_dict()
    results["City distribution matches reference"] = (actual_city_counts == expected_city_counts)

    # Check 4: Gender counts match reference
    expected_gender_counts = {
        "female": 610,
        "male": 560,
        "unknown": 30,
    }
    actual_gender_counts = df["gender"].value_counts().to_dict()
    results["Gender distribution matches reference"] = (actual_gender_counts == expected_gender_counts)

    # Check 5: Zero missing ages
    missing_ages = int(df["age"].isna().sum())
    results["Zero missing ages"] = (missing_ages == 0)

    # Check 6: All ages between 10 and 100 (and between 18 and 60)
    all_ages_in_range = bool(((df["age"] >= 18) & (df["age"] <= 60)).all())
    results["All ages between 18 and 60"] = all_ages_in_range

    # Check 7: signup_date is datetime dtype
    results["signup_date is datetime"] = pd.api.types.is_datetime64_any_dtype(df["signup_date"])

    return results


def print_validation_report(df: pd.DataFrame, validations: dict[str, bool]):
    print("=" * 78)
    print("INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT")
    print("TASK A2: CUSTOMER CLEANING VALIDATION REPORT")
    print("=" * 78)

    print("\n1. SUMMARY METRICS:")
    print(f"  - Total Cleaned Rows     : {len(df)}")
    print(f"  - Total Columns          : {len(df.columns)} ({list(df.columns)})")
    print(f"  - Missing Values (Total) : {df.isna().sum().sum()}")
    print(f"  - Age Range              : Min = {df['age'].min():.1f}, Max = {df['age'].max():.1f}, Mean = {df['age'].mean():.2f}")
    print(f"  - Date Range             : {df['signup_date'].min().strftime('%Y-%m-%d')} to {df['signup_date'].max().strftime('%Y-%m-%d')}")

    print("\n2. STANDARDIZED CITY COUNTS:")
    for city, count in df["city"].value_counts().items():
        print(f"  - {city:<12}: {count:>4}")

    print("\n3. STANDARDIZED GENDER COUNTS:")
    for gender, count in df["gender"].value_counts().items():
        print(f"  - {gender:<12}: {count:>4}")

    print("\n4. VALIDATION CHECKPOINTS:")
    print(f"{'Checkpoint Description':<50} {'Status':<10}")
    print("-" * 65)
    all_passed = True
    for desc, passed in validations.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"{desc:<50} {status:<10}")

    print("-" * 65)
    if all_passed:
        print("OVERALL RESULT: ALL VALIDATION CHECKPOINTS PASSED PERFECTLY")
    else:
        print("OVERALL RESULT: VALIDATION FAILED")
    print("=" * 78)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "data" / "raw" / "customers.csv"
    clean_csv = project_root / "data" / "clean" / "customers.csv"

    df_cleaned = clean_customers(raw_path=raw_csv, clean_path=clean_csv)
    validations = validate_cleaned_customers(df_cleaned, raw_path=raw_csv)
    print_validation_report(df_cleaned, validations)

