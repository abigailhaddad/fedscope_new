"""
Subset OPM accessions and separations data for specific agencies (USDA, DOI).
Pulls from HuggingFace, filters, combines both data types, and saves as CSV.
Validates that all expected months are present before saving.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from huggingface_hub import list_datasets, hf_hub_download
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

HF_USERNAME = "abigailhaddad"
OUTPUT_DIR = Path("data/agency_subsets")
LOCAL_CACHE_DIR = Path("data/parquet")  # Check here first before downloading

# Expected date range (inclusive)
START_YEAR_MONTH = (2021, 1)
END_YEAR_MONTH = (2025, 11)

AGENCIES = {
    "AG": "usda",  # Department of Agriculture
    "IN": "doi",   # Department of Interior
}


def get_expected_months() -> set[str]:
    """Generate all expected YYYYMM strings in the date range."""
    expected = set()
    year, month = START_YEAR_MONTH
    end_year, end_month = END_YEAR_MONTH

    while (year, month) <= (end_year, end_month):
        expected.add(f"{year}{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return expected


def get_available_datasets(data_type: str) -> list[str]:
    """Get all available dataset repo IDs for a data type."""
    datasets = list_datasets(author=HF_USERNAME, search=f"opm-federal-{data_type.lower()}")
    return [d.id for d in datasets]


def extract_year_month(repo_id: str) -> str | None:
    """Extract YYYYMM from repo ID like 'user/opm-federal-accessions-202511'."""
    match = re.search(r'-(\d{6})$', repo_id)
    return match.group(1) if match else None


def validate_months(repos: list[str], data_type: str) -> bool:
    """Check that all expected months are present. Returns True if valid."""
    expected = get_expected_months()
    found = set()

    for repo_id in repos:
        ym = extract_year_month(repo_id)
        if ym:
            found.add(ym)

    missing = expected - found

    if missing:
        missing_sorted = sorted(missing)
        print(f"\n  ERROR: Missing {len(missing)} months for {data_type}:")
        for ym in missing_sorted:
            print(f"    - {ym[:4]}-{ym[4:]}")
        return False

    return True


def download_and_filter(repo_id: str, agency_codes: list[str], data_type: str) -> pd.DataFrame:
    """Download a dataset and filter for specific agencies.

    Also filters on personnel_action_effective_date_yyyymm to exclude delayed
    reporting from prior years (~50 rows per file are from exactly 2 years prior).
    We extract the expected year from the repo_id and only include matching years.
    """
    path = hf_hub_download(repo_id=repo_id, filename="data.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df[df["agency_code"].isin(agency_codes)]

    # Extract expected year from repo_id (e.g., "opm-federal-accessions-202503" -> "2025")
    date_col = "personnel_action_effective_date_yyyymm"
    if date_col in df.columns:
        repo_month = repo_id.split("-")[-1]  # e.g., "202503"
        expected_year = repo_month[:4]  # e.g., "2025"
        df = df[df[date_col].str.startswith(expected_year)]

    df["data_type"] = data_type  # Add column to distinguish accessions vs separations
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    agency_codes = list(AGENCIES.keys())
    expected_count = len(get_expected_months())

    print(f"Expected date range: {START_YEAR_MONTH[0]}-{START_YEAR_MONTH[1]:02d} to {END_YEAR_MONTH[0]}-{END_YEAR_MONTH[1]:02d}")
    print(f"Expected months per data type: {expected_count}")

    # First, validate all data types have complete months
    print(f"\n{'='*60}")
    print("VALIDATING DATA AVAILABILITY")
    print(f"{'='*60}")

    all_repos = {}
    for data_type in ["accessions", "separations"]:
        repos = get_available_datasets(data_type)
        all_repos[data_type] = repos
        print(f"\n  {data_type}: found {len(repos)} datasets")

        if not validate_months(repos, data_type):
            print(f"\n  ABORTING: Missing months detected. Please run the download script first.")
            sys.exit(1)

        print(f"  {data_type}: all {expected_count} months present")

    # Collect data per agency across both data types
    agency_dfs = {code: [] for code in agency_codes}

    for data_type in ["accessions", "separations"]:
        print(f"\n{'='*60}")
        print(f"Processing {data_type.upper()}")
        print(f"{'='*60}")

        repos = all_repos[data_type]

        for repo_id in tqdm(repos, desc=f"Downloading {data_type}"):
            try:
                df = download_and_filter(repo_id, agency_codes, data_type)
                for code in agency_codes:
                    agency_df = df[df["agency_code"] == code]
                    if len(agency_df) > 0:
                        agency_dfs[code].append(agency_df)
            except Exception as e:
                print(f"  Error with {repo_id}: {e}")
                continue

    # Save combined data for each agency
    print(f"\n{'='*60}")
    print("SAVING COMBINED FILES")
    print(f"{'='*60}")

    for code, name in AGENCIES.items():
        if not agency_dfs[code]:
            print(f"  No data for {name.upper()}")
            continue

        combined = pd.concat(agency_dfs[code], ignore_index=True)
        filename = f"{name}_accessions_separations.csv"
        filepath = OUTPUT_DIR / filename
        combined.to_csv(filepath, index=False)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        accessions_count = len(combined[combined["data_type"] == "accessions"])
        separations_count = len(combined[combined["data_type"] == "separations"])

        print(f"\n  {name.upper()} ({code}):")
        print(f"    File: {filepath}")
        print(f"    Accessions rows: {accessions_count:,}")
        print(f"    Separations rows: {separations_count:,}")
        print(f"    Total rows: {len(combined):,}")
        print(f"    Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
