"""
Download OPM workforce data, convert to parquet, upload to Hugging Face.
Each monthly file becomes its own HF dataset with original OPM naming.
"""

import asyncio
import os
from pathlib import Path
import pandas as pd
from playwright.async_api import async_playwright
from huggingface_hub import HfApi, create_repo, repo_exists, list_repo_files
import time
import argparse
import re
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = "abigailhaddad"
START_DATE = "2021-01-01"
END_DATE = "2025-11-30"
DATA_TYPES = ["Accessions", "Separations", "Employment"]
DOWNLOAD_DIR = Path("data/downloads")
PARQUET_DIR = Path("data/parquet")

SIZE_ESTIMATES = {
    "Accessions": 6,
    "Separations": 6,
    "Employment": 780,
}


def get_repo_name_from_filename(filename: str) -> str:
    """
    Extract repo name from OPM filename or display name.
    Example: accessions_202511_1_2026-01-09.csv -> opm-federal-accessions-202511
    Example: Accessions data from November 2025 -> opm-federal-accessions-202511
    """
    # Check if it's the new display format: "Accessions data from November 2025"
    if " data from " in filename:
        import calendar
        month_map = {name.lower(): num for num, name in enumerate(calendar.month_name) if num}

        parts = filename.split(" data from ")
        data_type = parts[0].lower()  # accessions, separations, employment
        date_part = parts[1].strip()  # "November 2025"

        # Parse "November 2025" -> 202511
        date_parts = date_part.split()
        if len(date_parts) == 2:
            month_name = date_parts[0].lower()
            year = date_parts[1]
            month_num = month_map.get(month_name, 0)
            if month_num:
                return f"opm-federal-{data_type}-{year}{month_num:02d}"

    # Original format: {type}_{YYYYMM}_{version}_{date}.csv
    parts = filename.replace('.csv', '').replace('.parquet', '').split('_')
    if len(parts) >= 2:
        data_type = parts[0]  # accessions, separations, employment
        year_month = parts[1]  # 202511
        return f"opm-federal-{data_type}-{year_month}"
    return f"opm-federal-{filename.split('.')[0]}"


def is_already_uploaded(repo_id: str, token: str) -> bool:
    """Check if a dataset already exists on HuggingFace with data."""
    try:
        if not repo_exists(repo_id, repo_type="dataset", token=token):
            return False
        files = list_repo_files(repo_id, repo_type="dataset", token=token)
        return "data.parquet" in files
    except:
        return False


async def get_card_filename(page, card_index: int) -> str:
    """Extract the filename from a card's download button label."""
    buttons = page.locator('button[aria-label^="Download options for"]')
    button = buttons.nth(card_index)
    label = await button.get_attribute('aria-label')
    # "Download options for accessions_202511_1_2026-01-09" -> "accessions_202511_1_2026-01-09"
    if label and label.startswith("Download options for "):
        return label.replace("Download options for ", "").strip()
    return ""


async def setup_page(playwright):
    """Launch browser and navigate to OPM data downloads page."""
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(accept_downloads=True)
    page = await context.new_page()

    print("🌐 Navigating to OPM data downloads page...")
    await page.goto("https://data.opm.gov/explore-data/data/data-downloads")
    await page.wait_for_load_state("networkidle")
    await asyncio.sleep(2)

    return browser, context, page


async def set_filters(page, data_type: str, start_date: str, end_date: str):
    """Set the date range and data type filters."""
    start_input = page.locator('input[aria-label="Select start date"]')
    await start_input.fill(start_date)
    await start_input.press('Enter')
    await asyncio.sleep(1)

    end_input = page.locator('input[aria-label="Select end date"]')
    await end_input.fill(end_date)
    await end_input.press('Enter')
    await asyncio.sleep(1)

    dropdown = page.locator('#data-sources')
    await dropdown.select_option(data_type)
    await asyncio.sleep(2)

    try:
        count_locator = page.locator('p').filter(has_text=re.compile(r'\d+-\d+ of \d+'))
        count_text = await count_locator.first.text_content()
        match = re.search(r'of (\d+)', count_text)
        total = int(match.group(1)) if match else 0
    except:
        total = 0

    return total


async def download_file_from_card(page, card_index: int, download_dir: Path) -> Path:
    """Download a single file by clicking its download button."""
    buttons = page.locator('button[aria-label^="Download options for"]')
    button = buttons.nth(card_index)

    await button.click()
    await asyncio.sleep(0.3)

    csv_option = page.get_by_label("CSV", exact=False).first

    async with page.expect_download(timeout=600000) as download_info:
        await csv_option.click(force=True)

    download = await download_info.value
    dest_path = download_dir / download.suggested_filename
    await download.save_as(dest_path)

    await page.keyboard.press('Escape')
    await asyncio.sleep(0.3)

    return dest_path


def convert_to_parquet(csv_path: Path, parquet_dir: Path) -> Path:
    """Convert CSV to parquet format with zstd compression."""
    # Read with low_memory=False and all columns as strings to avoid type issues
    df = pd.read_csv(csv_path, delimiter='|', low_memory=False, dtype=str)
    parquet_name = csv_path.stem + ".parquet"
    parquet_path = parquet_dir / parquet_name
    df.to_parquet(parquet_path, compression='zstd', index=False)
    return parquet_path


def upload_to_huggingface(parquet_path: Path, repo_id: str, token: str, max_retries: int = 3):
    """Upload a parquet file to Hugging Face with retry logic."""
    api = HfApi()

    # Create repo if needed
    try:
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    except:
        pass

    # Upload with retries
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=str(parquet_path),
                path_in_repo="data.parquet",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            return repo_id
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                time.sleep(wait_time)
            else:
                raise e
    return repo_id


async def download_and_upload_all(page, data_type: str, download_dir: Path, parquet_dir: Path,
                                   start_date: str, end_date: str, token: str):
    """Download all files for a data type, uploading each immediately."""
    print(f"\n{'='*60}")
    print(f"📥 {data_type.upper()}")
    print(f"{'='*60}")

    total = await set_filters(page, data_type, start_date, end_date)
    if total == 0:
        print("  No files found, skipping...")
        return [], []

    print(f"  Found {total} files")

    # Set to 100 items per page
    try:
        rows_dropdown = page.locator('select').filter(has_text='10')
        await rows_dropdown.select_option('100')
        await asyncio.sleep(2)
    except:
        pass

    uploaded_repos = []
    failed_files = []
    pbar = tqdm(total=total, desc=f"  {data_type}", unit="file")

    page_num = 1
    while True:
        buttons = page.locator('button[aria-label^="Download options for"]')
        count = await buttons.count()

        for i in range(count):
            card_filename = None
            try:
                # Check if already uploaded before downloading
                card_filename = await get_card_filename(page, i)
                if card_filename:
                    repo_name = get_repo_name_from_filename(card_filename)
                    repo_id = f"{HF_USERNAME}/{repo_name}"
                    if is_already_uploaded(repo_id, token):
                        pbar.set_postfix({"status": "skipped (exists)"})
                        pbar.update(1)
                        uploaded_repos.append(repo_id)
                        continue

                # Download
                csv_path = await download_file_from_card(page, i, download_dir)
                csv_size = csv_path.stat().st_size / (1024 * 1024)

                # Convert
                parquet_path = convert_to_parquet(csv_path, parquet_dir)
                parquet_size = parquet_path.stat().st_size / (1024 * 1024)

                # Get repo name from filename
                repo_name = get_repo_name_from_filename(csv_path.name)
                repo_id = f"{HF_USERNAME}/{repo_name}"

                pbar.set_postfix({
                    "repo": repo_name[-25:],
                    "size": f"{csv_size:.0f}→{parquet_size:.1f}MB"
                })

                # Upload (now with retries built-in)
                upload_to_huggingface(parquet_path, repo_id, token)
                uploaded_repos.append(repo_id)

                # Cleanup
                csv_path.unlink()
                parquet_path.unlink()

                pbar.update(1)
                await asyncio.sleep(0.3)

            except Exception as e:
                error_msg = str(e)[:60]
                pbar.write(f"  ⚠️ Error: {error_msg}")
                if card_filename:
                    failed_files.append({"filename": card_filename, "error": error_msg})
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
                pbar.update(1)
                continue

        next_button = page.locator('button[aria-label="Go to next page"]')
        if await next_button.is_disabled():
            break

        await next_button.click()
        await asyncio.sleep(2)
        page_num += 1

    pbar.close()
    print(f"  ✅ Uploaded {len(uploaded_repos)} datasets")
    if failed_files:
        print(f"  ❌ Failed {len(failed_files)} files")
    return uploaded_repos, failed_files


async def main():
    parser = argparse.ArgumentParser(description="Download OPM data and upload to HuggingFace")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HuggingFace token")
    parser.add_argument("--start", default=START_DATE, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=END_DATE, help="End date (YYYY-MM-DD)")
    parser.add_argument("--types", nargs="+", default=DATA_TYPES, help="Data types to download")
    args = parser.parse_args()

    if not args.token:
        print("❌ Error: HF_TOKEN environment variable or --token required")
        return

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("🚀 OPM Data → HuggingFace (one dataset per month)")
    print("="*60)
    print(f"📅 Date range: {args.start} to {args.end}")
    print(f"📊 Data types: {', '.join(args.types)}")

    # Estimates
    print("\n📊 Estimated totals:")
    for dtype in args.types:
        months = 59
        est_csv = SIZE_ESTIMATES.get(dtype, 10) * months
        est_parquet = est_csv * 0.04
        print(f"  {dtype}: {months} datasets, ~{est_parquet:.0f} MB total parquet")

    all_repos = []
    all_failures = []

    async with async_playwright() as playwright:
        browser, context, page = await setup_page(playwright)

        try:
            for data_type in args.types:
                repos, failures = await download_and_upload_all(
                    page, data_type, DOWNLOAD_DIR, PARQUET_DIR,
                    args.start, args.end, args.token
                )
                all_repos.extend(repos)
                all_failures.extend(failures)

        finally:
            await browser.close()

    print("\n" + "="*60)
    print("🎉 DONE!")
    print("="*60)
    print(f"Created {len(all_repos)} HuggingFace datasets")

    if all_failures:
        print(f"\n⚠️ {len(all_failures)} files failed:")
        for f in all_failures:
            print(f"  - {f['filename']}: {f['error']}")
        print("\nRe-run the script to retry failed files.")

    print(f"\nExample datasets:")
    for repo in all_repos[:3]:
        print(f"  📁 https://huggingface.co/datasets/{repo}")
    if len(all_repos) > 3:
        print(f"  ... and {len(all_repos) - 3} more")


if __name__ == "__main__":
    asyncio.run(main())
