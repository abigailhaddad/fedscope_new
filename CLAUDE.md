# OPM Federal Workforce Data Pipeline

This repo downloads OPM (Office of Personnel Management) federal workforce data and uploads it to HuggingFace for public access.

## What This Does

Downloads three types of federal workforce data from https://data.opm.gov/explore-data/data/data-downloads:
- **Accessions**: New federal hires (~6 MB/month CSV, ~0.2 MB parquet)
- **Separations**: Federal employee departures (~6 MB/month CSV, ~0.2 MB parquet)
- **Employment**: Full federal workforce snapshot (~780 MB/month CSV, ~30 MB parquet)

Each monthly file becomes its own HuggingFace dataset named like `opm-federal-accessions-202511`.

## Key Files

- `download_and_upload.py` - Main script using Playwright to automate OPM site, converts CSV to parquet, uploads to HuggingFace
- `data/downloads/` - Temporary CSV storage (deleted after upload)
- `data/parquet/` - Temporary parquet storage (deleted after upload)

## Technical Details

- OPM files are **pipe-delimited** (`|`), not comma-separated
- All columns read as strings (`dtype=str`) to avoid mixed-type issues
- Parquet uses zstd compression (~96% size reduction)
- Script has resume logic: checks HuggingFace before downloading, skips existing datasets
- Uses Playwright because OPM site is a Blazor app with no direct download URLs

## Running

```bash
export HF_TOKEN=your_token_here
python download_and_upload.py
```

Options:
- `--start 2021-01-01` - Start date
- `--end 2025-11-30` - End date
- `--types Accessions Separations Employment` - Which data types

## HuggingFace Username

Currently hardcoded to `abigailhaddad` in the script (line 18).
