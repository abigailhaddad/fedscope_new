# OPM Federal Workforce Data

> **Note:** This was put together quickly and may have errors. Please verify anything important against the original source at [data.opm.gov](https://data.opm.gov).

OPM already has a bunch of premade visualizations on [data.opm.gov](https://data.opm.gov) that cover common questions about the federal workforce - workforce size, demographics, separations, etc. If one of those answers your question, use that.

This repo has some of the raw data (accessions, separations, employment) for custom analysis:

1. **HuggingFace** - Parquet files you can grab directly or query with DuckDB
2. **Colab notebook** - Load and explore multiple months of data without downloading anything

## Data on HuggingFace

Datasets are public at [huggingface.co/abigailhaddad](https://huggingface.co/abigailhaddad):

- **Accessions** (new hires): Jan 2021 - Nov 2025
- **Separations** (departures): Jan 2021 - Nov 2025
- **Employment** (workforce snapshots): Jan 2021 - Nov 2025

Each monthly file is its own dataset, named like `opm-federal-accessions-202511`.

You can query them directly with DuckDB without downloading:

```python
import duckdb

url = "https://huggingface.co/datasets/abigailhaddad/opm-federal-accessions-202511/resolve/main/data.parquet"
df = duckdb.execute(f"SELECT * FROM read_parquet('{url}')").df()
```

## Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abigailhaddad/fedscope_new/blob/main/demo.ipynb)

Click the badge above to open the notebook. It loads all accessions and separations data into DuckDB and lets you:
- See hiring vs attrition trends over time
- Filter by agency, state, occupation, etc.
- Download CSVs of your filtered results

No auth required, no local setup.

## Other Resources

- [OPM Visualization Catalog](https://newfedscope.netlify.app/) - Searchable index of OPM's built-in dashboards and what filters/variables each one supports

## Status

- Accessions: 5 years (Jan 2021 - Nov 2025)
- Separations: 5 years (Jan 2021 - Nov 2025)
- Employment: 5 years (Jan 2021 - Nov 2025)
