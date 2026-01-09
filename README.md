# OPM Federal Workforce Data

OPM already has a bunch of premade visualizations on [data.opm.gov](https://data.opm.gov) that cover common questions about the federal workforce - workforce size, demographics, separations, etc. If one of those answers your question, use that.

This repo is for when you need the raw data for custom analysis:

1. **HuggingFace** - I've uploaded the data as parquet files so you can grab them directly or query with DuckDB
2. **Colab notebook** - Load and explore multiple months of data without downloading anything

## Data on HuggingFace

All datasets are public at [huggingface.co/abigailhaddad](https://huggingface.co/abigailhaddad):

- **Accessions** (new hires): Jan 2021 - Nov 2025
- **Separations** (departures): Jan 2021 - Nov 2025
- **Employment** (workforce snapshots): *in progress*

Each monthly file is its own dataset, named like `opm-federal-accessions-202511`.

You can query them directly with DuckDB without downloading:

```python
import duckdb

url = "https://huggingface.co/datasets/abigailhaddad/opm-federal-accessions-202511/resolve/main/data.parquet"
df = duckdb.execute(f"SELECT * FROM read_parquet('{url}')").df()
```

## Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abigailhaddad/opm/blob/main/demo.ipynb)

Click the badge above to open the notebook. It loads all accessions and separations data into DuckDB and lets you:
- See hiring vs attrition trends over time
- Filter by agency, state, occupation, etc.
- Download CSVs of your filtered results

No auth required, no local setup.

## Status

- Accessions: 5 years done
- Separations: 5 years done
- Employment: working on it (these files are ~780MB each so it takes a while)
