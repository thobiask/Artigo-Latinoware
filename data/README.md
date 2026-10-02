# Data inputs and inventory

The computational pipeline distributed in this repository uses the exact PM2.5 station-level snapshots stored under `data/pm25/`.

## PM2.5 input snapshots

| Country | PM2.5 records | PM2.5 stations available | Stations entering prediction | Period |
|---|---:|---:|---:|---|
| Brazil | 2,000 | 2 | 2 | 2017–2021 |
| Argentina | 1,439 | 4 | 2 | 2016–2023 |
| Chile | 2,000 | 2 | 2 | 2016–2020 |

PM2.5 was available from eight stations in total, but two Argentine series contained only 30 and 9 observations and did not meet the minimum-length requirement for the predictive train/validation/test pipeline. See `../results/station_split_audit.csv`.

## Broader study dataset

The article reports a processed study inventory of **64,734 records** across PM2.5, PM10, O3, NO2, and CO. The full multi-pollutant snapshots are not duplicated here because the final predictive experiment uses PM2.5 only. For auditability, this directory includes:

- `dataset_inventory.csv` — country-level raw counts and PM2.5 counts;
- `pollutant_inventory.csv` — counts by country and pollutant after quality control;
- `quality_exclusions.csv` — the two non-PM2.5 records excluded during quality control;
- `source_snapshot_hashes.csv` — SHA-256 hashes of the original full country snapshots used to build the study inventory.

The two quality-control exclusions do not alter the PM2.5 input snapshots or any predictive result.
