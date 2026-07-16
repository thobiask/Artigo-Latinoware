# Data files

The CSV files are the exact raw and processed records used by the computational proof of concept. Column names are preserved from the original processing pipeline.

## PM2.5 subset used for prediction

| Country | Total processed records | PM2.5 records | PM2.5 stations | Period |
|---|---:|---:|---:|---|
| Brazil | 29,000 | 2,000 | 2 | 2017-2021 |
| Argentina | 15,729 | 1,439 | 4 | 2016-2023 |
| Chile | 20,005 | 2,000 | 2 | 2016-2020 |

The full processed dataset contains 29 stations. PM2.5 records are available from eight stations, but only six stations meet the minimum-length requirement and enter the predictive train/validation/test pipeline. Two Argentine series contain only 30 and 9 records and are excluded from model training. See `../results/station_split_audit.csv`.
