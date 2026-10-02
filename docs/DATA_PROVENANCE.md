# Data provenance

The measurements used in the study were retrieved through OpenAQ API v3 from providers associated with public air-quality monitoring systems in Brazil, Argentina, and Chile.

The broader processed study inventory contains **64,734 records** after two explicit non-PM2.5 quality-control exclusions. It covers PM2.5, PM10, O3, NO2, and CO. Counts by country and pollutant are preserved in `data/pollutant_inventory.csv`, and the exclusions are listed in `data/quality_exclusions.csv`.

For reproducibility of the final computational proof of concept, this repository distributes the **exact PM2.5 subsets** used by the contextual z-score and predictive pipeline under `data/pm25/`. PM2.5 is available from eight stations; six contain enough observations to enter the predictive split. Therefore, the reported predictive results must not be interpreted as estimates for complete national monitoring networks.

`data/source_snapshot_hashes.csv` records SHA-256 hashes and row counts of the original full country snapshots used to construct the study inventory. The larger multi-pollutant snapshots are not duplicated in the camera-ready repository because the final predictive experiment does not use them.

OpenAQ documentation:

- https://docs.openaq.org/about/about
- https://docs.openaq.org/resources/licenses
