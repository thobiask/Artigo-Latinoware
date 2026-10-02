# FDTE TinyML - Latin.Science 2026 Reproducibility Repository

Reproducibility artifacts for the accepted Latin.Science 2026 paper:

> **When the Cloud Is Too Far Away: A Proposed Multicriteria Framework for Edge AI in Public Air Quality Monitoring in Latin America**

The **Techno-Economic Decision Framework (FDTE)** supports an auditable choice among Edge AI, hybrid, and cloud architectures for public air-quality monitoring under unequal infrastructure conditions. The matrix considers connectivity, five-year total cost of ownership, operational latency, data governance, and local technical capacity.

## What this repository reproduces

The computational proof of concept is intentionally narrow and mirrors the paper:

- current contextual deviations are detected with a transparent z-score rule;
- machine learning is used only for the non-trivial prediction task;
- the target is whether at least one of the next three hourly PM2.5 observations exceeds the operational threshold of 25 µg/m³;
- train/validation/test splits are chronological (70/10/20) within each usable station;
- the earlier multigas experiment is excluded because pollutant timestamps were not sufficiently simultaneous across all three countries;
- FDTE weight sensitivity and five-year TCO sensitivity are reproduced from fixed random seeds.

## Repository layout

```text
data/       exact PM2.5 inputs plus broader-dataset inventory and provenance
notebook/   Google Colab/Jupyter entry point
src/        canonical Python implementation
models/     three INT8 TFLite exports and preprocessing metadata
results/    metrics, station audit, FDTE scores and sensitivity analyses
docs/       methodology, provenance, references and verification record
```

## Main predictive results

| Country | Test positive rate | Accuracy | Precision | Recall | F1 | AUC-ROC | INT8 file |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brazil | 29.55% | 92.93% | 90.09% | 85.47% | 87.72% | 0.9701 | 3.28 KB |
| Argentina | 4.35% | 85.51% | 15.00% | 50.00% | 23.08% | 0.8033 | 3.28 KB |
| Chile | 48.99% | 69.95% | 75.86% | 56.70% | 64.90% | 0.7555 | 3.28 KB |

The Argentine result is intentionally retained. Its low precision makes the current model unsuitable as an autonomous alert system.

## Reproduction

### Google Colab

1. Clone or download this repository.
2. Open `notebook/FDTE_TinyML_LatinScience2026_FINAL.ipynb` in Google Colab.
3. Run all cells.
4. Generated models are written to `models/`; generated reports are written to `results/`.

### Local execution

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/fdte_experiment.py
```

Reference environment used for verification:

- Python 3.13.5
- TensorFlow 2.20.0
- NumPy 2.5.1
- pandas 2.2.3
- scikit-learn 1.8.0

The executable pipeline stores the exact PM2.5 station-level snapshots used by the final experiment under `data/pm25/`. The broader study inventory contains 64,734 records across five pollutants after two documented non-PM2.5 quality-control exclusions; its counts and source-snapshot hashes are preserved in `data/` without duplicating the unused multi-pollutant files.

## Interpretation boundaries

- The 25 µg/m³ value is an operational threshold inspired by the WHO 24-hour Interim Target 4; an isolated hourly observation is not treated as a regulatory 24-hour exceedance.
- The contextual z-score identifies deviation from a station's recent local pattern; it does not certify a public-health episode.
- TFLite file size demonstrates storage compatibility only. ESP32 latency, tensor-arena memory, energy consumption, sensor integration and long-term stability were not measured in the accepted study.
- FDTE scores are documentary scenario assessments, not official diagnoses of agencies or entire countries.

## Next research stage

Following the reviewers' recommendations, the planned extension is to combine institutional calibration with physical field validation: deploy ESP32-based monitoring nodes with calibrated particulate sensors, measure latency, memory, energy use, availability and predictive behavior under real conditions, and compare observed deployment/maintenance costs with the TCO assumptions used here. A Delphi-style consultation with public managers is also planned to recalibrate FDTE criteria, weights and decision ranges.

## Licenses and provenance

- Code and model files: MIT License.
- Data snapshots: subject to the licenses and terms of the original data providers exposed through OpenAQ. See `DATA_LICENSE.md` and `docs/DATA_PROVENANCE.md`.
- Traceable bibliography: `docs/TRACEABLE_REFERENCES.md`.

Repository: https://github.com/thobiask/Artigo-Latinoware
