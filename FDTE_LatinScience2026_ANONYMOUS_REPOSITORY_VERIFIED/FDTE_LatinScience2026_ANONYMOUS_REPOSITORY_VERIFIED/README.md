# FDTE TinyML - Anonymous Supplementary Repository

Anonymous supplementary material for a Latin.Science 2026 submission.

## Scope

The **Techno-Economic Decision Framework (FDTE)** supports an auditable choice among Edge AI, hybrid, and cloud architectures for public air-quality monitoring under unequal infrastructure conditions. The decision matrix considers connectivity, five-year total cost of ownership, operational latency, data governance, and local technical capacity.

The computational proof of concept is intentionally narrow:

- current contextual deviations are detected by a transparent z-score rule;
- machine learning is used only for the non-trivial prediction task;
- the target is whether at least one of the next three hourly PM2.5 observations exceeds the operational threshold of 25 µg/m³;
- train/validation/test splits are chronological (70/10/20) within each station;
- the earlier multigas experiment is not included because pollutant timestamps were not sufficiently simultaneous across all three countries.

## Repository layout

```text
notebook/   reproducible Google Colab/Jupyter notebook
src/        script version of the computational pipeline
data/       raw and processed CSV files used in the study
models/     three INT8 TFLite files plus preprocessing metadata
results/    reported metrics, station audit, FDTE and TCO sensitivity
docs/       methodology notes, data provenance, and traceable references
```

## Main predictive results

| Country | Test positive rate | Accuracy | Precision | Recall | F1 | AUC-ROC | INT8 file |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brazil | 29.55% | 92.93% | 90.09% | 85.47% | 87.72% | 0.9701 | 3.28 KB |
| Argentina | 4.35% | 85.51% | 15.00% | 50.00% | 23.08% | 0.8033 | 3.28 KB |
| Chile | 48.99% | 69.95% | 75.86% | 56.70% | 64.90% | 0.7555 | 3.28 KB |

The Argentine result is retained for transparency. Its low precision makes the current model unsuitable as an autonomous alert system.

## Reproduction

### Google Colab

1. Download or clone this repository.
2. Open `notebook/FDTE_TinyML_LatinScience2026_FINAL.ipynb` in Google Colab.
3. Run all cells.
4. Generated models are written to `models/`; generated reports are written to `results/`.

The notebook automatically detects the repository layout. No manual upload is required when the repository folder is mounted or uploaded to Colab.

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

The verified reference environment is pinned in `requirements.txt`:

- Python 3.13.5
- TensorFlow 2.20.0
- NumPy 2.5.1
- pandas 2.2.3
- scikit-learn 1.8.0

In this environment, the script and notebook regenerate the reported metrics and the exact three TFLite binaries included in `models/`. Other library versions or hardware may introduce small floating-point differences.

## Important interpretation notes

- The 25 µg/m³ value is an operational threshold inspired by the WHO 24-hour Interim Target 4; an isolated hourly observation is not treated as a regulatory 24-hour exceedance.
- The contextual z-score identifies deviation from a station's recent local pattern; it does not certify a public-health episode.
- TFLite file size demonstrates storage compatibility only. ESP32 latency, tensor-arena memory, energy use, and physical stability were not measured in this study.
- FDTE scores are documentary scenario assessments, not official diagnoses of the agencies or entire countries.

## Licenses and data provenance

- Code and model files: MIT License.
- Data: redistributed for academic audit subject to the licenses and terms of the original OpenAQ data providers. See `DATA_LICENSE.md` and `docs/DATA_PROVENANCE.md`.
- This repository intentionally omits author identity during double-blind review.
