# Verification Record

The repository package was verified before release in the following environment:

- Python 3.13.5
- TensorFlow CPU 2.20.0
- NumPy 2.5.1
- pandas 2.2.3
- scikit-learn 1.8.0

## Tests performed

1. ZIP integrity and complete manifest validation.
2. Python syntax compilation for the script and every notebook code cell.
3. End-to-end execution of `python src/fdte_experiment.py`.
4. End-to-end sequential execution of all notebook code cells.
5. Regeneration of the three INT8 TFLite files.
6. SHA-256 comparison between regenerated and distributed TFLite files.
7. TFLite interpreter loading, tensor allocation, and one inference invocation per model.
8. CSV schema, row-count, station-count, and reported-metric consistency checks.
9. Anonymous-content scan for author names, email addresses, local user paths, and placeholders.

## Verified model hashes

```text
brazil_pm25_prediction_3h.tflite
SHA-256 5e8385bfd72789e59cf985834519629a847ec0afd976e6dbe641f97e4c9b7f0c

argentina_pm25_prediction_3h.tflite
SHA-256 d0ccef6863ee5132a967463ac9fc7a30cf1e849964019facdad3162c93dcdca7

chile_pm25_prediction_3h.tflite
SHA-256 b4d32cc8cac97663aa46fd04c6abc23016c87cf71912b9fb95c42994dd1a26f6
```

The successful software test does not constitute physical ESP32 validation. Hardware latency, tensor-arena memory, power use, sensor integration, and long-term stability remain outside the current proof of concept.
