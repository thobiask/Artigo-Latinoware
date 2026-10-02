"""Reproduce the FDTE TinyML computational proof of concept.

Run from the repository root after installing ``requirements.txt``.
The script regenerates the three quantized models and the reported CSV/JSON
artifacts from the processed data included in the repository.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
import tensorflow as tf

SEED = 42
FEATURES = [
    "pm25",
    "pm25_lag1",
    "pm25_lag2",
    "pm25_lag3",
    "pm25_mean3",
    "pm25_std3",
    "pm25_baseline_mean24",
    "pm25_baseline_std24",
    "hour",
    "month",
    "day_of_week",
]
COUNTRY_FILES = {
    "Brazil": "brazil_pm25.csv.gz",
    "Argentina": "argentina_pm25.csv.gz",
    "Chile": "chile_pm25.csv.gz",
}
COUNTRY_SLUG = {"Brazil": "brazil", "Argentina": "argentina", "Chile": "chile"}


def set_seeds() -> None:
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)


def load_country_snapshot(pm25_dir: Path, filename: str) -> pd.DataFrame:
    """Load the exact PM2.5 input snapshot used by the predictive pipeline."""
    return pd.read_csv(pm25_dir / filename)


def pm25_rows(df: pd.DataFrame) -> pd.DataFrame:
    pollutant = df["poluente"].astype(str).str.lower().str.replace(".", "", regex=False)
    data = df[pollutant.eq("pm25")].copy()
    data["data_hora"] = pd.to_datetime(data["data_hora"], utc=True, errors="coerce")
    data = data.rename(columns={"valor": "pm25"}).dropna(
        subset=["data_hora", "pm25", "estacao"]
    )
    return data[data.pm25 >= 0]


def contextual_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, station_data in pm25_rows(df).groupby("estacao"):
        station_data = station_data.sort_values("data_hora").copy()
        station_data["mean24"] = (
            station_data.pm25.shift(1).rolling(24, min_periods=8).mean()
        )
        station_data["std24"] = (
            station_data.pm25.shift(1).rolling(24, min_periods=8).std()
        )
        station_data["z"] = (
            (station_data.pm25 - station_data.mean24).abs()
            / station_data.std24.replace(0, np.nan)
        )
        station_data["flag"] = (station_data.z >= 2.5).astype(int)
        rows.append(station_data)
    return pd.concat(rows, ignore_index=True).dropna(subset=["z"])


def prepare_temporal(
    df: pd.DataFrame, country: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    train: list[pd.DataFrame] = []
    validation: list[pd.DataFrame] = []
    test: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []

    for station, station_data in pm25_rows(df).groupby("estacao"):
        station_data = station_data.sort_values("data_hora").copy()
        raw_count = len(station_data)
        if raw_count < 40:
            continue

        station_data["hour"] = station_data.data_hora.dt.hour
        station_data["month"] = station_data.data_hora.dt.month
        station_data["day_of_week"] = station_data.data_hora.dt.dayofweek
        station_data["pm25_lag1"] = station_data.pm25.shift(1)
        station_data["pm25_lag2"] = station_data.pm25.shift(2)
        station_data["pm25_lag3"] = station_data.pm25.shift(3)
        station_data["pm25_mean3"] = (
            station_data.pm25.shift(1).rolling(3).mean()
        )
        station_data["pm25_std3"] = (
            station_data.pm25.shift(1).rolling(3).std()
        )
        station_data["pm25_baseline_mean24"] = (
            station_data.pm25.shift(1).rolling(24, min_periods=8).mean()
        )
        station_data["pm25_baseline_std24"] = (
            station_data.pm25.shift(1).rolling(24, min_periods=8).std()
        )

        future_max = pd.concat(
            [
                station_data.pm25.shift(-1),
                station_data.pm25.shift(-2),
                station_data.pm25.shift(-3),
            ],
            axis=1,
        ).max(axis=1)
        station_data["target_3h"] = (future_max > 25).astype(float)
        station_data.loc[station_data.index[-3:], "target_3h"] = np.nan
        station_data = station_data.replace([np.inf, -np.inf], np.nan).dropna(
            subset=FEATURES + ["target_3h"]
        )

        split_70 = int(len(station_data) * 0.70)
        split_80 = int(len(station_data) * 0.80)
        station_train = station_data.iloc[:split_70]
        station_validation = station_data.iloc[split_70:split_80]
        station_test = station_data.iloc[split_80:]
        train.append(station_train)
        validation.append(station_validation)
        test.append(station_test)

        audit.append(
            {
                "country": country,
                "station": station,
                "raw_pm25_records": raw_count,
                "usable_records": len(station_data),
                "start_utc": station_data.data_hora.min().isoformat(),
                "end_utc": station_data.data_hora.max().isoformat(),
                "train_n": len(station_train),
                "validation_n": len(station_validation),
                "test_n": len(station_test),
            }
        )

    if not train:
        raise ValueError(f"No usable PM2.5 station series for {country}")
    return (
        pd.concat(train),
        pd.concat(validation),
        pd.concat(test),
        audit,
    )


def convert_int8(model: tf.keras.Model, X_train: np.ndarray) -> bytes:
    def representative_dataset():
        for index in range(min(200, len(X_train))):
            yield [X_train[index : index + 1].astype("float32")]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def write_decision_and_sensitivity_results(result_dir: Path) -> None:
    scenarios = {
        "Brazil / State-level consolidated network": np.array([2, 0, 2, 2, 2]),
        "Argentina / Municipal intermediate network": np.array([1, 0, 1, 2, 1]),
        "Chile / Remote stations in a federal network": np.array([0, 0, 0, 2, 1]),
    }
    nominal_rows = []
    recommendation = {8: "Cloud", 5: "Hybrid", 3: "Edge AI"}
    for scenario, scores in scenarios.items():
        total = int(scores.sum())
        nominal_rows.append(
            {
                "scenario": scenario,
                "D1": int(scores[0]),
                "D2": int(scores[1]),
                "D3": int(scores[2]),
                "D4": int(scores[3]),
                "D5": int(scores[4]),
                "total": total,
                "recommendation": recommendation[total],
            }
        )
    pd.DataFrame(nominal_rows).to_csv(
        result_dir / "fdte_scenarios.csv", index=False
    )

    weight_rng = np.random.default_rng(SEED)
    weights = weight_rng.dirichlet(np.ones(5), 100_000)
    weight_rows = []
    for scenario, scores in scenarios.items():
        weighted_score = (weights @ scores) * 5
        classes = np.where(
            weighted_score <= 3,
            "Edge AI",
            np.where(weighted_score <= 6, "Hybrid", "Cloud"),
        )
        weight_rows.append(
            {
                "scenario": scenario,
                "edge_ai_percent": round((classes == "Edge AI").mean() * 100, 1),
                "hybrid_percent": round((classes == "Hybrid").mean() * 100, 1),
                "cloud_percent": round((classes == "Cloud").mean() * 100, 1),
            }
        )
    pd.DataFrame(weight_rows).to_csv(
        result_dir / "fdte_weight_sensitivity.csv", index=False
    )

    base_tco = {
        "Brazil": (24_800, 258_000),
        "Argentina": (22_320, 253_800),
        "Chile": (35_800, 480_000),
    }
    base_rows = []
    for scenario, (edge, cloud) in base_tco.items():
        base_rows.append(
            {
                "scenario": scenario,
                "edge_brl": edge,
                "cloud_brl": cloud,
                "base_savings_percent": round((1 - edge / cloud) * 100, 1),
            }
        )
    pd.DataFrame(base_rows).to_csv(
        result_dir / "tco_base_scenarios.csv", index=False
    )

    # A separate fixed seed makes the published one-decimal intervals stable
    # and independent of the preceding FDTE-weight simulation.
    tco_rng = np.random.default_rng(2)
    tco_rows = []
    for scenario, (edge, cloud) in base_tco.items():
        edge_mc = edge * tco_rng.uniform(0.8, 1.2, 100_000)
        cloud_mc = cloud * tco_rng.uniform(0.8, 1.2, 100_000)
        savings = (1 - edge_mc / cloud_mc) * 100
        tco_rows.append(
            {
                "scenario": scenario,
                "edge_brl": edge,
                "cloud_brl": cloud,
                "base_savings_percent": round((1 - edge / cloud) * 100, 1),
                "p2_5_savings_percent": round(np.percentile(savings, 2.5), 1),
                "p97_5_savings_percent": round(np.percentile(savings, 97.5), 1),
                "edge_cheaper_percent": round((edge_mc < cloud_mc).mean() * 100, 1),
            }
        )
    pd.DataFrame(tco_rows).to_csv(
        result_dir / "tco_sensitivity.csv", index=False
    )


def main() -> None:
    set_seeds()
    root = Path(__file__).resolve().parents[1]
    pm25_dir = root / "data" / "pm25"
    model_dir = root / "models"
    result_dir = root / "results"
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    performance: list[dict[str, object]] = []
    contextual: list[dict[str, object]] = []
    station_audit: list[dict[str, object]] = []
    preprocessing: dict[str, object] = {"feature_order": FEATURES, "countries": {}}
    model_metadata: dict[str, object] = {
        "task": "Predict whether at least one of the next three hourly PM2.5 observations exceeds 25 µg/m³",
        "architecture": "Dense(16, ReLU) -> Dense(8, ReLU) -> Dense(1, sigmoid)",
        "training": {
            "epochs": 50,
            "optimizer": "Adam",
            "loss": "binary_crossentropy",
            "split": "chronological 70/10/20 by station",
            "seed": SEED,
        },
        "quantization": "TensorFlow Lite full integer INT8",
        "reported_metrics_note": (
            "Reported classification metrics are computed on the held-out test split "
            "using the trained Keras model before TFLite conversion. The included "
            "INT8 files are the corresponding quantized exports; physical ESP32 "
            "validation remains future work."
        ),
        "models": {},
    }

    for country, filename in COUNTRY_FILES.items():
        df = load_country_snapshot(pm25_dir, filename)
        contextual_rows = contextual_summary(df)
        contextual.append(
            {
                "country": country,
                "valid_observations": len(contextual_rows),
                "contextual_anomalies": int(contextual_rows.flag.sum()),
                "prevalence_percent": round(contextual_rows.flag.mean() * 100, 2),
            }
        )

        train, validation, test, audit = prepare_temporal(df, country)
        station_audit.extend(audit)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURES]).astype("float32")
        X_validation = scaler.transform(validation[FEATURES]).astype("float32")
        X_test = scaler.transform(test[FEATURES]).astype("float32")
        y_train = train.target_3h.to_numpy("float32")
        y_validation = validation.target_3h.to_numpy("float32")
        y_test = test.target_3h.to_numpy("float32")

        positives = y_train.sum()
        negatives = len(y_train) - positives
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input((len(FEATURES),)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            verbose=0,
            class_weight={0: 1.0, 1: float(negatives / max(positives, 1))},
        )

        validation_probability = model.predict(X_validation, verbose=0).ravel()
        thresholds = np.linspace(0.05, 0.95, 181)
        threshold = float(
            thresholds[
                int(
                    np.argmax(
                        [
                            f1_score(
                                y_validation,
                                validation_probability >= candidate,
                                zero_division=0,
                            )
                            for candidate in thresholds
                        ]
                    )
                )
            ]
        )
        test_probability = model.predict(X_test, verbose=0).ravel()
        prediction = (test_probability >= threshold).astype(int)
        blob = convert_int8(model, X_train)
        filename_tflite = f"{COUNTRY_SLUG[country]}_pm25_prediction_3h.tflite"
        (model_dir / filename_tflite).write_bytes(blob)

        metrics = {
            "country": country,
            "train_n": len(train),
            "validation_n": len(validation),
            "test_n": len(test),
            "test_positive_pct": float(y_test.mean() * 100),
            "threshold": threshold,
            "accuracy_pct": float(accuracy_score(y_test, prediction) * 100),
            "precision_pct": float(precision_score(y_test, prediction, zero_division=0) * 100),
            "recall_pct": float(recall_score(y_test, prediction, zero_division=0) * 100),
            "f1_pct": float(f1_score(y_test, prediction, zero_division=0) * 100),
            "auc_roc": float(roc_auc_score(y_test, test_probability)),
            "size_kb": float(len(blob) / 1024),
        }
        performance.append(metrics)

        preprocessing["countries"][country] = {
            "scaler_mean": dict(zip(FEATURES, map(float, scaler.mean_))),
            "scaler_scale": dict(zip(FEATURES, map(float, scaler.scale_))),
            "decision_threshold": threshold,
            "split_counts": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
            "test_positive_percent": float(y_test.mean() * 100),
        }
        model_metadata["models"][country] = {
            "file": filename_tflite,
            "decision_threshold": threshold,
            "test_metrics": {
                "accuracy_percent": metrics["accuracy_pct"],
                "precision_percent": metrics["precision_pct"],
                "recall_percent": metrics["recall_pct"],
                "f1_percent": metrics["f1_pct"],
                "auc_roc": metrics["auc_roc"],
                "test_positive_percent": metrics["test_positive_pct"],
            },
            "file_size_kb": metrics["size_kb"],
        }

    pd.DataFrame(contextual).to_csv(
        result_dir / "contextual_zscore_summary.csv", index=False
    )
    pd.DataFrame(performance).to_csv(
        result_dir / "model_performance_temporal.csv", index=False
    )
    pd.DataFrame(station_audit).to_csv(
        result_dir / "station_split_audit.csv", index=False
    )
    (model_dir / "preprocessing_parameters.json").write_text(
        json.dumps(preprocessing, indent=2), encoding="utf-8"
    )
    (model_dir / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2), encoding="utf-8"
    )
    write_decision_and_sensitivity_results(result_dir)

    print(pd.DataFrame(performance).to_string(index=False))


if __name__ == "__main__":
    main()
