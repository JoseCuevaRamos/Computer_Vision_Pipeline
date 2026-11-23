from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    predictions_by_run: list[pd.DataFrame],
    resources_by_run: list[pd.DataFrame],
    summary: pd.DataFrame,
    first_metrics: dict[str, Any],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(predictions_by_run, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(combined["total_latency_ms"], bins=30, edgecolor="black", alpha=0.8)
    ax.set(title="Distribución de latencias por imagen", xlabel="Latencia total (ms)", ylabel="Frecuencia")
    _save(fig, figures_dir / "01_latency_histogram.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = [frame["total_latency_ms"].to_numpy() for frame in predictions_by_run]
    ax.boxplot(grouped, tick_labels=[f"run_{index:02d}" for index in range(1, len(grouped) + 1)])
    ax.set(title="Latencia por ejecución", xlabel="Ejecución", ylabel="Latencia total (ms)")
    _save(fig, figures_dir / "02_latency_boxplot.png")

    stage_columns = ["read_ms", "preprocess_ms", "inference_ms", "postprocess_ms"]
    stage_labels = ["Lectura", "Preprocesamiento", "Inferencia", "Posprocesamiento"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(stage_labels, [combined[column].mean() for column in stage_columns])
    ax.set(title="Tiempo promedio por etapa", xlabel="Etapa", ylabel="Tiempo promedio (ms)")
    _save(fig, figures_dir / "03_average_stage_time.png")

    run_labels = summary["run_id"].tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(run_labels, summary["total_seconds"])
    ax.set(title="Tiempo total por ejecución", xlabel="Ejecución", ylabel="Tiempo total (s)")
    _save(fig, figures_dir / "04_total_time_by_run.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(run_labels, summary["throughput_images_per_second"])
    ax.set(title="Throughput por ejecución", xlabel="Ejecución", ylabel="Imágenes/s")
    _save(fig, figures_dir / "05_throughput_by_run.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, frame in enumerate(resources_by_run, start=1):
        ax.plot(frame["elapsed_seconds"], frame["process_cpu_percent"], label=f"run_{index:02d}")
    ax.set(title="Utilización de CPU del proceso", xlabel="Tiempo (s)", ylabel="CPU del proceso (%)")
    ax.legend()
    _save(fig, figures_dir / "06_cpu_over_time.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, frame in enumerate(resources_by_run, start=1):
        ax.plot(frame["elapsed_seconds"], frame["process_rss_mb"], label=f"run_{index:02d}")
    ax.set(title="Memoria RSS del proceso", xlabel="Tiempo (s)", ylabel="RSS (MB)")
    ax.legend()
    _save(fig, figures_dir / "07_ram_over_time.png")

    classification = first_metrics["classification"]
    per_class = classification["per_class"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(list(per_class), [values["accuracy"] for values in per_class.values()])
    ax.set(title="Accuracy por categoría", xlabel="Categoría", ylabel="Accuracy", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=35)
    _save(fig, figures_dir / "08_accuracy_by_class.png")

    matrix = np.asarray(classification["confusion_matrix"], dtype=int)
    row_labels = classification["confusion_row_labels"]
    column_labels = classification["confusion_column_labels"]
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Imágenes")
    ax.set_xticks(range(len(column_labels)), column_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set(title="Matriz de confusión", xlabel="Predicción", ylabel="Clase real")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=7)
    _save(fig, figures_dir / "09_confusion_matrix.png")

    positions = np.arange(len(summary))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions - width / 2, summary["latency_mean_ms"], width, label="Media")
    ax.bar(positions + width / 2, summary["latency_p95_ms"], width, label="p95")
    ax.set_xticks(positions, run_labels)
    ax.set(title="Latencia media y p95", xlabel="Ejecución", ylabel="Latencia (ms)")
    ax.legend()
    _save(fig, figures_dir / "10_mean_p95_comparison.png")

