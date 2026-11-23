# Proyecto 6 — Pipeline local secuencial con EfficientNet-B0

Implementación del experimento 1 del proyecto **Partición de un pipeline de visión entre un equipo local y un servidor**. Clasifica 500 imágenes de Imagenette con EfficientNet-B0 preentrenada, sin entrenamiento ni ajuste fino, y mide rendimiento, recursos y calidad de clasificación.

## Diseño del experimento

- Modelo: `torchvision.models.efficientnet_b0` con pesos `IMAGENET1K_V1`.
- Dataset: Imagenette 320 px, únicamente el split `val`.
- Muestra: 50 imágenes de cada una de las 10 clases, semilla 42.
- Ejecución: CPU, batch size 1 y procesamiento estrictamente secuencial.
- Repeticiones: cinco corridas de 500 imágenes, precedidas por 20 inferencias de calentamiento.
- Entorno: Python 3.11 dentro de Docker Linux.

La predicción se obtiene sobre las 1000 categorías de ImageNet-1K. Las predicciones externas a las diez clases de Imagenette no se reasignan.

## Requisitos

- Windows 10/11 con Docker Desktop en modo de contenedores Linux.
- Docker Compose v2.
- Aproximadamente 2 GB libres para imagen, dataset, pesos y resultados.

No es necesario instalar Python en Windows.

## Carpetas persistentes

| Anfitrión | Contenedor | Uso durante el benchmark |
|---|---|---|
| `./data/imagenette` | `/app/data/imagenette` | Solo lectura |
| `./results` | `/app/results` | Lectura y escritura |
| `./model_cache` | `/app/model_cache` | Solo lectura |

Son bind mounts: los resultados aparecen directamente en Windows y no se necesita `docker cp`.

## Ejecución

Abra PowerShell en la raíz del repositorio y asegúrese de que Docker Desktop esté iniciado.

### 1. Construir la imagen

```powershell
docker compose build
```

La construcción instala las ruedas oficiales de PyTorch 2.8.0 y Torchvision 0.23.0 exclusivas para CPU.

### 2. Preparar dataset, pesos y manifiesto

```powershell
docker compose run --rm prepare
```

Este servicio reutiliza Imagenette si ya está completo, descarga los pesos en `model_cache`, comprueba el mapeo de categorías y crea `results/experiment1/dataset_manifest.csv` con 50 imágenes por clase, dimensiones, tamaño y SHA-256.

Si el manifiesto ya existe, se valida íntegramente y no se genera una selección nueva. Una estructura incompleta o un hash distinto produce un error explícito; no se eliminan archivos automáticamente.

### 3. Prueba rápida

```powershell
docker compose run --rm experiment1-quick
```

Procesa una imagen de cada clase y guarda la salida en `results/experiment1_quick`. Utiliza el manifiesto definitivo sin modificarlo.

Para repetir la prueba sobre resultados existentes:

```powershell
docker compose run --rm experiment1-quick python -m experiments.experiment1_local --num-images 10 --runs 1 --warmup 2 --output-subdir experiment1_quick --overwrite
```

### 4. Experimento completo

```powershell
docker compose run --rm experiment1
```

El servicio no tiene acceso de red, monta dataset y pesos en modo solo lectura y ejecuta cinco corridas. No sobrescribe corridas existentes de manera implícita.

### 5. Pruebas automatizadas

```powershell
docker compose run --rm test
```

Las pruebas cubren el mapeo de categorías, la salida de 1000 logits, la selección equilibrada, seguridad de rutas, percentiles, throughput, persistencia de CSV/JSON y ausencia de solapamiento entre imágenes.

## Resultados

```text
results/experiment1/
├── dataset_manifest.csv
├── config.json
├── system_info.json
├── run_01/ ... run_05/
│   ├── predictions.csv
│   ├── metrics.json
│   └── resources.csv
├── summary/
│   ├── summary_runs.csv
│   ├── summary_metrics.json
│   └── technical_summary.md
└── figures/
    └── 10 gráficos PNG
```

Los tiempos se miden con `time.perf_counter_ns()`. El reloj total rodea directamente el bucle de imágenes; descarga, carga del modelo, calentamiento, escritura y gráficos quedan fuera. El tiempo de carga del modelo se registra en `system_info.json`.

Un hilo auxiliar muestrea CPU y RSS del proceso, pero no procesa imágenes. El porcentaje de CPU puede superar 100 % cuando PyTorch utiliza varios núcleos. El RSS del proceso no equivale al consumo total del contenedor.

## Configuración opcional

El número de hilos internos de PyTorch puede fijarse sin introducir concurrencia entre imágenes:

```powershell
docker compose run --rm experiment1 python -m experiments.experiment1_local --torch-threads 4
```

Para ver todas las opciones:

```powershell
docker compose run --rm experiment1 python -m experiments.experiment1_local --help
```

## Consideraciones de interpretación

- La precisión de referencia de EfficientNet-B0 corresponde a ImageNet-1K, no a este subconjunto.
- Imagenette deriva de ImageNet; no es una prueba independiente de generalización.
- Las 500 imágenes se repiten para rendimiento. Las 2,500 inferencias no son observaciones independientes de calidad.
- Información del anfitrión o de Docker no visible desde el contenedor se registra como no disponible.

Fuentes: [EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html), [Imagenette](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Imagenette.html) y [bind mounts de Docker](https://docs.docker.com/engine/storage/bind-mounts/).
