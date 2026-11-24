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

---

## Experimento 2 — Distribuido con PNG

El experimento 2 conserva exactamente el manifiesto de 500 imágenes del experimento 1. La PC cliente convierte cada imagen a RGB y la codifica como PNG sin pérdida; la PC servidor recibe el PNG, aplica las transformaciones oficiales de `IMAGENET1K_V1` y ejecuta EfficientNet-B0. El cliente no contiene PyTorch.

La comunicación prevista es:

```text
PC cliente (Docker) → HTTPS de ngrok → localhost:8000 → PC servidor (Docker)
```

Ngrok se ejecuta en Windows en la PC servidor, no dentro de Docker. El authtoken de ngrok y la `API_KEY` de FastAPI son credenciales distintas y ninguna debe subirse a Git.

### Preparación de la PC servidor

Copie el ejemplo y reemplace la clave por una cadena segura de al menos 12 caracteres:

```powershell
Copy-Item .env.server.example .env.server
notepad .env.server
```

Construya la imagen y prepare únicamente los pesos; este servicio no descarga Imagenette:

```powershell
docker compose -f compose.server.yaml build
docker compose -f compose.server.yaml run --rm prepare_model
docker compose -f compose.server.yaml up -d server
```

Compruebe `http://localhost:8000/health`. Después, en otra terminal de Windows:

```powershell
ngrok config add-authtoken TU_TOKEN
ngrok http 8000
```

El primer comando de ngrok solo se necesita una vez. Copie la URL HTTPS que muestra el segundo. El servidor publica Docker únicamente en `127.0.0.1:8000`; ngrok es quien crea el túnel externo.

### Preparación de la PC cliente

Esta PC debe tener el dataset y el manifiesto ya generados por el experimento 1. No vuelve a descargar ni vuelve a seleccionar imágenes.

```powershell
Copy-Item .env.client.example .env.client
notepad .env.client
docker compose -f compose.client.yaml build
```

En `.env.client`, configure `SERVER_URL` con la URL HTTPS de ngrok y use exactamente la misma `API_KEY` del servidor.

Ejecute primero diez imágenes, una por clase:

```powershell
docker compose -f compose.client.yaml run --rm client-quick
```

La prueba guarda resultados en `results/experiment2/client/quick_test` y compara automáticamente sus predicciones con `results/experiment1_single_corrected/run_01/predictions.csv` cuando esa línea base está disponible. Para ser rápida, valida estructura, unicidad, tamaños y rutas del manifiesto, pero omite recalcular los 500 SHA-256; la ejecución completa sí verifica todos los hashes.

Después ejecute el experimento completo:

```powershell
docker compose -f compose.client.yaml run --rm client
```

Ese único comando ejecuta C1, C2 y C4 secuencialmente, con cinco repeticiones completas y 20 solicitudes de calentamiento antes de cada repetición: son 15 corridas medidas en total. `docker compose run --rm` crea un contenedor temporal para la ejecución y lo elimina al terminar; los CSV, JSON y gráficos no se pierden porque están en un bind mount de Windows.

Si una corrida ya existe, el programa se detiene para no sobrescribirla. Para repetir deliberadamente, mueva los resultados anteriores o agregue `--overwrite` al comando del módulo.

### Pruebas automatizadas

```powershell
docker compose -f compose.client.yaml run --rm test-client
docker compose -f compose.server.yaml run --rm test-server
```

Las pruebas cubren autenticación, contrato HTTP, reintentos, PNG inválidos, correspondencia de `image_id`, conservación de píxeles, límite de concurrencia, solapamiento del pipeline y serialización de las inferencias.

### Resultados e interpretación

- Cliente: `results/experiment2/client/run_c1`, `run_c2`, `run_c4`, `summary` y `figures`.
- Servidor: `results/experiment2/server/requests.csv`, `resources.csv`, `resource_summary.json` y `system_info.json`.
- El tiempo HTTP incluye comunicación, ngrok, colas, inferencia y serialización; no se denomina “latencia de red”.
- Los bytes PNG son payload de aplicación, no bytes totales transmitidos por la red.
- El speedup frente al experimento 1 combina hardware, distribución y comunicación. Para estudiar la concurrencia, compare C2/C4 contra C1 bajo condiciones equivalentes.
- Los resultados del cliente y servidor viven en computadoras distintas y no se sincronizan automáticamente.

Para detener el servidor:

```powershell
docker compose -f compose.server.yaml down
```
