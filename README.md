# Transcripción local con diarización y sentimiento

Servicio local (sin llamadas a la nube) que recibe audios, los encola para su procesamiento y genera un JSON con:

- Transcripción
- Identificador de interlocutor (diarización)
- Marcas de tiempo (inicio/fin)
- Sentimiento por segmento

## Requisitos previos

- Python 3.10+
- `ffmpeg` instalado en el sistema.
- Modelos descargados **localmente** (sin Internet) en `./models/`:
  - Whisper (p. ej. `models/whisper-small` o cualquier variante soportada por `faster-whisper`).
  - Diarización de `pyannote.audio` (p. ej. `models/diarization` con un `config.yaml` válido).
  - Sentimiento de `transformers` (p. ej. `models/sentiment` con `config.json`, `pytorch_model.bin`, `tokenizer.json`).

Ejemplo de estructura:

```
models/
  whisper-small/
    model.bin ...
  diarization/
    config.yaml ...
  sentiment/
    config.json
    tokenizer.json
    pytorch_model.bin
```

> Nota: descarga los modelos con conexión disponible y cópialos al directorio `models/` antes de ejecutar el servicio.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Ejecución

```bash
export HF_HUB_OFFLINE=1  # evita descargas
export TRANSCRIPTION_MODEL_PATH=models/whisper-small
export DIARIZATION_MODEL_PATH=models/diarization
export SENTIMENT_MODEL_PATH=models/sentiment
export NUM_SPEAKERS=2  # asume llamadas de dos interlocutores
uvicorn app.main:app --reload --port 8000
```

## Endpoints principales

- `POST /jobs` — Sube un archivo de audio (`multipart/form-data`, campo `file`) y lo encola.
- `GET /jobs/{job_id}` — Consulta estado y, cuando termina, devuelve el JSON generado.

Los resultados se guardan en `data/results/{job_id}.json`. Los audios se almacenan en `data/uploads/`.

## Notas sobre procesamiento

- Todo el pipeline se ejecuta en local.
- Se usa una cola basada en SQLite y un worker en segundo plano que procesa los trabajos secuencialmente.
- El JSON de salida incluye: `full_transcript` con la transcripción completa y, por segmento, texto, speaker, inicio, fin, etiqueta y puntuación de sentimiento. El diarizador se limita a los `NUM_SPEAKERS` más largos (por defecto 2).

## Desarrollo rápido

- Variables de entorno opcionales:
  - `QUEUE_POLL_INTERVAL` (segundos, por defecto 2)
  - `DEVICE` (`cpu` o `cuda`, por defecto `cpu`)
  - `TRANSCRIPTION_MODEL_SIZE` si prefieres cargar por nombre en vez de ruta (debe existir en `~/.cache` o `models/`).
- Para pruebas manuales, puedes usar `scripts/sample_request.http` (insértale tu `job_id` y ruta de audio).
