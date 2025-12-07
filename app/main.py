import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# Model imports are intentionally lazy-loaded inside AudioProcessor to avoid heavy startup.
from faster_whisper import WhisperModel  # type: ignore
from pyannote.audio import Pipeline  # type: ignore
from transformers import pipeline as hf_pipeline  # type: ignore


DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
DB_PATH = DATA_DIR / "jobs.db"

# Evita llamadas a la red de HuggingFace en entornos sin Internet.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


class SegmentOutput(BaseModel):
    text: str
    speaker: str
    start: float
    end: float
    sentiment: str
    sentiment_score: float


class JobResponse(BaseModel):
    id: str
    status: str
    error: Optional[str] = None
    result: Optional[List[SegmentOutput]] = None


def ensure_paths() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_db() -> sqlite3.Connection:
    ensure_paths()
    # check_same_thread=False because the worker and request handlers run in different threads.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            error_message TEXT,
            result_path TEXT
        )
        """
    )
    return conn


class AudioProcessor:
    def __init__(self) -> None:
        self.device = os.getenv("DEVICE", "cpu")
        self.compute_type = os.getenv("COMPUTE_TYPE", "int8")
        self.transcription_model_path = os.getenv(
            "TRANSCRIPTION_MODEL_PATH", os.getenv("TRANSCRIPTION_MODEL_SIZE", "models/whisper-small")
        )
        self.diarization_model_path = os.getenv("DIARIZATION_MODEL_PATH", "models/diarization")
        self.sentiment_model_path = os.getenv("SENTIMENT_MODEL_PATH", "models/sentiment")
        self.whisper_model: Optional[WhisperModel] = None
        self.diarization_pipeline: Optional[Pipeline] = None
        self.sentiment_pipeline = None
        self._validate_local_paths()

    def _require_local(self, path_str: str, env_keys: list[str]) -> Path:
        path = Path(path_str)
        if not path.exists():
            hint = " o ".join(env_keys)
            raise RuntimeError(
                f"Modelo no encontrado en ruta local '{path}'. "
                f"Configura {hint} apuntando a un modelo descargado en disco."
            )
        return path

    def _validate_local_paths(self) -> None:
        self._require_local(self.transcription_model_path, ["TRANSCRIPTION_MODEL_PATH", "TRANSCRIPTION_MODEL_SIZE"])
        self._require_local(self.diarization_model_path, ["DIARIZATION_MODEL_PATH"])
        self._require_local(self.sentiment_model_path, ["SENTIMENT_MODEL_PATH"])

    def load_transcriber(self) -> WhisperModel:
        if self.whisper_model is None:
            self.whisper_model = WhisperModel(
                self.transcription_model_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self.whisper_model

    def load_diarizer(self) -> Pipeline:
        if self.diarization_pipeline is None:
            self.diarization_pipeline = Pipeline.from_pretrained(self.diarization_model_path)
        return self.diarization_pipeline

    def load_sentiment(self):
        if self.sentiment_pipeline is None:
            self.sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_path,
                device=0 if self.device.startswith("cuda") else -1,
            )
        return self.sentiment_pipeline

    def transcribe(self, audio_path: Path) -> List[TranscriptionSegment]:
        model = self.load_transcriber()
        segments_iter, _ = model.transcribe(
            str(audio_path),
            vad_filter=True,
            word_timestamps=False,
        )
        segments: List[TranscriptionSegment] = []
        for segment in segments_iter:
            segments.append(
                TranscriptionSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                )
            )
        return segments

    def diarize(self, audio_path: Path) -> List[DiarizationSegment]:
        pipeline = self.load_diarizer()
        diarization = pipeline(str(audio_path))
        diar_segments: List[DiarizationSegment] = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            diar_segments.append(
                DiarizationSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker=str(speaker),
                )
            )
        return diar_segments

    def analyze_sentiment(self, text: str) -> tuple[str, float]:
        sentiment_pipe = self.load_sentiment()
        res = sentiment_pipe(text, truncation=True)[0]
        label = res["label"].lower()
        score = float(res["score"])
        return label, score

    @staticmethod
    def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    def _assign_speaker(self, segment: TranscriptionSegment, diar_segments: List[DiarizationSegment]) -> str:
        best_speaker = "unknown"
        best_overlap = 0.0
        for diar in diar_segments:
            overlap = self._overlap(segment.start, segment.end, diar.start, diar.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar.speaker
        return best_speaker

    def process(self, audio_path: Path) -> List[SegmentOutput]:
        transcription_segments = self.transcribe(audio_path)
        diarization_segments = self.diarize(audio_path)
        result_segments: List[SegmentOutput] = []
        for t_segment in transcription_segments:
            speaker = self._assign_speaker(t_segment, diarization_segments)
            sentiment_label, sentiment_score = self.analyze_sentiment(t_segment.text)
            result_segments.append(
                SegmentOutput(
                    text=t_segment.text,
                    speaker=speaker,
                    start=t_segment.start,
                    end=t_segment.end,
                    sentiment=sentiment_label,
                    sentiment_score=sentiment_score,
                )
            )
        return result_segments


class JobQueueWorker:
    def __init__(self, processor: AudioProcessor, poll_interval: float = 2.0) -> None:
        self.processor = processor
        self.poll_interval = poll_interval
        self.stop_event = Event()
        self.thread: Optional[Thread] = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)

    def fetch_next_job(self, conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
        cur = conn.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1",
            (JobStatus.QUEUED,),
        )
        row = cur.fetchone()
        return row

    def update_status(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        status: str,
        error_message: Optional[str] = None,
        result_path: Optional[str] = None,
    ) -> None:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, updated_at = ?, error_message = ?, result_path = ?
            WHERE id = ?
            """,
            (status, time.strftime("%Y-%m-%d %H:%M:%S"), error_message, result_path, job_id),
        )
        conn.commit()

    def process_job(self, conn: sqlite3.Connection, job_row: sqlite3.Row) -> None:
        job_id = job_row["id"]
        audio_path = Path(job_row["filename"])
        self.update_status(conn, job_id, JobStatus.PROCESSING)
        try:
            segments = self.processor.process(audio_path)
            result_path = RESULTS_DIR / f"{job_id}.json"
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "job_id": job_id,
                        "source": str(audio_path),
                        "segments": [s.model_dump() for s in segments],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            self.update_status(conn, job_id, JobStatus.COMPLETED, result_path=str(result_path))
        except Exception as exc:  # pylint: disable=broad-except
            self.update_status(conn, job_id, JobStatus.FAILED, error_message=str(exc))

    def run(self) -> None:
        conn = get_db()
        while not self.stop_event.is_set():
            job = self.fetch_next_job(conn)
            if job is not None:
                self.process_job(conn, job)
                continue
            time.sleep(self.poll_interval)


app = FastAPI(title="Transcripción local", version="0.1.0")
processor = AudioProcessor()
worker = JobQueueWorker(processor=processor, poll_interval=float(os.getenv("QUEUE_POLL_INTERVAL", 2)))


def get_connection():
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()


@app.on_event("startup")
async def startup_event() -> None:
    ensure_paths()
    worker.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    worker.stop()


@app.post("/jobs", response_model=JobResponse)
async def create_job(file: UploadFile = File(...), conn=Depends(get_connection)) -> JobResponse:
    ensure_paths()
    suffix = Path(file.filename).suffix
    job_id = str(uuid.uuid4())
    dest_path = UPLOADS_DIR / f"{job_id}{suffix}"
    with dest_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        """
        INSERT INTO jobs(id, filename, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (job_id, str(dest_path), JobStatus.QUEUED, now, now),
    )
    conn.commit()
    return JobResponse(id=job_id, status=JobStatus.QUEUED)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, conn=Depends(get_connection)) -> JobResponse:
    cur = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")

    status = row["status"]
    error = row["error_message"]
    result: Optional[List[SegmentOutput]] = None
    if status == JobStatus.COMPLETED and row["result_path"]:
        result_file = Path(row["result_path"])
        if result_file.exists():
            with result_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
                result = [SegmentOutput(**segment) for segment in payload.get("segments", [])]
    return JobResponse(id=row["id"], status=status, error=error, result=result)
