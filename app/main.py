import json
import os
import sqlite3
import time
import uuid
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from faster_whisper import WhisperModel  # type: ignore
from transformers import pipeline as hf_pipeline  # type: ignore
import soundfile as sf


# ========================
# Paths / Config
# ========================

DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
DB_PATH = DATA_DIR / "jobs.db"

os.environ.setdefault("HF_HUB_OFFLINE", "1")


# ========================
# Models
# ========================

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


class SegmentOutput(BaseModel):
    text: str
    speaker: str
    start: float
    end: float
    sentiment: str
    sentiment_score: float


class JobResult(BaseModel):
    full_transcript: str
    segments: List[SegmentOutput]


class JobResponse(BaseModel):
    id: str
    status: str
    error: Optional[str] = None
    result: Optional[JobResult] = None


# ========================
# Utils
# ========================

def ensure_paths() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOADS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def get_db() -> sqlite3.Connection:
    ensure_paths()
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


# ========================
# Audio preprocessing
# ========================

def normalize_audio(input_path: Path) -> Path:
    """
    Normaliza audio VoIP:
    - mono
    - 16kHz
    - PCM
    - corrige drift RTP
    """
    out = input_path.with_suffix(".16k.wav")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-af", "aresample=async=1:first_pts=0",
            str(out),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


def split_stereo(input_path: Path) -> tuple[Path, Path]:
    """
    Split estéreo robusto para Asterisk usando filtros pan
    (NO map_channel)
    """
    left = input_path.with_suffix(".ch0.wav")
    right = input_path.with_suffix(".ch1.wav")

    # Canal izquierdo
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-af", "pan=mono|c0=FL",
            left,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Canal derecho
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-af", "pan=mono|c0=FR",
            right,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return left, right


# ========================
# Audio Processor
# ========================

class AudioProcessor:
    def __init__(self) -> None:
        self.device = os.getenv("DEVICE", "cuda")
        self.compute_type = os.getenv("COMPUTE_TYPE", "float16")
        self.model_path = os.getenv("TRANSCRIPTION_MODEL_PATH", "models/whisper-medium")
        self.sentiment_model_path = os.getenv("SENTIMENT_MODEL_PATH", "models/sentiment")

        self.whisper: Optional[WhisperModel] = None
        self.sentiment = None

    def load_whisper(self) -> WhisperModel:
        if self.whisper is None:
            self.whisper = WhisperModel(
                self.model_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self.whisper

    def load_sentiment(self):
        if self.sentiment is None:
            self.sentiment = hf_pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_path,
                device=0 if self.device.startswith("cuda") else -1,
            )
        return self.sentiment

    def transcribe(self, audio_path: Path) -> List[TranscriptionSegment]:
        model = self.load_whisper()
        segments, _ = model.transcribe(
            str(audio_path),
            vad_filter=True,
            word_timestamps=False,
        )
        return [
            TranscriptionSegment(
                start=float(s.start),
                end=float(s.end),
                text=s.text.strip(),
            )
            for s in segments
        ]

    def analyze_sentiment(self, text: str) -> tuple[str, float]:
        pipe = self.load_sentiment()
        res = pipe(text, truncation=True)[0]
        return res["label"].lower(), float(res["score"])

    def process(self, audio_path: Path) -> tuple[str, List[SegmentOutput]]:
        with sf.SoundFile(audio_path) as f:
            channels = f.channels

        results: List[SegmentOutput] = []

        if channels >= 2:
            ch0, ch1 = split_stereo(audio_path)
            ch0 = normalize_audio(ch0)
            ch1 = normalize_audio(ch1)

            for speaker, ch in [("agent", ch0), ("customer", ch1)]:
                for seg in self.transcribe(ch):
                    sentiment, score = self.analyze_sentiment(seg.text)
                    results.append(
                        SegmentOutput(
                            text=seg.text,
                            speaker=speaker,
                            start=seg.start,
                            end=seg.end,
                            sentiment=sentiment,
                            sentiment_score=score,
                        )
                    )
        else:
            mono = normalize_audio(audio_path)
            for seg in self.transcribe(mono):
                sentiment, score = self.analyze_sentiment(seg.text)
                results.append(
                    SegmentOutput(
                        text=seg.text,
                        speaker="unknown",
                        start=seg.start,
                        end=seg.end,
                        sentiment=sentiment,
                        sentiment_score=score,
                    )
                )

        results.sort(key=lambda s: s.start)
        full_transcript = " ".join(r.text for r in results)
        return full_transcript, results


# ========================
# Worker
# ========================

class JobQueueWorker:
    def __init__(self, processor: AudioProcessor) -> None:
        self.processor = processor
        self.stop_event = Event()
        self.thread: Optional[Thread] = None

    def start(self) -> None:
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self) -> None:
        conn = get_db()
        while not self.stop_event.is_set():
            job = conn.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at LIMIT 1",
                (JobStatus.QUEUED,),
            ).fetchone()

            if not job:
                time.sleep(1)
                continue

            try:
                conn.execute(
                    "UPDATE jobs SET status=? WHERE id=?",
                    (JobStatus.PROCESSING, job["id"]),
                )
                conn.commit()

                full, segments = self.processor.process(Path(job["filename"]))
                out = RESULTS_DIR / f"{job['id']}.json"

                with out.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "job_id": job["id"],
                            "full_transcript": full,
                            "segments": [s.model_dump() for s in segments],
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                conn.execute(
                    "UPDATE jobs SET status=?, result_path=? WHERE id=?",
                    (JobStatus.COMPLETED, str(out), job["id"]),
                )
                conn.commit()

            except Exception as e:
                conn.execute(
                    "UPDATE jobs SET status=?, error_message=? WHERE id=?",
                    (JobStatus.FAILED, str(e), job["id"]),
                )
                conn.commit()


# ========================
# FastAPI
# ========================

app = FastAPI(title="Transcripción VoIP Estéreo")
processor = AudioProcessor()
worker = JobQueueWorker(processor)


@app.on_event("startup")
async def startup():
    ensure_paths()
    worker.start()


@app.post("/jobs", response_model=JobResponse)
async def create_job(file: UploadFile = File(...), conn=Depends(get_db)):
    job_id = str(uuid.uuid4())
    path = UPLOADS_DIR / f"{job_id}{Path(file.filename).suffix}"

    with path.open("wb") as f:
        f.write(await file.read())

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO jobs VALUES (?,?,?,?,?,?,?)",
        (job_id, str(path), JobStatus.QUEUED, now, now, None, None),
    )
    conn.commit()

    return JobResponse(id=job_id, status=JobStatus.QUEUED)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, conn=Depends(get_db)):
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(404)

    result = None
    if row["status"] == JobStatus.COMPLETED and row["result_path"]:
        with open(row["result_path"], "r", encoding="utf-8") as f:
            data = json.load(f)
            result = JobResult(
                full_transcript=data["full_transcript"],
                segments=[SegmentOutput(**s) for s in data["segments"]],
            )

    return JobResponse(
        id=row["id"],
        status=row["status"],
        error=row["error_message"],
        result=result,
    )
