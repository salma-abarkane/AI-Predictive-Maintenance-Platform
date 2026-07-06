from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(r"C:\Users\yassi\OneDrive\Documents\projet pfa")
WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"

sys.path.append(str(BASE_DIR / "src"))

from ml_pipeline import predict_from_file
from ollama_client import DEFAULT_OLLAMA_MODEL, list_models, resolve_model, safe_ollama_generate


UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LATEST_CONTEXT: dict = {}

app = FastAPI(title="Predictive Maintenance Web App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/ollama/models")
def ollama_models() -> JSONResponse:
    try:
        models = list_models()
        return JSONResponse(
            {
                "available": True,
                "models": models,
                "default_model": resolve_model(DEFAULT_OLLAMA_MODEL),
            }
        )
    except Exception as exc:
        return JSONResponse(
            {
                "available": False,
                "models": [],
                "default_model": DEFAULT_OLLAMA_MODEL,
                "error": str(exc),
            }
        )


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    ollama_model: str = Form(DEFAULT_OLLAMA_MODEL),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".txt"}:
        raise HTTPException(status_code=400, detail="Upload a .csv or .txt file.")

    upload_path = UPLOADS_DIR / file.filename
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        bundle = predict_from_file(upload_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    result_path = RESULTS_DIR / f"{upload_path.stem}_predictions.csv"
    bundle.dataframe.to_csv(result_path, index=False)

    rows = bundle.dataframe[
        [
            "unit_number",
            "time_in_cycles",
            "anomaly_prediction",
            "anomaly_probability",
            "severity",
            "maintenance_window",
        ]
    ].head(200)

    preview_rows = rows.to_dict(orient="records")
    llm_summary = build_ollama_summary(bundle.summary, preview_rows, ollama_model)

    global LATEST_CONTEXT
    LATEST_CONTEXT = {
        "summary": bundle.summary,
        "preview": preview_rows[:25],
        "download_url": f"/api/download/{result_path.name}",
        "model": ollama_model,
        "filename": file.filename,
    }

    payload = {
        "summary": bundle.summary,
        "llm_summary": llm_summary,
        "download_url": f"/api/download/{result_path.name}",
        "preview": preview_rows,
    }
    return JSONResponse(payload)


@app.get("/api/download/{filename}")
def download(filename: str) -> FileResponse:
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, filename=filename)


def build_ollama_summary(summary: dict, preview_rows: list[dict], model_name: str) -> dict:
    compact_rows = [
        {
            "engine": row["unit_number"],
            "cycle": row["time_in_cycles"],
            "anomaly": row["anomaly_prediction"],
            "probability": row["anomaly_probability"],
            "severity": row["severity"],
        }
        for row in preview_rows[:5]
    ]
    prompt = (
        "You are an industrial predictive maintenance assistant. "
        "Write a short operational summary in plain English based on these prediction results. "
        "Mention the anomaly volume, criticality, likely urgency, and one maintenance recommendation. "
        "Keep it under 120 words.\n\n"
        f"Summary metrics: {summary}\n"
        f"Preview rows: {compact_rows}"
    )
    return safe_ollama_generate(prompt, model_name, timeout=90)


class ChatPayload(BaseModel):
    message: str
    model: str | None = None


@app.post("/api/chat")
def chat(payload: ChatPayload) -> JSONResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message is required.")

    if not LATEST_CONTEXT:
        raise HTTPException(
            status_code=400,
            detail="Upload a dataset first so the chatbot has prediction context.",
        )

    chat_prompt = (
        "You are a predictive maintenance assistant inside a public website. "
        "Answer clearly and briefly in English. Explain anomalies, severity, maintenance urgency, "
        "or what the uploaded dataset suggests. If the user asks when a failure may arrive, answer "
        "using probability, severity, and pattern-based caution rather than pretending to know an exact real failure date.\n\n"
        f"Uploaded file: {LATEST_CONTEXT.get('filename')}\n"
        f"Prediction summary: {LATEST_CONTEXT.get('summary')}\n"
        f"Preview rows: {LATEST_CONTEXT.get('preview')[:5]}\n\n"
        f"User question: {payload.message.strip()}"
    )

    result = safe_ollama_generate(chat_prompt, payload.model or LATEST_CONTEXT.get("model"), timeout=90)
    return JSONResponse(result)
