from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.anomaly_detector import AnomalyDetector
from app.llm_service import TicketQueryInterpreter, UnsupportedIntentError
from app.schemas import AnomaliesResponse, HealthResponse, QueryRequest, QueryResponse


app = FastAPI(title="Support Ticket Analytics API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

query_interpreter = TicketQueryInterpreter()
anomaly_detector = AnomalyDetector()


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [_json_safe(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(key): _json_safe(item) for key, item in value.to_dict().items()}
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported JSON result type: {type(value).__name__}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse | JSONResponse:
    try:
        structured_query = query_interpreter.llm_service.interpret_question(request.question)
        result = query_interpreter.execute(structured_query)
        answer = query_interpreter.format_answer(structured_query, result)
        safe_result = _json_safe(result)
    except UnsupportedIntentError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})
    except RuntimeError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=502, content={"detail": str(exc)})
    except Exception:
        return JSONResponse(status_code=500, content={"detail": "Internal query error."})

    return QueryResponse(
        question=request.question,
        answer=answer,
        result=safe_result,
    )


@app.get("/anomalies", response_model=AnomaliesResponse)
def anomalies(reference_time: datetime | None = Query(default=None)) -> AnomaliesResponse | JSONResponse:
    try:
        records = anomaly_detector.find_all_anomalies(reference_time=reference_time)
        safe_records = _json_safe(records)
    except Exception:
        return JSONResponse(status_code=500, content={"detail": "Internal anomaly detection error."})

    return AnomaliesResponse(total=len(safe_records), anomalies=safe_records)