from __future__ import annotations

import math
import pickle
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.train_notebook_artifact import FEATURES, ARTIFACT_PATH, load_csv_data, train_artifact

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"

MODEL_LOCK = threading.Lock()
MODEL_ROWS: list[dict[str, Any]] = []
MODEL_COEFFICIENTS: dict[str, float] = {}
MODEL_INFO: dict[str, Any] = {}
CHAT_SESSIONS: dict[str, list[dict[str, str]]] = {}
SCHEDULER: BackgroundScheduler | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    detected_language: str
    sources: list[str]


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = 0.0
    dx = 0.0
    dy = 0.0
    for i in range(n):
        xv = xs[i] - mx
        yv = ys[i] - my
        num += xv * yv
        dx += xv * xv
        dy += yv * yv
    den = math.sqrt(dx * dy)
    return 0.0 if den == 0 else num / den


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(var, 0.0))


def _load_or_train_artifact() -> dict[str, Any]:
    if ARTIFACT_PATH.exists():
        try:
            with ARTIFACT_PATH.open("rb") as f:
                art = pickle.load(f)
            if (
                isinstance(art, dict)
                and art.get("source") == "extra_trees_notebook"
                and int(art.get("rows_fit", 0)) > 0
                and isinstance(art.get("pred_map"), dict)
                and len(art.get("pred_map", {})) > 0
            ):
                return art
        except Exception:
            pass
    return train_artifact()


def _build_runtime_state() -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    raw_rows = load_csv_data()
    artifact = _load_or_train_artifact()
    pred_map = artifact.get("pred_map", {})

    rows: list[dict[str, Any]] = []
    for r in raw_rows:
        nr = dict(r)
        key = (int(nr["Store"]), str(nr["Date"]))
        nr["Predicted_Sales"] = round(max(1000.0, float(pred_map.get(key, nr["Weekly_Sales"]))), 2)
        rows.append(nr)

    coeffs = artifact.get("feature_coefficients", {})
    info = {
        "model_source": "extra_trees_notebook_pickle",
        "trained_at": artifact.get("trained_at"),
        "rows_fit": artifact.get("rows_fit"),
        "r2_train": artifact.get("r2_train"),
        "pickle_path": str(ARTIFACT_PATH.relative_to(BASE_DIR)),
        "coef_source": "walmart_sales_forecasting.ipynb demand-equation terms",
        "prediction_source": "walmart_sales_forecasting.ipynb ExtraTrees pipeline",
        "retrain_cron": "every_6_hours",
    }
    return rows, coeffs, info


def _refresh_model_state() -> None:
    global MODEL_ROWS, MODEL_COEFFICIENTS, MODEL_INFO
    with MODEL_LOCK:
        rows, coeffs, info = _build_runtime_state()
        MODEL_ROWS = rows
        MODEL_COEFFICIENTS = coeffs
        MODEL_INFO = info


def _start_scheduler() -> None:
    global SCHEDULER
    if SCHEDULER is not None:
        return
    SCHEDULER = BackgroundScheduler(timezone="UTC")
    SCHEDULER.add_job(_refresh_model_state, "interval", hours=6, id="retrain_extra_trees", replace_existing=True)
    SCHEDULER.start()


def _filtered_rows(store: str, weeks: int) -> list[dict[str, Any]]:
    rows_snapshot = MODEL_ROWS
    if store == "all":
        by_date: dict[str, list[dict[str, Any]]] = {}
        for r in rows_snapshot:
            by_date.setdefault(str(r["Date"]), []).append(r)

        data = []
        for dt in sorted(by_date.keys()):
            rows = by_date[dt]
            sales = [float(x["Weekly_Sales"]) for x in rows]
            preds = [float(x["Predicted_Sales"]) for x in rows]
            data.append(
                {
                    "Store": 0,
                    "Date": dt,
                    "Weekly_Sales": round(sum(sales), 2),
                    "Predicted_Sales": round(sum(preds), 2),
                    "Holiday_Flag": int(max(int(x["Holiday_Flag"]) for x in rows)),
                    "Temperature": round(mean([float(x["Temperature"]) for x in rows]), 3),
                    "Fuel_Price": round(mean([float(x["Fuel_Price"]) for x in rows]), 3),
                    "CPI": round(mean([float(x["CPI"]) for x in rows]), 3),
                    "Unemployment": round(mean([float(x["Unemployment"]) for x in rows]), 3),
                }
            )
    else:
        try:
            sid = int(store)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid store") from exc
        data = [r for r in rows_snapshot if int(r["Store"]) == sid]

    if not data:
        raise HTTPException(status_code=404, detail="No data found")

    return sorted(data, key=lambda r: r["Date"])[-max(1, weeks) :]


def _detect_language(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["kya", "kaise", "kyu", "kyun", "batao", "samjhao", "hai", "nahi"]):
        return "hinglish"
    return "english"


def _economic_answer(message: str) -> str:
    q = message.lower()
    rows = _filtered_rows("all", 160)
    avg_sales = mean([float(r["Weekly_Sales"]) for r in rows])
    peak_sales = max([float(r["Weekly_Sales"]) for r in rows])

    if "coefficient" in q or "beta" in q:
        return "\n".join(
            [
                "Notebook-linked coefficients currently used:",
                *[f"- {f}: {float(MODEL_COEFFICIENTS.get(f, 0.0)):.4f}" for f in FEATURES],
            ]
        )
    if "forecast" in q or "predict" in q:
        pred_avg = mean([float(r["Predicted_Sales"]) for r in rows])
        return f"Baseline forecast average weekly sales: {pred_avg:,.0f}."
    return f"Average weekly sales are {avg_sales:,.0f}; peak weekly sales are {peak_sales:,.0f}."


app = FastAPI(title="Walmart Forecast API", version="6.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    _refresh_model_state()
    _start_scheduler()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "rows": len(MODEL_ROWS),
        "model_source": MODEL_INFO.get("model_source", "unknown"),
        "model_info": MODEL_INFO,
        "chatbot": "taAI",
    }


@app.get("/api/stores")
def stores() -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for r in MODEL_ROWS:
        grouped.setdefault(int(r["Store"]), []).append(r)

    out = []
    for sid in sorted(grouped.keys()):
        vals = grouped[sid]
        out.append(
            {
                "Store": sid,
                "records": len(vals),
                "avg_sales": round(mean([float(v["Weekly_Sales"]) for v in vals]), 2),
            }
        )
    return out


@app.get("/api/overview")
def overview(store: str = "all", weeks: int = 160) -> dict[str, Any]:
    rows = _filtered_rows(store, weeks)
    sales = [float(r["Weekly_Sales"]) for r in rows]
    holidays = [float(r["Weekly_Sales"]) for r in rows if int(r["Holiday_Flag"]) == 1]
    return {
        "store": store,
        "records": len(rows),
        "date_min": rows[0]["Date"],
        "date_max": rows[-1]["Date"],
        "avg_weekly_sales": round(mean(sales), 2),
        "peak_sales": round(max(sales), 2),
        "holiday_avg": round(mean(holidays), 2) if holidays else 0.0,
        "holiday_count": len(holidays),
    }


@app.get("/api/store-data")
def store_data(store: str = "all", weeks: int = 160) -> list[dict[str, Any]]:
    return _filtered_rows(store, weeks)


@app.get("/api/correlations")
def correlations(store: str = "all", weeks: int = 160) -> list[dict[str, Any]]:
    rows = _filtered_rows(store, weeks)
    y = [float(r["Weekly_Sales"]) for r in rows]
    out = []
    for f in FEATURES:
        x = [float(r[f]) for r in rows]
        out.append({"feature": f, "corr": round(_pearson(x, y), 4)})
    return out


@app.get("/api/coefficients")
def coefficients(store: str = "all", weeks: int = 160) -> dict[str, Any]:
    rows = _filtered_rows(store, weeks)
    y = [float(r["Weekly_Sales"]) for r in rows]
    out = []
    for f in FEATURES:
        x = [float(r[f]) for r in rows]
        corr = _pearson(x, y)
        std_x = _std(x)
        std_y = _std(y)
        mean_x = sum(x) / len(x) if x else 0.0
        beta_per_unit = float(MODEL_COEFFICIENTS.get(f, 0.0))
        beta_10pct = beta_per_unit * (0.1 * mean_x)
        out.append(
            {
                "feature": f,
                "corr": round(corr, 4),
                "std_x": round(std_x, 4),
                "std_y": round(std_y, 4),
                "mean_x": round(mean_x, 4),
                "beta_per_unit": round(beta_per_unit, 4),
                "beta_10pct": round(beta_10pct, 2),
            }
        )

    return {
        "store": store,
        "weeks": weeks,
        "rows": out,
        "target": "Weekly_Sales",
        "model_source": MODEL_INFO.get("model_source", "unknown"),
        "model_info": MODEL_INFO,
        "note": "Predictions come from notebook-style ExtraTrees pickle. Coefficients come from notebook demand-equation term mapping stored in pickle.",
    }


@app.post("/api/taai/chat", response_model=ChatResponse)
def taai_chat(payload: ChatRequest) -> ChatResponse:
    msg = (payload.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    session_id = payload.session_id or str(uuid4())
    history = CHAT_SESSIONS.setdefault(session_id, [])
    history.append({"role": "user", "content": msg})

    lang = _detect_language(msg)
    core = _economic_answer(msg)
    answer = f"taAI insight: {core}" if lang == "english" else f"taAI insight: {core}\nAgar chaho to aur detail de sakta hoon."

    history.append({"role": "assistant", "content": answer})
    if len(history) > 16:
        CHAT_SESSIONS[session_id] = history[-16:]

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        detected_language=lang,
        sources=["walmart_sales_forecasting.ipynb", "extra_trees_notebook.pkl"],
    )


@app.get("/api/taai/sessions/{session_id}")
def taai_session(session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "messages": CHAT_SESSIONS.get(session_id, [])}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/{full_path:path}")
def fallback(full_path: str) -> FileResponse:
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    target = STATIC_DIR / full_path
    if target.exists() and target.is_file():
        return FileResponse(target)
    return FileResponse(STATIC_DIR / "index.html")
