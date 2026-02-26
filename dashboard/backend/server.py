from __future__ import annotations

import math
import json
import os
import pickle
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
from uuid import uuid4

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.train_notebook_artifact import FEATURES, ARTIFACT_PATH, load_csv_data, resolve_data_path, train_artifact

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
CHAT_DB_PATH = BASE_DIR / "backend" / "taai_chat.db"

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
    intent: str | None = None
    confidence: float | None = None


def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(CHAT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_chat_db() -> None:
    CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_session_created ON chat_messages(session_id, created_at)")
        conn.commit()


def _save_chat_message(session_id: str, role: str, content: str) -> None:
    with _db_conn() as conn:
        conn.execute(
            "INSERT INTO chat_messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def _load_chat_messages(session_id: str, limit: int = 24) -> list[dict[str, str]]:
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, max(1, limit)),
        ).fetchall()
    out = [{"role": str(r["role"]), "content": str(r["content"]), "created_at": str(r["created_at"])} for r in reversed(rows)]
    return out


def _list_chat_sessions(limit: int = 20) -> list[dict[str, Any]]:
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT session_id, MAX(created_at) AS last_at,
                   SUBSTR(MAX(CASE WHEN role='user' THEN content ELSE '' END), 1, 90) AS preview
            FROM chat_messages
            GROUP BY session_id
            ORDER BY last_at DESC
            LIMIT ?
            """,
            (max(1, limit),),
        ).fetchall()
    return [{"session_id": str(r["session_id"]), "last_at": str(r["last_at"]), "preview": str(r["preview"] or "")} for r in rows]


KNOWLEDGE_SNIPPETS: list[dict[str, str]] = [
    {"topic": "elasticity", "text": "Price/income/feature elasticities describe percentage sensitivity in demand outcomes."},
    {"topic": "causal", "text": "Correlation is not causation; causal claims need counterfactual checks and confounder control."},
    {"topic": "forecast", "text": "Scenario forecasting should provide base, optimistic, and pessimistic cases with explicit assumptions."},
    {"topic": "anomaly", "text": "Anomaly detection highlights unusual observations; interpretation must separate event effects from noise."},
]


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
    src = resolve_data_path()

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
        "data_source_csv": str(src) if src else "demo_generated_data",
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
    if any("\u0900" <= ch <= "\u097f" for ch in text):
        return "hindi"
    if any(k in t for k in ["kya", "kaise", "kyu", "kyun", "batao", "samjhao", "hai", "nahi"]):
        return "hinglish"
    return "english"


def _classify_intent(text: str) -> str:
    q = text.lower()
    if any(k in q for k in ["why", "cause", "reason", "drop", "decline", "impact"]):
        return "causal_analysis"
    if any(k in q for k in ["forecast", "predict", "scenario", "next", "future"]):
        return "forecasting"
    if any(k in q for k in ["compare", "vs", "versus", "top", "best", "store"]):
        return "comparative_analysis"
    if any(k in q for k in ["coefficient", "beta", "corr", "correlation", "elasticity"]):
        return "model_interpretation"
    if any(k in q for k in ["anomaly", "unusual", "outlier", "spike"]):
        return "anomaly_analysis"
    return "general_metrics"


def _confidence_for_intent(intent: str) -> float:
    mapping = {
        "causal_analysis": 0.78,
        "forecasting": 0.84,
        "comparative_analysis": 0.88,
        "model_interpretation": 0.90,
        "anomaly_analysis": 0.80,
        "general_metrics": 0.86,
    }
    return float(mapping.get(intent, 0.75))


def _retrieve_knowledge(text: str, k: int = 2) -> list[str]:
    q = text.lower()
    scored: list[tuple[int, str]] = []
    for s in KNOWLEDGE_SNIPPETS:
        score = 0
        if s["topic"] in q:
            score += 2
        for token in s["text"].lower().split():
            if token in q:
                score += 1
        scored.append((score, s["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [txt for sc, txt in scored if sc > 0][:k]
    if not picked:
        picked = [KNOWLEDGE_SNIPPETS[0]["text"]]
    return picked


def _analysis_snapshot(store: str = "all", weeks: int = 160) -> dict[str, Any]:
    rows = _filtered_rows(store, weeks)
    sales = [float(r["Weekly_Sales"]) for r in rows]
    preds = [float(r["Predicted_Sales"]) for r in rows]
    if not sales:
        return {"avg_sales": 0.0, "avg_pred": 0.0, "residual_mean": 0.0, "residual_std": 0.0, "anomaly_count": 0}
    avg_sales = mean(sales)
    avg_pred = mean(preds)
    residuals = [abs(a - p) for a, p in zip(sales, preds)]
    res_mean = mean(residuals)
    res_std = _std(residuals)
    anomaly_count = sum(1 for r in residuals if r > res_mean + 2 * res_std) if res_std > 0 else 0
    return {
        "avg_sales": round(avg_sales, 2),
        "avg_pred": round(avg_pred, 2),
        "residual_mean": round(res_mean, 2),
        "residual_std": round(res_std, 2),
        "anomaly_count": int(anomaly_count),
    }


def _economic_answer(message: str) -> str:
    q = message.lower()
    intent = _classify_intent(message)
    rows = _filtered_rows("all", 160)
    avg_sales = mean([float(r["Weekly_Sales"]) for r in rows])
    peak_sales = max([float(r["Weekly_Sales"]) for r in rows])
    pred_avg = mean([float(r["Predicted_Sales"]) for r in rows])
    pred_gap_pct = ((pred_avg - avg_sales) / max(avg_sales, 1.0)) * 100.0

    if intent == "causal_analysis":
        ranked = sorted(
            [(f, abs(float(MODEL_COEFFICIENTS.get(f, 0.0)))) for f in FEATURES],
            key=lambda x: x[1],
            reverse=True,
        )
        drivers = ", ".join([f"{name} ({val:.3f})" for name, val in ranked[:3]])
        return (
            "Causal-style diagnostics using current deployed coefficients:\n"
            f"- Top sensitivity drivers: {drivers}\n"
            "- Holiday spikes and seasonal waves are also visible in trend decomposition.\n"
            "- Use What-If sliders to test counterfactual scenarios."
        )

    if intent == "model_interpretation":
        return "\n".join(
            [
                "Notebook-linked coefficients currently used:",
                *[f"- {f}: {float(MODEL_COEFFICIENTS.get(f, 0.0)):.4f}" for f in FEATURES],
            ]
        )
    if intent == "forecasting":
        optimistic = pred_avg * 1.06
        pessimistic = pred_avg * 0.94
        return (
            "Scenario forecast summary:\n"
            f"- Base case avg weekly sales: {pred_avg:,.0f}\n"
            f"- Optimistic (+6%): {optimistic:,.0f}\n"
            f"- Pessimistic (-6%): {pessimistic:,.0f}\n"
            f"- Baseline gap vs actual: {pred_gap_pct:+.2f}%"
        )
    if intent == "comparative_analysis":
        grouped: dict[int, list[dict[str, Any]]] = {}
        for r in MODEL_ROWS:
            grouped.setdefault(int(r["Store"]), []).append(r)
        top = sorted(
            [(sid, mean([float(x["Weekly_Sales"]) for x in vals])) for sid, vals in grouped.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        return "Top stores by avg weekly sales: " + ", ".join([f"Store {sid} ({val:,.0f})" for sid, val in top])
    if intent == "anomaly_analysis":
        snap = _analysis_snapshot("all", 160)
        return (
            "Anomaly scan summary:\n"
            f"- Estimated anomaly weeks: {snap['anomaly_count']}\n"
            f"- Mean absolute residual: {snap['residual_mean']:,.0f}\n"
            f"- Residual volatility: {snap['residual_std']:,.0f}\n"
            "- Review high-residual periods and holiday/context events for root-cause validation."
        )
    return (
        f"Current aggregate metrics: avg weekly sales {avg_sales:,.0f}, peak {peak_sales:,.0f}, baseline prediction avg {pred_avg:,.0f}. "
        "Ask for forecast scenarios, coefficient interpretation, causal diagnostics, anomaly scan, or store comparisons."
    )


def _grounding_context() -> str:
    rows = _filtered_rows("all", 160)
    sales = [float(r["Weekly_Sales"]) for r in rows]
    preds = [float(r["Predicted_Sales"]) for r in rows]
    avg_sales = mean(sales) if sales else 0.0
    avg_pred = mean(preds) if preds else 0.0
    coeff_lines = ", ".join([f"{f}={float(MODEL_COEFFICIENTS.get(f, 0.0)):.4f}" for f in FEATURES])
    return (
        f"Data window rows={len(rows)}, avg_sales={avg_sales:.2f}, avg_pred={avg_pred:.2f}. "
        f"Model source={MODEL_INFO.get('model_source','unknown')}. Coefficients: {coeff_lines}. "
        "All claims must be grounded in these values or explicit model caveats."
    )


def _call_llm_with_grounding(user_message: str, language: str, history: list[dict[str, str]]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt = (
        "You are taAI, a Walmart financial economist assistant. "
        "Respond with: 1) direct answer, 2) key drivers, 3) implication/recommendation. "
        "Use concise, professional language. Do not invent data."
    )
    if language == "hinglish":
        system_prompt += " Reply in Hinglish."
    elif language == "hindi":
        system_prompt += " Reply in Hindi."
    else:
        system_prompt += " Reply in English."

    intent = _classify_intent(user_message)
    context = _grounding_context()
    knowledge = "\n".join([f"- {x}" for x in _retrieve_knowledge(user_message, k=2)])
    recent = history[-8:]
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
    combined_user = (
        f"Intent: {intent}\n"
        f"Context:\n{context}\n\n"
        f"Relevant economics snippets:\n{knowledge}\n\n"
        f"Conversation:\n{transcript}\n\n"
        f"Current user question:\n{user_message}"
    )

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": combined_user}]},
        ],
        "temperature": 0.2,
        "max_output_tokens": 500,
    }

    req = urlrequest.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=25) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    text = body.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = body.get("output", [])
    parts: list[str] = []
    for item in output:
        for c in item.get("content", []):
            if c.get("type") == "output_text" and c.get("text"):
                parts.append(str(c["text"]))
    final = "\n".join(parts).strip()
    return final or None


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
    _init_chat_db()
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
    _save_chat_message(session_id, "user", msg)
    history = _load_chat_messages(session_id, limit=24)

    lang = _detect_language(msg)
    intent = _classify_intent(msg)
    confidence = _confidence_for_intent(intent)
    llm_answer = _call_llm_with_grounding(msg, lang, history)
    if llm_answer:
        answer = llm_answer
        source_tag = "openai_grounded"
    else:
        core = _economic_answer(msg)
        if lang == "hinglish":
            answer = f"{core}\nAgar required ho, main detailed economic decomposition bhi de sakta hoon."
        elif lang == "hindi":
            answer = f"{core}\nआवश्यक होने पर मैं इसे चरण-दर-चरण विस्तार से समझा सकता हूँ।"
        else:
            answer = core
        source_tag = "rule_based_fallback"

    _save_chat_message(session_id, "assistant", answer)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        detected_language=lang,
        sources=["walmart_sales_forecasting.ipynb", "extra_trees_notebook.pkl", source_tag],
        intent=intent,
        confidence=confidence,
    )


@app.get("/api/taai/sessions/{session_id}")
def taai_session(session_id: str) -> dict[str, Any]:
    return {"session_id": session_id, "messages": _load_chat_messages(session_id, limit=200)}


@app.get("/api/taai/sessions")
def taai_sessions(limit: int = 20) -> dict[str, Any]:
    return {"sessions": _list_chat_sessions(limit=limit)}


@app.get("/api/taai/suggestions")
def taai_suggestions() -> dict[str, Any]:
    return {
        "suggestions": [
            "Give me a 3-scenario forecast summary for next quarter.",
            "Why did sales drop recently? Show probable drivers.",
            "Compare top 3 stores by average weekly sales.",
            "Explain CPI and fuel price coefficients in simple terms.",
            "Run anomaly scan and summarize unusual weeks.",
        ]
    }


@app.get("/api/taai/insights")
def taai_insights(store: str = "all", weeks: int = 160) -> dict[str, Any]:
    snap = _analysis_snapshot(store, weeks)
    coeff = {f: round(float(MODEL_COEFFICIENTS.get(f, 0.0)), 4) for f in FEATURES}
    return {
        "store": store,
        "weeks": weeks,
        "snapshot": snap,
        "coefficients": coeff,
        "model_source": MODEL_INFO.get("model_source", "unknown"),
    }


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
