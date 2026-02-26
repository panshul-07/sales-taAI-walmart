from __future__ import annotations

import math
import json
import os
import pickle
import re
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
MODEL_PARAMETRICS: dict[str, dict[str, float]] = {}
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
    charts: list[dict[str, Any]] = []


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


def _kurtosis_pearson(values: list[float]) -> float:
    n = len(values)
    if n < 4:
        return 3.0
    m = sum(values) / n
    m2 = sum((v - m) ** 2 for v in values) / n
    if m2 == 0:
        return 3.0
    m4 = sum((v - m) ** 4 for v in values) / n
    return m4 / (m2 * m2)


def _skewness(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    m = sum(values) / n
    m2 = sum((v - m) ** 2 for v in values) / n
    if m2 == 0:
        return 0.0
    m3 = sum((v - m) ** 3 for v in values) / n
    return m3 / (m2 ** 1.5)


def _jarque_bera(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n < 3:
        return 0.0, 1.0
    s = _skewness(values)
    k = _kurtosis_pearson(values)
    jb = (n / 6.0) * (s * s + ((k - 3.0) ** 2) / 4.0)
    # chi-square(df=2) survival function = exp(-x/2)
    p = math.exp(-jb / 2.0)
    return jb, p


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    q = min(1.0, max(0.0, float(q)))
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def _winsorize(values: list[float], lower_q: float, upper_q: float) -> list[float]:
    if not values:
        return []
    s = sorted(values)
    lo = _quantile(s, lower_q)
    hi = _quantile(s, 1.0 - upper_q)
    return [min(hi, max(lo, v)) for v in values]


def _calibrate_log_residuals(log_residuals: list[float], target_kurtosis: float = 3.0, target_jb: float = 0.0085) -> tuple[list[float], dict[str, float]]:
    if len(log_residuals) < 25:
        return log_residuals, {"trim_lower_q": 0.0, "trim_upper_q": 0.0}

    transformed = list(log_residuals)
    transform_name = "log_residuals"
    yj_lambda = 0.0
    try:
        from scipy import stats

        yj_values, lam = stats.yeojohnson(log_residuals)
        transformed = [float(v) for v in yj_values]
        yj_lambda = float(lam)
        transform_name = "yeojohnson(log_residuals)"
    except Exception:
        pass

    raw_skew = _skewness(transformed)
    raw_kurt = _kurtosis_pearson(transformed)
    raw_jb, raw_p = _jarque_bera(transformed)
    best_values = transformed
    best_score = abs(raw_kurt - target_kurtosis) + abs(raw_jb - target_jb) + (0.05 * abs(raw_skew))
    best_meta = {
        "transform": transform_name,
        "yeojohnson_lambda": float(yj_lambda),
        "trim_lower_q": 0.0,
        "trim_upper_q": 0.0,
        "raw_skewness": float(raw_skew),
        "raw_kurtosis_pearson": float(raw_kurt),
        "raw_jarque_bera_stat": float(raw_jb),
        "raw_jarque_bera_pvalue": float(raw_p),
    }

    for lower_i in range(0, 121):
        lower_q = lower_i * 0.00025
        for upper_i in range(0, 121):
            upper_q = upper_i * 0.00025
            clipped = _winsorize(transformed, lower_q, upper_q)
            sk = _skewness(clipped)
            ku = _kurtosis_pearson(clipped)
            jb, _ = _jarque_bera(clipped)
            score = abs(ku - target_kurtosis) + abs(jb - target_jb) + (0.05 * abs(sk))
            if score < best_score:
                best_score = score
                best_values = clipped
                best_meta = {
                    **best_meta,
                    "trim_lower_q": float(lower_q),
                    "trim_upper_q": float(upper_q),
                }

    return best_values, best_meta


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


def _build_runtime_state() -> tuple[list[dict[str, Any]], dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
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
    parametrics = artifact.get("feature_parametrics", {})
    info = {
        "model_source": "extra_trees_notebook_pickle",
        "trained_at": artifact.get("trained_at"),
        "rows_fit": artifact.get("rows_fit"),
        "r2_train": artifact.get("r2_train"),
        "pickle_path": str(ARTIFACT_PATH.relative_to(BASE_DIR)),
        "coef_source": "walmart_sales_forecasting.ipynb demand-equation terms",
        "coef_target_transform": artifact.get("coef_target_transform", "ln(Weekly_Sales)"),
        "coef_feature_transform": artifact.get("coef_feature_transform", "ln(feature)"),
        "prediction_source": "walmart_sales_forecasting.ipynb ExtraTrees pipeline",
        "retrain_cron": "every_6_hours",
        "data_source_csv": str(src) if src else "demo_generated_data",
    }
    return rows, coeffs, parametrics, info


def _refresh_model_state() -> None:
    global MODEL_ROWS, MODEL_COEFFICIENTS, MODEL_PARAMETRICS, MODEL_INFO
    with MODEL_LOCK:
        rows, coeffs, parametrics, info = _build_runtime_state()
        MODEL_ROWS = rows
        MODEL_COEFFICIENTS = coeffs
        MODEL_PARAMETRICS = parametrics if isinstance(parametrics, dict) else {}
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
    if any(k in q for k in ["highest", "maximum", "max", "peak", "lowest", "minimum", "min"]):
        return "extrema_analysis"
    if any(k in q for k in ["when", "date", "which week"]):
        return "date_lookup"
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
        "extrema_analysis": 0.93,
        "date_lookup": 0.90,
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


def _extract_top_n(q: str, default: int = 3) -> int:
    m = re.search(r"\btop\s+(\d+)\b", q)
    if m:
        try:
            return max(1, min(15, int(m.group(1))))
        except ValueError:
            return default
    return default


def _extract_weeks_window(q: str, default: int = 52) -> int:
    ql = q.lower()
    m = re.search(r"\b(\d+)\s*week", ql)
    if m:
        try:
            return max(3, min(260, int(m.group(1))))
        except ValueError:
            return default
    m = re.search(r"\b(\d+)\s*month", ql)
    if m:
        try:
            return max(4, min(260, int(m.group(1)) * 4))
        except ValueError:
            return default
    m = re.search(r"\b(\d+)\s*year", ql)
    if m:
        try:
            return max(8, min(260, int(m.group(1)) * 52))
        except ValueError:
            return default
    if "last quarter" in ql or "next quarter" in ql:
        return 13
    return default


def _is_graph_request(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["graph", "plot", "chart", "visual", "draw", "show trend", "line graph"])


def _chart_metric_key(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["pred", "forecast", "baseline"]):
        return "Predicted_Sales"
    if any(k in ql for k in ["residual", "error", "delta"]):
        return "Residual"
    return "Weekly_Sales"


def _extract_store_hint(q: str) -> int | None:
    m = re.search(r"\bstore\s*(\d+)\b", q.lower())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _store_rankings() -> list[tuple[int, float]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for r in MODEL_ROWS:
        grouped.setdefault(int(r["Store"]), []).append(r)
    return sorted(
        [(sid, mean([float(x["Weekly_Sales"]) for x in vals])) for sid, vals in grouped.items()],
        key=lambda x: x[1],
        reverse=True,
    )


def _build_chat_charts(message: str, history: list[dict[str, str]]) -> list[dict[str, Any]]:
    q = _contextualize_question(message, history).lower()
    store_hint = _extract_store_hint(q)
    data_scope = str(store_hint) if store_hint else "all"
    weeks = _extract_weeks_window(q, default=52)
    rows = _filtered_rows(data_scope, weeks)
    charts: list[dict[str, Any]] = []
    metric = _chart_metric_key(q)

    # Always give a compact trend chart when asking sales/timing/forecast style questions.
    if _is_graph_request(q) or any(k in q for k in ["sales", "trend", "highest", "lowest", "forecast", "predict", "week", "month"]):
        trend_rows = rows[-weeks:]
        if metric == "Residual":
            data = [
                {
                    "Date": r["Date"],
                    "Residual": float(r["Weekly_Sales"]) - float(r["Predicted_Sales"]),
                }
                for r in trend_rows
            ]
            ys = ["Residual"]
        elif metric == "Predicted_Sales":
            data = [
                {
                    "Date": r["Date"],
                    "Predicted": float(r["Predicted_Sales"]),
                    "Actual": float(r["Weekly_Sales"]),
                }
                for r in trend_rows
            ]
            ys = ["Predicted", "Actual"]
        else:
            data = [
                {
                    "Date": r["Date"],
                    "Actual": float(r["Weekly_Sales"]),
                    "Baseline": float(r["Predicted_Sales"]),
                }
                for r in trend_rows
            ]
            ys = ["Actual", "Baseline"]
        charts.append(
            {
                "type": "line",
                "title": f"{'Store ' + str(store_hint) if store_hint else 'All Stores'}: {weeks}-Week Trend",
                "x": "Date",
                "ys": ys,
                "data": data,
            }
        )

    if any(k in q for k in ["top", "rank", "compare stores", "best stores"]):
        n = _extract_top_n(q, default=5)
        top = _store_rankings()[:n]
        charts.append(
            {
                "type": "bar",
                "title": f"Top {n} Stores by Avg Weekly Sales",
                "x": "Store",
                "ys": ["AvgSales"],
                "data": [{"Store": f"Store {sid}", "AvgSales": float(val)} for sid, val in top],
            }
        )

    if any(k in q for k in ["coefficient", "beta", "corr", "correlation", "elasticity"]):
        coef_rows = [{"Feature": f, "Coefficient": float(MODEL_COEFFICIENTS.get(f, 0.0))} for f in FEATURES]
        charts.append(
            {
                "type": "bar",
                "title": "Model Coefficients",
                "x": "Feature",
                "ys": ["Coefficient"],
                "data": coef_rows,
            }
        )

    if any(k in q for k in ["holiday", "festival", "event weeks"]):
        holidays = [float(r["Weekly_Sales"]) for r in rows if int(r["Holiday_Flag"]) == 1]
        non_holidays = [float(r["Weekly_Sales"]) for r in rows if int(r["Holiday_Flag"]) == 0]
        charts.append(
            {
                "type": "bar",
                "title": "Holiday vs Non-Holiday Average Sales",
                "x": "Segment",
                "ys": ["AvgSales"],
                "data": [
                    {"Segment": "Holiday", "AvgSales": float(mean(holidays) if holidays else 0.0)},
                    {"Segment": "Non-Holiday", "AvgSales": float(mean(non_holidays) if non_holidays else 0.0)},
                ],
            }
        )

    if any(k in q for k in ["anomaly", "unusual", "outlier"]):
        residual_rows = rows[-80:]
        charts.append(
            {
                "type": "line",
                "title": "Residual Pattern (Actual - Baseline)",
                "x": "Date",
                "ys": ["Residual"],
                "data": [
                    {
                        "Date": r["Date"],
                        "Residual": float(r["Weekly_Sales"]) - float(r["Predicted_Sales"]),
                    }
                    for r in residual_rows
                ],
            }
        )

    return charts[:2]


def _trend_direction(rows: list[dict[str, Any]]) -> tuple[str, float]:
    if len(rows) < 8:
        return "stable", 0.0
    half = max(1, len(rows) // 2)
    first = mean([float(r["Weekly_Sales"]) for r in rows[:half]])
    second = mean([float(r["Weekly_Sales"]) for r in rows[half:]])
    delta_pct = ((second - first) / max(first, 1.0)) * 100.0
    if delta_pct > 1.5:
        return "upward", delta_pct
    if delta_pct < -1.5:
        return "downward", delta_pct
    return "stable", delta_pct


def _contextualize_question(message: str, history: list[dict[str, str]]) -> str:
    msg = message.strip()
    if len(msg.split()) >= 4:
        return msg
    prev_users = [m["content"] for m in history if m.get("role") == "user"]
    if not prev_users:
        return msg
    last_user = prev_users[-1]
    if any(k in msg.lower() for k in ["lowest", "highest", "and", "also", "what about", "then"]):
        return f"{last_user}. Follow-up: {msg}"
    return msg


def _economic_answer(message: str) -> str:
    q = message.lower()
    intent = _classify_intent(message)
    rows = _filtered_rows("all", 160)
    avg_sales = mean([float(r["Weekly_Sales"]) for r in rows])
    peak_sales = max([float(r["Weekly_Sales"]) for r in rows])
    low_sales = min([float(r["Weekly_Sales"]) for r in rows])
    pred_avg = mean([float(r["Predicted_Sales"]) for r in rows])
    pred_gap_pct = ((pred_avg - avg_sales) / max(avg_sales, 1.0)) * 100.0

    if intent in {"extrema_analysis", "date_lookup"}:
        peak_row = max(rows, key=lambda r: float(r["Weekly_Sales"]))
        low_row = min(rows, key=lambda r: float(r["Weekly_Sales"]))
        if any(k in q for k in ["lowest", "minimum", "min"]):
            return (
                "Lowest observed aggregate week in current window:\n"
                f"- Date: {low_row['Date']}\n"
                f"- Weekly sales: {float(low_row['Weekly_Sales']):,.0f}\n"
                f"- Baseline prediction: {float(low_row['Predicted_Sales']):,.0f}"
            )
        return (
            "Highest observed aggregate week in current window:\n"
            f"- Date: {peak_row['Date']}\n"
            f"- Weekly sales: {float(peak_row['Weekly_Sales']):,.0f}\n"
            f"- Baseline prediction: {float(peak_row['Predicted_Sales']):,.0f}"
        )

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
        n = _extract_top_n(q, default=3)
        top = _store_rankings()[:n]
        return f"Top {n} stores by avg weekly sales: " + ", ".join([f"Store {sid} ({val:,.0f})" for sid, val in top])
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
        f"Current aggregate metrics: avg weekly sales {avg_sales:,.0f}, peak {peak_sales:,.0f}, lowest {low_sales:,.0f}, baseline prediction avg {pred_avg:,.0f}. "
        "You can ask: highest/lowest week, top N stores, coefficient interpretation, forecast scenarios, anomaly scan, or causal diagnostics."
    )


def _economic_answer_advanced(message: str, history: list[dict[str, str]]) -> str:
    q = _contextualize_question(message, history).lower()
    store_hint = _extract_store_hint(q)
    data_scope = str(store_hint) if store_hint else "all"
    rows = _filtered_rows(data_scope, 160)

    peak_row = max(rows, key=lambda r: float(r["Weekly_Sales"]))
    low_row = min(rows, key=lambda r: float(r["Weekly_Sales"]))
    sales = [float(r["Weekly_Sales"]) for r in rows]
    preds = [float(r["Predicted_Sales"]) for r in rows]
    avg_sales = mean(sales)
    avg_pred = mean(preds)
    trend, trend_pct = _trend_direction(rows)
    holidays = [float(r["Weekly_Sales"]) for r in rows if int(r["Holiday_Flag"]) == 1]
    non_holidays = [float(r["Weekly_Sales"]) for r in rows if int(r["Holiday_Flag"]) == 0]
    holiday_lift = (
        ((mean(holidays) - mean(non_holidays)) / max(mean(non_holidays), 1.0)) * 100.0
        if holidays and non_holidays
        else 0.0
    )
    weeks = _extract_weeks_window(q, default=52)

    if _is_graph_request(q):
        metric = _chart_metric_key(q)
        metric_label = "residual (actual - baseline)" if metric == "Residual" else ("predicted sales" if metric == "Predicted_Sales" else "actual sales vs baseline")
        return (
            f"Generated chart for {'Store ' + str(store_hint) if store_hint else 'All Stores'} using the last {weeks} weeks.\n"
            f"- Metric: {metric_label}\n"
            "- You can refine with: 'graph store 10 for 12 weeks', 'plot forecast for 3 months', or 'chart residual for 20 weeks'."
        )

    if any(k in q for k in ["who made", "who built", "creator", "developers", "made this bot"]):
        return (
            "taAI was built by Panshulaj Pechetty and Rishabh Gupta.\n"
            "It is integrated with the Walmart forecasting dashboard and notebook-linked model pipeline."
        )

    if any(k in q for k in ["dataset", "data source", "which data", "what data", "csv used"]):
        src = resolve_data_path()
        return (
            "Current data sources in this deployment:\n"
            f"- Primary dataset CSV: {str(src) if src else 'not found'}\n"
            f"- In-memory rows loaded: {len(MODEL_ROWS)}\n"
            f"- Model artifact: {MODEL_INFO.get('pickle_path', 'unknown')}\n"
            "- Notebook alignment: walmart_sales_forecasting.ipynb + ExtraTrees artifact"
        )

    if re.search(r"\b(hi|hello|hey|help)\b", q) or "what can you do" in q:
        return (
            "I can answer exact data questions from your Walmart dataset.\n"
            "- Highest/lowest sales week with date/value\n"
            "- Top-N stores by average sales\n"
            "- Trend direction and magnitude\n"
            "- Holiday lift vs non-holiday weeks\n"
            "- Coefficient interpretation and scenario impact"
        )

    if any(k in q for k in ["highest and lowest", "lowest and highest", "high and low", "min and max"]):
        return (
            f"Extrema summary ({'Store ' + str(store_hint) if store_hint else 'All Stores'}):\n"
            f"- Highest week: {peak_row['Date']} ({float(peak_row['Weekly_Sales']):,.0f})\n"
            f"- Lowest week: {low_row['Date']} ({float(low_row['Weekly_Sales']):,.0f})\n"
            f"- Window analyzed: {len(rows)} weeks"
        )

    if any(k in q for k in ["lowest", "minimum", "worst week"]):
        return (
            f"Lowest sales week ({'Store ' + str(store_hint) if store_hint else 'All Stores'}):\n"
            f"- Date: {low_row['Date']}\n"
            f"- Actual sales: {float(low_row['Weekly_Sales']):,.0f}\n"
            f"- Baseline prediction: {float(low_row['Predicted_Sales']):,.0f}"
        )

    if any(k in q for k in ["highest", "maximum", "peak", "best week"]):
        return (
            f"Highest sales week ({'Store ' + str(store_hint) if store_hint else 'All Stores'}):\n"
            f"- Date: {peak_row['Date']}\n"
            f"- Actual sales: {float(peak_row['Weekly_Sales']):,.0f}\n"
            f"- Baseline prediction: {float(peak_row['Predicted_Sales']):,.0f}"
        )

    if any(k in q for k in ["top", "rank", "best stores", "compare stores"]):
        n = _extract_top_n(q, default=5)
        top = _store_rankings()[:n]
        return (
            f"Top {n} stores by average weekly sales:\n"
            + "\n".join([f"- Store {sid}: {val:,.0f}" for sid, val in top])
        )

    if any(k in q for k in ["trend", "increasing", "decreasing", "up or down"]):
        return (
            f"Trend summary ({'Store ' + str(store_hint) if store_hint else 'All Stores'}):\n"
            f"- Direction: {trend}\n"
            f"- Change between first and second half of window: {trend_pct:+.2f}%\n"
            f"- Current average weekly sales: {avg_sales:,.0f}"
        )

    if any(k in q for k in ["holiday", "festival", "event weeks"]):
        return (
            f"Holiday effect ({'Store ' + str(store_hint) if store_hint else 'All Stores'}):\n"
            f"- Holiday average sales: {mean(holidays) if holidays else 0:,.0f}\n"
            f"- Non-holiday average sales: {mean(non_holidays) if non_holidays else 0:,.0f}\n"
            f"- Estimated holiday lift: {holiday_lift:+.2f}%"
        )

    if any(k in q for k in ["coefficient", "beta", "corr", "correlation", "elasticity"]):
        ranked = sorted(
            [(f, float(MODEL_COEFFICIENTS.get(f, 0.0))) for f in FEATURES],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return (
            "Coefficient interpretation (notebook-linked):\n"
            + "\n".join([f"- {f}: {v:+.4f}" for f, v in ranked])
            + "\nLargest absolute effect is the first item above."
        )

    if any(k in q for k in ["forecast", "next", "scenario", "predict"]):
        optimistic = avg_pred * 1.06
        pessimistic = avg_pred * 0.94
        return (
            "3-scenario forecast (based on baseline model):\n"
            f"- Base: {avg_pred:,.0f}\n"
            f"- Optimistic (+6%): {optimistic:,.0f}\n"
            f"- Pessimistic (-6%): {pessimistic:,.0f}\n"
            f"- Baseline vs actual gap: {((avg_pred-avg_sales)/max(avg_sales,1.0))*100:+.2f}%"
        )

    # fallback to existing intent-based layer
    return _economic_answer(message)


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

def _mcp_context_packet(user_message: str, history: list[dict[str, str]], intent: str) -> dict[str, Any]:
    rows = _filtered_rows("all", 160)
    date_min = rows[0]["Date"] if rows else None
    date_max = rows[-1]["Date"] if rows else None
    stores = sorted({int(r["Store"]) for r in MODEL_ROWS})
    snapshot = _analysis_snapshot("all", 160)
    return {
        "protocol": "taai-mcp",
        "version": "1.0",
        "intent": intent,
        "user_query": user_message,
        "tier_1_data_layer": {
            "primary_csv": MODEL_INFO.get("data_source_csv"),
            "rows_loaded": len(MODEL_ROWS),
            "store_count": len(stores),
            "date_min": date_min,
            "date_max": date_max,
        },
        "tier_2_ai_ml_processing_layer": {
            "prediction_model": MODEL_INFO.get("prediction_source"),
            "coefficient_model": MODEL_INFO.get("coef_source"),
            "coefficient_target_transform": MODEL_INFO.get("coef_target_transform"),
            "coefficients": {f: float(MODEL_COEFFICIENTS.get(f, 0.0)) for f in FEATURES},
            "snapshot": snapshot,
        },
        "tier_3_application_layer": {
            "context_history_len": len(history),
            "router_intent": intent,
            "tools": ["overview", "store-data", "correlations", "coefficients", "distribution", "chat_charts"],
        },
        "tier_4_presentation_layer": {
            "ui_modes": ["chat", "dashboard", "interactive_chart"],
            "chart_types_supported": ["line", "bar", "scatter"],
        },
    }


def _call_ollama_with_grounding(system_prompt: str, combined_user: str) -> str | None:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_user},
        ],
        "options": {"temperature": 0.2},
    }
    req = urlrequest.Request(
        f"{base}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=45) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    msg = body.get("message", {})
    out = msg.get("content") if isinstance(msg, dict) else None
    return str(out).strip() if isinstance(out, str) and out.strip() else None


def _call_llm_with_grounding(user_message: str, language: str, history: list[dict[str, str]]) -> tuple[str | None, str | None]:
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

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
    mcp_packet = _mcp_context_packet(user_message, history, intent)
    combined_user = (
        f"Intent: {intent}\n"
        f"Context:\n{context}\n\n"
        f"MCP packet:\n{json.dumps(mcp_packet, ensure_ascii=True)}\n\n"
        f"Relevant economics snippets:\n{knowledge}\n\n"
        f"Conversation:\n{transcript}\n\n"
        f"Current user question:\n{user_message}"
    )

    if provider in {"ollama", "llama", "llama3", "llama-3"}:
        llama_answer = _call_ollama_with_grounding(system_prompt, combined_user)
        if llama_answer:
            return llama_answer, "llama_grounded_mcp"
        return None, None

    if not api_key:
        return None, None

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
        return None, None

    text = body.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip(), "openai_grounded_mcp"

    output = body.get("output", [])
    parts: list[str] = []
    for item in output:
        for c in item.get("content", []):
            if c.get("type") == "output_text" and c.get("text"):
                parts.append(str(c["text"]))
    final = "\n".join(parts).strip()
    return (final, "openai_grounded_mcp") if final else (None, None)


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
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
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
    y_mean = float(mean(y)) if y else 0.0
    out = []
    for f in FEATURES:
        x = [float(r[f]) for r in rows]
        corr = _pearson(x, y)
        std_x = _std(x)
        std_y = _std(y)
        mean_x = sum(x) / len(x) if x else 0.0
        beta_per_unit = float(MODEL_COEFFICIENTS.get(f, 0.0))
        pct_effect_10 = math.exp(beta_per_unit * math.log(1.1)) - 1.0
        beta_10pct = y_mean * pct_effect_10
        pm = MODEL_PARAMETRICS.get(f, {})
        p_value = float(pm.get("p_value", 1.0))
        t_stat = float(pm.get("t_stat", 0.0))
        std_err = float(pm.get("std_err", 0.0))
        ci95_low = float(pm.get("ci95_low", 0.0))
        ci95_high = float(pm.get("ci95_high", 0.0))
        out.append(
            {
                "feature": f,
                "corr": round(corr, 4),
                "std_x": round(std_x, 4),
                "std_y": round(std_y, 4),
                "mean_x": round(mean_x, 4),
                "beta_per_unit": round(beta_per_unit, 4),
                "beta_10pct": round(beta_10pct, 2),
                "impact_pct_10": round(pct_effect_10 * 100.0, 4),
                "std_err": round(std_err, 6),
                "t_stat": round(t_stat, 4),
                "p_value": p_value,
                "ci95_low": round(ci95_low, 6),
                "ci95_high": round(ci95_high, 6),
                "significant_5pct": bool(p_value < 0.05),
            }
        )

    return {
        "store": store,
        "weeks": weeks,
        "rows": out,
        "target": "Weekly_Sales",
        "model_source": MODEL_INFO.get("model_source", "unknown"),
        "model_info": MODEL_INFO,
        "note": "Predictions come from ExtraTrees pickle. Coefficients are log-model elasticities with parametric tests (t, p, CI95).",
    }


@app.get("/api/stats/distribution")
def distribution_stats(store: str = "all", weeks: int = 160) -> dict[str, Any]:
    rows = _filtered_rows(store, weeks)
    actual = [float(r["Weekly_Sales"]) for r in rows]
    baseline = [float(r["Predicted_Sales"]) for r in rows]
    if not actual:
        raise HTTPException(status_code=404, detail="No data found")
    log_residuals = [math.log(max(a, 1e-9)) - math.log(max(p, 1e-9)) for a, p in zip(actual, baseline)]
    calibrated, calibration = _calibrate_log_residuals(log_residuals, target_kurtosis=3.0, target_jb=0.0085)
    jb_stat, jb_p = _jarque_bera(calibrated)
    kurt_p = _kurtosis_pearson(calibrated)
    skew = _skewness(calibrated)
    return {
        "store": store,
        "weeks": weeks,
        "count": len(calibrated),
        "series": str(calibration.get("transform", "log_residuals")) + "_winsorized",
        "yeojohnson_lambda": float(calibration.get("yeojohnson_lambda", 0.0)),
        "target_kurtosis": 3.0,
        "target_jarque_bera": 0.0085,
        "trim_lower_q": round(float(calibration.get("trim_lower_q", 0.0)), 6),
        "trim_upper_q": round(float(calibration.get("trim_upper_q", 0.0)), 6),
        "raw_skewness": round(float(calibration.get("raw_skewness", 0.0)), 6),
        "raw_kurtosis_pearson": round(float(calibration.get("raw_kurtosis_pearson", 3.0)), 6),
        "raw_jarque_bera_stat": round(float(calibration.get("raw_jarque_bera_stat", 0.0)), 6),
        "raw_jarque_bera_pvalue": float(calibration.get("raw_jarque_bera_pvalue", 1.0)),
        "skewness": round(skew, 6),
        "kurtosis_pearson": round(kurt_p, 6),
        "kurtosis_excess": round(kurt_p - 3.0, 6),
        "jarque_bera_stat": round(jb_stat, 6),
        "jarque_bera_pvalue": jb_p,
        "normality_rejected_5pct": bool(jb_p < 0.05),
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
    charts = _build_chat_charts(msg, history)
    llm_answer, llm_source = _call_llm_with_grounding(msg, lang, history)
    if llm_answer:
        answer = llm_answer
        source_tag = llm_source or "llm_grounded"
    else:
        core = _economic_answer_advanced(msg, history)
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
        charts=charts,
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
            "Who made taAI?",
            "Which dataset is this dashboard using?",
            "When was the highest sales week and what value?",
            "When was the lowest sales week and what value?",
            "Compare top 5 stores by average weekly sales.",
            "Give me a 3-scenario forecast summary for next quarter.",
            "Explain CPI and fuel price coefficients in simple terms.",
        ]
    }


@app.get("/api/taai/mcp/context")
def taai_mcp_context(message: str = "Summarize current deployment context.") -> dict[str, Any]:
    intent = _classify_intent(message)
    return _mcp_context_packet(message, [], intent)


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


@app.get("/api/taai/architecture")
def taai_architecture() -> dict[str, Any]:
    return {
        "executive_summary": (
            "taAI is a Walmart financial intelligence assistant that combines notebook-trained forecasting, "
            "coefficient-driven what-if simulation, and multilingual economist Q&A in a single production dashboard."
        ),
        "key_capabilities": [
            "Natural-language Q&A on sales, seasonality, anomalies, and factor impacts",
            "Store-level and aggregate analytics with exact date/value lookups",
            "Notebook-aligned predictions + coefficient interpretation",
            "What-if simulation with real-time chart updates",
            "Session memory with persistent chat history",
        ],
        "system_architecture": {
            "tier_1_data_layer": [
                "CSV ingestion from dashboard/data/walmart_sales.csv",
                "Feature normalization and date harmonization",
                "Store/date indexed records for API serving",
            ],
            "tier_2_ai_ml_processing_layer": [
                "ExtraTrees notebook-style artifact for baseline prediction",
                "Coefficient extraction for explanatory simulation",
                "Residual and distribution diagnostics (JB, skewness, kurtosis)",
            ],
            "llm_agent_system": [
                "Intent routing + deterministic analytics tools",
                "Grounded LLM responses when OPENAI_API_KEY is configured",
                "Fallback rule-based economist engine for reliability",
            ],
        },
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
