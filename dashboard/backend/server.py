from __future__ import annotations

import csv
import math
import os
import pickle
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = BASE_DIR / "backend" / "model_artifacts"
MODEL_PATH = MODEL_DIR / "sales_log_linear.pkl"
FEATURES = ["CPI", "Unemployment", "Fuel_Price", "Temperature"]

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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _generate_demo_data() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = datetime(2010, 2, 5)
    weeks = 170

    for store in range(1, 46):
        for i in range(weeks):
            dt = start + timedelta(days=7 * i)
            season = math.sin((i / 52.0) * 2 * math.pi)
            holiday = 1 if (i % 26 == 0 or i % 26 == 25) else 0
            fuel = 2.45 + ((i % 30) / 30.0) * 1.15
            cpi = 205.0 + i * 0.11 + (store % 5) * 0.2
            unemp = 9.1 - i * 0.012 + (store % 7) * 0.03
            temp = 56 + math.sin((i / 52.0) * 2 * math.pi + store * 0.08) * 22
            base = 150000 + store * 1100 + season * 17000 + holiday * 21000
            noise = math.sin(i * 0.31 + store * 0.7) * 4200
            sales = max(15000, base - fuel * 5200 - unemp * 1800 + cpi * 68 + noise)
            rows.append(
                {
                    "Store": store,
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Weekly_Sales": round(sales, 2),
                    "Holiday_Flag": holiday,
                    "Temperature": round(temp, 3),
                    "Fuel_Price": round(fuel, 3),
                    "CPI": round(cpi, 3),
                    "Unemployment": round(unemp, 3),
                }
            )
    return rows


def _load_csv_data() -> list[dict[str, Any]]:
    candidates = [
        Path(os.getenv("DATA_PATH", "")) if os.getenv("DATA_PATH") else None,
        BASE_DIR / "data" / "walmart_sales.csv",
        BASE_DIR.parent / "data" / "walmart_sales.csv",
    ]

    for candidate in candidates:
        if not candidate or not candidate.exists():
            continue

        rows: list[dict[str, Any]] = []
        with candidate.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r:
                    continue
                date_raw = str(r.get("Date", "")).strip()
                try:
                    parsed_date = datetime.fromisoformat(date_raw)
                except ValueError:
                    try:
                        parsed_date = datetime.strptime(date_raw, "%Y-%m-%d")
                    except ValueError:
                        continue

                row = {
                    "Store": _safe_int(r.get("Store")),
                    "Date": parsed_date.strftime("%Y-%m-%d"),
                    "Weekly_Sales": _safe_float(r.get("Weekly_Sales")),
                    "Holiday_Flag": _safe_int(r.get("Holiday_Flag")),
                    "Temperature": _safe_float(r.get("Temperature")),
                    "Fuel_Price": _safe_float(r.get("Fuel_Price")),
                    "CPI": _safe_float(r.get("CPI")),
                    "Unemployment": _safe_float(r.get("Unemployment")),
                }
                if row["Store"] > 0:
                    rows.append(row)

        if rows:
            rows.sort(key=lambda x: (x["Store"], x["Date"]))
            return rows

    return _generate_demo_data()


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


def _feature_engineering(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, float]:
    df = pd.DataFrame(rows).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    df["trend"] = (df["Date"] - df["Date"].min()).dt.days

    df["log_cpi"] = np.log(np.clip(df["CPI"], 1e-9, None))
    df["log_unemployment"] = np.log(np.clip(df["Unemployment"], 1e-9, None))
    df["log_fuel"] = np.log(np.clip(df["Fuel_Price"], 1e-9, None))

    temp_shift = float(abs(df["Temperature"].min()) + 1.0)
    df["log_temperature"] = np.log(np.clip(df["Temperature"] + temp_shift, 1e-9, None))

    df["log_sales"] = np.log(np.clip(df["Weekly_Sales"], 1e-9, None))
    df["log_sales_lag_1"] = df.groupby("Store")["log_sales"].shift(1)
    df["log_sales_lag_4"] = df.groupby("Store")["log_sales"].shift(4)
    df["log_sales_roll4_mean"] = df.groupby("Store")["log_sales"].shift(1).rolling(4).mean()
    df["log_sales_roll4_std"] = df.groupby("Store")["log_sales"].shift(1).rolling(4).std()

    return df, temp_shift


def _train_and_pickle_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df, temp_shift = _feature_engineering(rows)
    model_df = df.dropna().copy()
    if len(model_df) < 160:
        raise RuntimeError("insufficient rows after feature engineering")

    feature_cols = [
        "Holiday_Flag",
        "log_cpi",
        "log_unemployment",
        "log_fuel",
        "log_temperature",
        "trend",
        "week_sin",
        "week_cos",
        "log_sales_lag_1",
        "log_sales_lag_4",
        "log_sales_roll4_mean",
        "log_sales_roll4_std",
    ]

    X_core = model_df[feature_cols].copy()
    X_store = pd.get_dummies(model_df["Store"].astype(str), prefix="store", drop_first=False)
    X = pd.concat([X_core, X_store], axis=1)
    y = model_df["log_sales"].values

    model = LinearRegression()
    model.fit(X, y)
    pred_log = model.predict(X)
    model_df["Predicted_Sales"] = np.exp(pred_log)

    pred_map: dict[tuple[int, str], float] = {}
    for _, r in model_df[["Store", "Date", "Predicted_Sales"]].iterrows():
        pred_map[(int(r["Store"]), pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"))] = float(r["Predicted_Sales"])

    coef_lookup = {name: float(val) for name, val in zip(X.columns, model.coef_)}
    elasticity = {
        "CPI": coef_lookup.get("log_cpi", 0.0),
        "Unemployment": coef_lookup.get("log_unemployment", 0.0),
        "Fuel_Price": coef_lookup.get("log_fuel", 0.0),
        "Temperature": coef_lookup.get("log_temperature", 0.0),
    }

    artifact = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "feature_columns": list(X.columns),
        "core_features": feature_cols,
        "temp_shift": temp_shift,
        "pred_map": pred_map,
        "elasticity": elasticity,
        "rows_fit": int(len(model_df)),
        "r2_train": float(model.score(X, y)),
        "source": "pickled_log_linear",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(artifact, f)

    return artifact


def _load_or_train_artifact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if MODEL_PATH.exists():
        try:
            with MODEL_PATH.open("rb") as f:
                artifact = pickle.load(f)
            if isinstance(artifact, dict) and artifact.get("source") == "pickled_log_linear":
                return artifact
        except Exception:
            pass
    return _train_and_pickle_model(rows)


def _build_runtime_state(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    try:
        artifact = _load_or_train_artifact(rows)
    except Exception as exc:
        fallback_rows = []
        by_store: dict[int, list[float]] = {}
        for row in sorted(rows, key=lambda r: (r["Store"], r["Date"])):
            s = int(row["Store"])
            hist = by_store.setdefault(s, [])
            pred = float(row["Weekly_Sales"]) if len(hist) < 4 else mean(hist[-4:])
            hist.append(float(row["Weekly_Sales"]))
            nr = dict(row)
            nr["Predicted_Sales"] = round(max(1000.0, pred), 2)
            fallback_rows.append(nr)

        coeffs = {}
        for f in FEATURES:
            x = [float(v[f]) for v in fallback_rows]
            y = [float(v["Weekly_Sales"]) for v in fallback_rows]
            corr = _pearson(x, y)
            std_x = _std(x)
            std_y = _std(y)
            coeffs[f] = 0.0 if std_x == 0 else corr * (std_y / std_x)

        info = {
            "model_source": "heuristic_fallback",
            "reason": str(exc),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "retrain_cron": "every_6_hours",
        }
        return fallback_rows, coeffs, info

    pred_map = artifact.get("pred_map", {})
    out_rows: list[dict[str, Any]] = []
    for r in rows:
        nr = dict(r)
        key = (int(nr["Store"]), str(nr["Date"]))
        nr["Predicted_Sales"] = round(max(1000.0, float(pred_map.get(key, nr["Weekly_Sales"]))), 2)
        out_rows.append(nr)

    info = {
        "model_source": "pickled_log_linear",
        "trained_at": artifact.get("trained_at"),
        "rows_fit": artifact.get("rows_fit"),
        "r2_train": artifact.get("r2_train"),
        "pickle_path": str(MODEL_PATH.relative_to(BASE_DIR)),
        "retrain_cron": "every_6_hours",
    }
    return out_rows, artifact.get("elasticity", {}), info


def _refresh_model_state() -> None:
    global MODEL_ROWS, MODEL_COEFFICIENTS, MODEL_INFO
    with MODEL_LOCK:
        raw = _load_csv_data()
        rows, coefs, info = _build_runtime_state(raw)
        MODEL_ROWS = rows
        MODEL_COEFFICIENTS = coefs
        MODEL_INFO = info


def _start_scheduler() -> None:
    global SCHEDULER
    if SCHEDULER is not None:
        return
    SCHEDULER = BackgroundScheduler(timezone="UTC")
    SCHEDULER.add_job(_refresh_model_state, "interval", hours=6, id="retrain_sales_model", replace_existing=True)
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
    if re.search(r"[\u0900-\u097f]", text):
        return "hindi"
    if any(k in t for k in ["kya", "kaise", "kyun", "kyu", "batao", "samjhao", "hai", "nahi"]):
        return "hinglish"
    return "english"


def _economic_answer(message: str) -> str:
    q = message.lower()
    rows = _filtered_rows("all", 160)
    avg_sales = mean([float(r["Weekly_Sales"]) for r in rows])
    peak_sales = max([float(r["Weekly_Sales"]) for r in rows])

    if "coefficient" in q or "beta" in q or "elastic" in q:
        lines = ["Current elasticity-style coefficients from the deployed log model:"]
        for f in FEATURES:
            b = float(MODEL_COEFFICIENTS.get(f, 0.0))
            lines.append(f"- {f}: {b:.4f} (approx. % sales change for 1% {f} change)")
        return "\n".join(lines)

    if "forecast" in q or "predict" in q:
        pred_avg = mean([float(r["Predicted_Sales"]) for r in rows])
        gap = pred_avg - avg_sales
        return (
            f"Baseline forecast over the selected horizon is {pred_avg:,.0f} average weekly sales. "
            f"Compared to actual average {avg_sales:,.0f}, model gap is {gap:,.0f}. "
            "Use what-if controls to simulate macro-factor shocks."
        )

    if "compare" in q or "store" in q:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for r in MODEL_ROWS:
            grouped.setdefault(int(r["Store"]), []).append(r)
        top = sorted(
            [(sid, mean([float(x["Weekly_Sales"]) for x in vals])) for sid, vals in grouped.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        return "Top stores by average sales: " + ", ".join([f"Store {sid} ({v:,.0f})" for sid, v in top])

    return (
        f"Average weekly sales are {avg_sales:,.0f} and peak weekly sales are {peak_sales:,.0f}. "
        "Ask for forecast, coefficient interpretation, scenario impact, store comparisons, anomalies, or macro-factor analysis."
    )


app = FastAPI(title="Walmart Forecast API", version="5.0.0")
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
    mean_sales = sum(y) / len(y) if y else 0.0
    out = []

    for f in FEATURES:
        x = [float(r[f]) for r in rows]
        corr = _pearson(x, y)
        std_x = _std(x)
        std_y = _std(y)
        mean_x = sum(x) / len(x) if x else 0.0
        beta_log = float(MODEL_COEFFICIENTS.get(f, 0.0))

        beta_per_unit = 0.0 if mean_x == 0 else beta_log * (mean_sales / mean_x)
        beta_10pct = beta_per_unit * (0.1 * mean_x)
        pct_impact_10pct = beta_log * 0.10 * 100.0

        out.append(
            {
                "feature": f,
                "corr": round(corr, 4),
                "std_x": round(std_x, 4),
                "std_y": round(std_y, 4),
                "mean_x": round(mean_x, 4),
                "beta_log": round(beta_log, 4),
                "beta_per_unit": round(beta_per_unit, 4),
                "beta_10pct": round(beta_10pct, 2),
                "pct_impact_10pct": round(pct_impact_10pct, 3),
            }
        )

    return {
        "store": store,
        "weeks": weeks,
        "rows": out,
        "target": "Weekly_Sales",
        "model_source": MODEL_INFO.get("model_source", "unknown"),
        "model_info": MODEL_INFO,
        "note": "Model is trained on natural-log transformed target/features and persisted as pickle. beta_log is elasticity-style coefficient.",
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

    if lang == "hinglish":
        answer = f"taAI insight: {core}\n\nAgar chaho toh main isi ko detailed scenario analysis mein tod sakta hoon."
    elif lang == "hindi":
        answer = f"taAI विश्लेषण: {core}\n\nअगर चाहें तो मैं इसे और विस्तृत परिदृश्य विश्लेषण में बदल सकता हूँ।"
    else:
        answer = f"taAI insight: {core}"

    history.append({"role": "assistant", "content": answer})
    if len(history) > 16:
        CHAT_SESSIONS[session_id] = history[-16:]

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        detected_language=lang,
        sources=[
            "walmart_sales_forecasting.ipynb",
            "Walmart_Financial_Chatbot_Architecture.docx",
            "runtime_sales_model_pickle",
        ],
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
