from __future__ import annotations

import csv
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"


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
        if not candidate:
            continue
        if not candidate.exists():
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
                if row["Store"] <= 0:
                    continue
                rows.append(row)

        if rows:
            rows.sort(key=lambda x: (x["Store"], x["Date"]))
            return rows

    return _generate_demo_data()


def _predict_series(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_store: dict[int, list[float]] = {}
    out: list[dict[str, Any]] = []

    for row in sorted(rows, key=lambda r: (r["Store"], r["Date"])):
        store = int(row["Store"])
        history = by_store.setdefault(store, [])
        if len(history) < 4:
            pred = float(row["Weekly_Sales"])
        else:
            recent = history[-4:]
            drift = (recent[-1] - recent[0]) / 3.0
            pred = mean(recent) + 0.25 * drift
        history.append(float(row["Weekly_Sales"]))

        new_row = dict(row)
        new_row["Predicted_Sales"] = round(max(1000.0, pred), 2)
        out.append(new_row)

    return out


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
    if den == 0:
        return 0.0
    return num / den


RAW_ROWS = _load_csv_data()
MODEL_ROWS = _predict_series(RAW_ROWS)

app = FastAPI(title="Walmart Forecast API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "rows": len(MODEL_ROWS)}


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


def _filtered_rows(store: str, weeks: int) -> list[dict[str, Any]]:
    if store == "all":
        data = MODEL_ROWS
    else:
        try:
            sid = int(store)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid store") from exc
        data = [r for r in MODEL_ROWS if int(r["Store"]) == sid]

    if not data:
        raise HTTPException(status_code=404, detail="No data found")

    data = sorted(data, key=lambda r: r["Date"])[-max(1, weeks) :]
    return data


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
    fields = ["CPI", "Unemployment", "Fuel_Price", "Temperature"]
    out = []
    for f in fields:
        x = [float(r[f]) for r in rows]
        out.append({"feature": f, "corr": round(_pearson(x, y), 4)})
    return out


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
