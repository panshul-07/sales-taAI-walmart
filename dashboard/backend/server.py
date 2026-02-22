from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
DATA_CANDIDATES = [
    Path(os.getenv("DATA_PATH", "")) if os.getenv("DATA_PATH") else None,
    BASE_DIR / "data" / "walmart_sales.csv",
    BASE_DIR.parent / "data" / "walmart_sales.csv",
]

FEATURES = [
    "Store",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "year",
    "month",
    "weekofyear",
    "quarter",
    "week_sin",
    "week_cos",
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_4",
    "sales_roll4_mean",
]


class ModelBundle:
    def __init__(self, raw_df: pd.DataFrame, model_df: pd.DataFrame, model: Pipeline):
        self.raw_df = raw_df
        self.model_df = model_df
        self.model = model


def _generate_demo_data() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = pd.Timestamp("2010-02-05")
    weeks = 170
    rng = np.random.default_rng(42)

    for store in range(1, 46):
        for i in range(weeks):
            dt = start + pd.Timedelta(days=7 * i)
            season = math.sin((i / 52.0) * 2 * math.pi)
            holiday = 1 if (i % 26 == 0 or i % 26 == 25) else 0
            fuel = 2.45 + ((i % 30) / 30.0) * 1.15
            cpi = 205.0 + i * 0.11 + (store % 5) * 0.2
            unemp = 9.1 - i * 0.012 + (store % 7) * 0.03
            temp = 56 + math.sin((i / 52.0) * 2 * math.pi + store * 0.08) * 22
            base = 150000 + store * 1100 + season * 17000 + holiday * 21000
            sales = max(15000, base - fuel * 5200 - unemp * 1800 + cpi * 68 + rng.normal(0, 5500))
            rows.append(
                {
                    "Store": store,
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Weekly_Sales": round(float(sales), 2),
                    "Holiday_Flag": holiday,
                    "Temperature": round(float(temp), 3),
                    "Fuel_Price": round(float(fuel), 3),
                    "CPI": round(float(cpi), 3),
                    "Unemployment": round(float(unemp), 3),
                }
            )

    return pd.DataFrame(rows)


def _load_raw_data() -> pd.DataFrame:
    for candidate in DATA_CANDIDATES:
        if candidate and candidate.exists():
            df = pd.read_csv(candidate)
            break
    else:
        df = _generate_demo_data()

    required = {
        "Store",
        "Date",
        "Weekly_Sales",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def _build_model_bundle() -> ModelBundle:
    raw_df = _load_raw_data()
    df = raw_df.copy()

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["Date"].dt.quarter
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52.0)

    df["sales_lag_1"] = df.groupby("Store")["Weekly_Sales"].shift(1)
    df["sales_lag_2"] = df.groupby("Store")["Weekly_Sales"].shift(2)
    df["sales_lag_4"] = df.groupby("Store")["Weekly_Sales"].shift(4)
    df["sales_roll4_mean"] = df.groupby("Store")["Weekly_Sales"].shift(1).rolling(4).mean()
    df = df.dropna().reset_index(drop=True)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                [c for c in FEATURES if c != "Store"],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Store"]),
        ]
    )

    reg = ExtraTreesRegressor(
        n_estimators=400,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=-1,
    )

    model = Pipeline([("prep", preprocessor), ("reg", reg)])
    model.fit(df[FEATURES], df["Weekly_Sales"])

    df = df.copy()
    df["Predicted_Sales"] = model.predict(df[FEATURES]).astype(float)
    return ModelBundle(raw_df=raw_df, model_df=df, model=model)


bundle = _build_model_bundle()
app = FastAPI(title="Walmart Forecast API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stores")
def stores() -> list[dict[str, Any]]:
    g = (
        bundle.model_df.groupby("Store", as_index=False)
        .agg(records=("Weekly_Sales", "count"), avg_sales=("Weekly_Sales", "mean"))
        .sort_values("Store")
    )
    out = []
    for _, row in g.iterrows():
        out.append(
            {
                "Store": int(row["Store"]),
                "records": int(row["records"]),
                "avg_sales": round(float(row["avg_sales"]), 2),
            }
        )
    return out


def _filtered_df(store: str, weeks: int) -> pd.DataFrame:
    df = bundle.model_df
    if store != "all":
        try:
            sid = int(store)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid store") from exc
        df = df[df["Store"] == sid]

    df = df.sort_values("Date").tail(weeks)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    return df


@app.get("/api/overview")
def overview(store: str = "all", weeks: int = 160) -> dict[str, Any]:
    df = _filtered_df(store, weeks)
    holiday_sales = df[df["Holiday_Flag"] == 1]["Weekly_Sales"]
    return {
        "store": store,
        "records": int(len(df)),
        "date_min": str(df["Date"].min().date()),
        "date_max": str(df["Date"].max().date()),
        "avg_weekly_sales": round(float(df["Weekly_Sales"].mean()), 2),
        "peak_sales": round(float(df["Weekly_Sales"].max()), 2),
        "holiday_avg": round(float(holiday_sales.mean() if len(holiday_sales) else 0.0), 2),
        "holiday_count": int((df["Holiday_Flag"] == 1).sum()),
    }


@app.get("/api/store-data")
def store_data(store: str = "all", weeks: int = 160) -> list[dict[str, Any]]:
    df = _filtered_df(store, weeks)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "Store": int(r["Store"]),
                "Date": r["Date"].strftime("%Y-%m-%d"),
                "Weekly_Sales": round(float(r["Weekly_Sales"]), 2),
                "Predicted_Sales": round(float(r["Predicted_Sales"]), 2),
                "Holiday_Flag": int(r["Holiday_Flag"]),
                "Temperature": round(float(r["Temperature"]), 3),
                "Fuel_Price": round(float(r["Fuel_Price"]), 3),
                "CPI": round(float(r["CPI"]), 3),
                "Unemployment": round(float(r["Unemployment"]), 3),
            }
        )
    return rows


@app.get("/api/correlations")
def correlations(store: str = "all", weeks: int = 160) -> list[dict[str, Any]]:
    df = _filtered_df(store, weeks)
    y = df["Weekly_Sales"].astype(float)
    features = ["CPI", "Unemployment", "Fuel_Price", "Temperature"]
    out = []
    for f in features:
        c = float(np.corrcoef(df[f].astype(float), y)[0, 1]) if len(df) > 1 else 0.0
        if np.isnan(c):
            c = 0.0
        out.append({"feature": f, "corr": round(c, 4)})
    return out


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/{full_path:path}")
def spa_fallback(full_path: str) -> FileResponse:
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")
    target = STATIC_DIR / full_path
    if target.exists() and target.is_file():
        return FileResponse(target)
    return FileResponse(STATIC_DIR / "index.html")
