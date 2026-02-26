from __future__ import annotations

import csv
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "backend" / "model_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "extra_trees_notebook.pkl"
FEATURES = ["CPI", "Unemployment", "Fuel_Price", "Temperature"]
DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%Y", "%m/%d/%Y")


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


def generate_demo_data() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = datetime(2010, 2, 5)
    weeks = 170
    for store in range(1, 46):
        for i in range(weeks):
            dt = start + timedelta(days=7 * i)
            season = np.sin((i / 52.0) * 2 * np.pi)
            holiday = 1 if (i % 26 == 0 or i % 26 == 25) else 0
            fuel = 2.45 + ((i % 30) / 30.0) * 1.15
            cpi = 205.0 + i * 0.11 + (store % 5) * 0.2
            unemp = 9.1 - i * 0.012 + (store % 7) * 0.03
            temp = 56 + np.sin((i / 52.0) * 2 * np.pi + store * 0.08) * 22
            base = 150000 + store * 1100 + season * 17000 + holiday * 21000
            noise = np.sin(i * 0.31 + store * 0.7) * 4200
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


def resolve_data_path() -> Path | None:
    candidates = [
        Path(os.getenv("DATA_PATH", "")) if os.getenv("DATA_PATH") else None,
        BASE_DIR / "data" / "walmart_sales.csv",
        BASE_DIR.parent / "data" / "walmart_sales.csv",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _parse_date(value: Any) -> datetime | None:
    d = str(value or "").strip()
    if not d:
        return None
    try:
        return datetime.fromisoformat(d)
    except ValueError:
        pass
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    return None


def load_csv_data() -> list[dict[str, Any]]:
    candidate = resolve_data_path()
    if candidate and candidate.exists():
        rows: list[dict[str, Any]] = []
        with candidate.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r:
                    continue
                dt = _parse_date(r.get("Date"))
                if dt is None:
                    continue
                row = {
                    "Store": _safe_int(r.get("Store")),
                    "Date": dt.strftime("%Y-%m-%d"),
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
    return generate_demo_data()


def feature_engineer(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["Date"].dt.quarter
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    for lag in [1, 2, 4, 8]:
        df[f"sales_lag_{lag}"] = df.groupby("Store")["Weekly_Sales"].shift(lag)
    df["sales_roll4_mean"] = df.groupby("Store")["Weekly_Sales"].shift(1).rolling(4).mean()
    df["sales_roll4_std"] = df.groupby("Store")["Weekly_Sales"].shift(1).rolling(4).std()
    return df.dropna().reset_index(drop=True)


def train_artifact() -> dict[str, Any]:
    raw_rows = load_csv_data()
    df = feature_engineer(raw_rows)

    feature_cols = [
        "Store", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "year", "month", "weekofyear", "quarter", "is_month_start", "is_month_end",
        "week_sin", "week_cos", "sales_lag_1", "sales_lag_2", "sales_lag_4", "sales_lag_8",
        "sales_roll4_mean", "sales_roll4_std",
    ]

    X = df[feature_cols].copy()
    y = df["Weekly_Sales"].copy()

    numeric_features = [c for c in feature_cols if c != "Store"]
    categorical_features = ["Store"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = ExtraTreesRegressor(
        n_estimators=260,
        max_depth=16,
        min_samples_leaf=4,
        min_samples_split=2,
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X, y)

    pred = pipe.predict(X)
    pred_df = df[["Store", "Date"]].copy()
    pred_df["Predicted_Sales"] = pred
    pred_map = {
        (int(r.Store), pd.Timestamp(r.Date).strftime("%Y-%m-%d")): float(r.Predicted_Sales)
        for r in pred_df.itertuples(index=False)
    }

    # Notebook-like demand-equation coefficient extraction using linear regression on core terms.
    core_terms = [
        "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "sales_lag_1", "sales_lag_4", "sales_roll4_mean", "sales_roll4_std", "week_sin", "week_cos",
    ]
    reg = LinearRegression()
    reg.fit(df[core_terms], df["Weekly_Sales"])
    coef_map = {term: float(val) for term, val in zip(core_terms, reg.coef_)}
    feature_coef = {
        "CPI": coef_map.get("CPI", 0.0),
        "Unemployment": coef_map.get("Unemployment", 0.0),
        "Fuel_Price": coef_map.get("Fuel_Price", 0.0),
        "Temperature": coef_map.get("Temperature", 0.0),
    }

    artifact = {
        "source": "extra_trees_notebook",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_cols,
        "pred_map": pred_map,
        "coef_table": coef_map,
        "feature_coefficients": feature_coef,
        "rows_fit": int(len(df)),
        "r2_train": float(pipe.score(X, y)),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("wb") as f:
        pickle.dump(artifact, f)
    return artifact


if __name__ == "__main__":
    art = train_artifact()
    print("saved", ARTIFACT_PATH)
    print("source", art["source"])
    print("rows_fit", art["rows_fit"])
    print("r2_train", round(art["r2_train"], 4))
