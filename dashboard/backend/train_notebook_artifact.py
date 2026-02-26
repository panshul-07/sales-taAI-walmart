from __future__ import annotations

import csv
import math
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
    df = pd.DataFrame.from_records(rows)
    date_col = pd.to_datetime(df.pop("Date"), errors="coerce")
    df.insert(1, "Date", date_col)
    df = df.dropna(subset=["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    df.loc[:, "year"] = df["Date"].dt.year
    df.loc[:, "month"] = df["Date"].dt.month
    df.loc[:, "weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df.loc[:, "quarter"] = df["Date"].dt.quarter
    df.loc[:, "is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df.loc[:, "is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df.loc[:, "week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df.loc[:, "week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    for lag in [1, 2, 4, 8]:
        df.loc[:, f"sales_lag_{lag}"] = df.groupby("Store")["Weekly_Sales"].shift(lag)
    df.loc[:, "sales_roll4_mean"] = df.groupby("Store")["Weekly_Sales"].shift(1).rolling(4).mean()
    df.loc[:, "sales_roll4_std"] = df.groupby("Store")["Weekly_Sales"].shift(1).rolling(4).std()
    return df.dropna().reset_index(drop=True)


def _student_t_sf(value: float, dof: int) -> float:
    try:
        from scipy.stats import t as student_t

        return float(student_t.sf(abs(value), dof))
    except Exception:
        return 0.5 * math.erfc(abs(value) / math.sqrt(2.0))


def _student_t_ppf(prob: float, dof: int) -> float:
    try:
        from scipy.stats import t as student_t

        return float(student_t.ppf(prob, dof))
    except Exception:
        return 1.959963984540054


def _ols_inference(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    x = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float).reshape(-1)
    n = x.shape[0]
    x_design = np.column_stack([np.ones(n), x])
    names = ["Intercept", *feature_names]
    p = x_design.shape[1]
    dof = max(1, n - p)

    xtx_inv = np.linalg.pinv(x_design.T @ x_design)
    beta = xtx_inv @ x_design.T @ target
    fitted = x_design @ beta
    resid = target - fitted

    rss = float(np.dot(resid, resid))
    tss = float(np.dot(target - float(np.mean(target)), target - float(np.mean(target))))
    sigma2 = max(rss / dof, 1e-12)
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    t_stats = beta / se
    p_values = [float(2.0 * _student_t_sf(float(t), dof)) for t in t_stats]
    t_crit = _student_t_ppf(0.975, dof)
    ci_low = beta - (t_crit * se)
    ci_high = beta + (t_crit * se)

    rows: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        rows[name] = {
            "coef": float(beta[i]),
            "std_err": float(se[i]),
            "t_stat": float(t_stats[i]),
            "p_value": float(p_values[i]),
            "ci95_low": float(ci_low[i]),
            "ci95_high": float(ci_high[i]),
        }

    return {
        "rows": rows,
        "n_obs": int(n),
        "dof": int(dof),
        "r2": float(0.0 if tss <= 0 else 1.0 - (rss / tss)),
    }


def train_artifact() -> dict[str, Any]:
    raw_rows = load_csv_data()
    df = feature_engineer(raw_rows)

    feature_cols = [
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
        "is_month_start",
        "is_month_end",
        "week_sin",
        "week_cos",
        "sales_lag_1",
        "sales_lag_2",
        "sales_lag_4",
        "sales_lag_8",
        "sales_roll4_mean",
        "sales_roll4_std",
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
    pred_df.loc[:, "Predicted_Sales"] = pred
    pred_map = {
        (int(r.Store), pd.Timestamp(r.Date).strftime("%Y-%m-%d")): float(r.Predicted_Sales)
        for r in pred_df.itertuples(index=False)
    }

    ln_cols = [
        "CPI",
        "Unemployment",
        "Fuel_Price",
        "Temperature",
        "sales_lag_1",
        "sales_lag_4",
        "sales_roll4_mean",
        "sales_roll4_std",
    ]
    for c in ln_cols:
        df.loc[:, f"ln_{c}"] = np.log(np.clip(df[c].to_numpy(dtype=float), 1e-6, None))
    df.loc[:, "ln_Weekly_Sales"] = np.log(np.clip(df["Weekly_Sales"].to_numpy(dtype=float), 1.0, None))

    core_terms = [
        "Holiday_Flag",
        "week_sin",
        "week_cos",
        "ln_CPI",
        "ln_Unemployment",
        "ln_Fuel_Price",
        "ln_Temperature",
        "ln_sales_lag_1",
        "ln_sales_lag_4",
        "ln_sales_roll4_mean",
        "ln_sales_roll4_std",
    ]
    inference = _ols_inference(df[core_terms].to_numpy(dtype=float), df["ln_Weekly_Sales"].to_numpy(dtype=float), core_terms)
    coef_map = {term: float(inference["rows"][term]["coef"]) for term in core_terms}
    feature_term_map = {
        "CPI": "ln_CPI",
        "Unemployment": "ln_Unemployment",
        "Fuel_Price": "ln_Fuel_Price",
        "Temperature": "ln_Temperature",
    }
    feature_coef = {f: coef_map.get(term, 0.0) for f, term in feature_term_map.items()}
    feature_parametrics = {
        f: {
            "term": term,
            "coef": float(inference["rows"][term]["coef"]),
            "std_err": float(inference["rows"][term]["std_err"]),
            "t_stat": float(inference["rows"][term]["t_stat"]),
            "p_value": float(inference["rows"][term]["p_value"]),
            "ci95_low": float(inference["rows"][term]["ci95_low"]),
            "ci95_high": float(inference["rows"][term]["ci95_high"]),
        }
        for f, term in feature_term_map.items()
    }

    artifact = {
        "source": "extra_trees_notebook",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_cols,
        "pred_map": pred_map,
        "coef_table": coef_map,
        "feature_coefficients": feature_coef,
        "feature_parametrics": feature_parametrics,
        "coef_inference": inference,
        "coef_target_transform": "ln(Weekly_Sales)",
        "coef_feature_transform": "ln(feature) for CPI/Unemployment/Fuel_Price/Temperature",
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
