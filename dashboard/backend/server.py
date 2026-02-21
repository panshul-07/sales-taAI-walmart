import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path('/Users/panshulaj/Documents/sale forecasting/dashboard')
STATIC_DIR = BASE_DIR / 'static'
DATA_PATH = Path('/Users/panshulaj/Documents/sales-forecasting-walmart/data/walmart_sales.csv')

FEATURE_COLS = [
    'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'year', 'month', 'weekofyear', 'quarter', 'is_month_start', 'is_month_end',
    'week_sin', 'week_cos', 'sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_lag_8',
    'sales_roll4_mean', 'sales_roll4_std'
]

NUMERIC_FEATURES = [c for c in FEATURE_COLS if c != 'Store']
CATEGORICAL_FEATURES = ['Store']


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _safe_float(v: float, ndigits: int = 4) -> float:
    return float(round(float(v), ndigits))


def prepare_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Pipeline, dict]:
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    df_fe = df.copy()
    df_fe['year'] = df_fe['Date'].dt.year
    df_fe['month'] = df_fe['Date'].dt.month
    df_fe['weekofyear'] = df_fe['Date'].dt.isocalendar().week.astype(int)
    df_fe['quarter'] = df_fe['Date'].dt.quarter
    df_fe['is_month_start'] = df_fe['Date'].dt.is_month_start.astype(int)
    df_fe['is_month_end'] = df_fe['Date'].dt.is_month_end.astype(int)
    df_fe['week_sin'] = np.sin(2 * np.pi * df_fe['weekofyear'] / 52)
    df_fe['week_cos'] = np.cos(2 * np.pi * df_fe['weekofyear'] / 52)

    for lag in [1, 2, 4, 8]:
        df_fe[f'sales_lag_{lag}'] = df_fe.groupby('Store')['Weekly_Sales'].shift(lag)

    df_fe['sales_roll4_mean'] = (
        df_fe.groupby('Store')['Weekly_Sales'].shift(1).rolling(4).mean()
    )
    df_fe['sales_roll4_std'] = (
        df_fe.groupby('Store')['Weekly_Sales'].shift(1).rolling(4).std()
    )

    df_fe = df_fe.dropna().reset_index(drop=True)

    cutoff_date = df_fe['Date'].quantile(0.80)
    train_df = df_fe[df_fe['Date'] <= cutoff_date].copy()
    test_df = df_fe[df_fe['Date'] > cutoff_date].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ]
    )

    model = ExtraTreesRegressor(
        n_estimators=900,
        max_depth=20,
        min_samples_leaf=4,
        min_samples_split=2,
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', model),
    ])

    X_train = train_df[FEATURE_COLS]
    y_train = train_df['Weekly_Sales']
    X_test = test_df[FEATURE_COLS]
    y_test = test_df['Weekly_Sales']

    pipeline.fit(X_train, y_train)
    y_pred_test = pipeline.predict(X_test)

    metrics = {
        'mae': _safe_float(mean_absolute_error(y_test, y_pred_test), 4),
        'rmse': _safe_float(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4),
        'mape_pct': _safe_float(_mape(y_test.values, y_pred_test), 4),
        'accuracy_pct': _safe_float(100.0 - _mape(y_test.values, y_pred_test), 4),
        'r2': _safe_float(r2_score(y_test, y_pred_test), 5),
        'train_rows': int(len(train_df)),
        'test_rows': int(len(test_df)),
    }

    df_fe = df_fe.copy()
    df_fe['predicted_sales'] = pipeline.predict(df_fe[FEATURE_COLS])
    return df, df_fe, test_df, pipeline, metrics


RAW_DF, MODEL_DF, TEST_DF, MODEL_PIPELINE, METRICS = prepare_dataset()


def _json_response(handler: BaseHTTPRequestHandler, payload: dict | list, status: int = 200) -> None:
    data = json.dumps(payload).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_file_bytes(path: Path) -> bytes | None:
    if not path.exists() or not path.is_file():
        return None
    return path.read_bytes()


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path.startswith('/api/'):
            self._handle_api(path, query)
            return

        self._handle_static(path)

    def _handle_api(self, path: str, query: dict[str, list[str]]):
        if path == '/api/health':
            _json_response(self, {'status': 'ok', 'timestamp': datetime.utcnow().isoformat() + 'Z'})
            return

        if path == '/api/stores':
            stores = (
                MODEL_DF.groupby('Store', as_index=False)
                .agg(
                    records=('Weekly_Sales', 'count'),
                    avg_sales=('Weekly_Sales', 'mean'),
                    total_sales=('Weekly_Sales', 'sum'),
                )
            )
            out = stores.sort_values('Store').to_dict(orient='records')
            for row in out:
                row['Store'] = int(row['Store'])
                row['avg_sales'] = _safe_float(row['avg_sales'], 2)
                row['total_sales'] = _safe_float(row['total_sales'], 2)
            _json_response(self, out)
            return

        store = query.get('store', ['all'])[0]
        weeks = int(query.get('weeks', ['160'])[0])
        feature = query.get('feature', ['CPI'])[0]

        df = MODEL_DF
        if store != 'all':
            try:
                sid = int(store)
                df = MODEL_DF[MODEL_DF['Store'] == sid]
            except ValueError:
                _json_response(self, {'error': 'Invalid store id'}, 400)
                return

        if path == '/api/overview':
            latest = df.sort_values('Date').tail(weeks)
            payload = {
                'store': store,
                'records': int(len(latest)),
                'date_min': str(latest['Date'].min().date()) if len(latest) else None,
                'date_max': str(latest['Date'].max().date()) if len(latest) else None,
                'total_sales': _safe_float(latest['Weekly_Sales'].sum(), 2),
                'avg_weekly_sales': _safe_float(latest['Weekly_Sales'].mean(), 2),
                'avg_temp': _safe_float(latest['Temperature'].mean(), 2),
                'avg_cpi': _safe_float(latest['CPI'].mean(), 2),
                'avg_unemployment': _safe_float(latest['Unemployment'].mean(), 2),
                'model': METRICS,
            }
            _json_response(self, payload)
            return

        if path == '/api/timeseries':
            s = df.sort_values('Date').tail(weeks)
            payload = {
                'date': [d.strftime('%Y-%m-%d') for d in s['Date']],
                'actual': [float(v) for v in s['Weekly_Sales']],
                'predicted': [float(v) for v in s['predicted_sales']],
            }
            _json_response(self, payload)
            return

        if path == '/api/feature-dependencies':
            feats = ['CPI', 'Unemployment', 'Fuel_Price', 'Temperature']
            s = df.sort_values('Date').tail(weeks)
            payload = {
                f: {
                    'x': [float(v) for v in s[f]],
                    'y': [float(v) for v in s['Weekly_Sales']],
                }
                for f in feats
            }
            _json_response(self, payload)
            return

        if path == '/api/feature-series':
            if feature not in {'CPI', 'Unemployment', 'Fuel_Price', 'Temperature'}:
                _json_response(self, {'error': 'Unsupported feature'}, 400)
                return
            s = df.sort_values('Date').tail(weeks)
            payload = {
                'feature': feature,
                'date': [d.strftime('%Y-%m-%d') for d in s['Date']],
                'feature_values': [float(v) for v in s[feature]],
                'sales': [float(v) for v in s['Weekly_Sales']],
            }
            _json_response(self, payload)
            return

        if path == '/api/correlation':
            s = df.sort_values('Date').tail(weeks)
            cols = ['Weekly_Sales', 'CPI', 'Unemployment', 'Fuel_Price', 'Temperature']
            corr = s[cols].corr()
            _json_response(self, {
                'labels': cols,
                'z': [[_safe_float(v, 4) for v in row] for row in corr.values.tolist()],
            })
            return

        _json_response(self, {'error': 'Not found'}, 404)

    def _handle_static(self, path: str):
        if path in ('/', '/index.html'):
            file_path = STATIC_DIR / 'index.html'
            content_type = 'text/html; charset=utf-8'
        else:
            rel = path.lstrip('/')
            file_path = STATIC_DIR / rel
            ext = file_path.suffix.lower()
            content_type = {
                '.css': 'text/css; charset=utf-8',
                '.js': 'application/javascript; charset=utf-8',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.svg': 'image/svg+xml',
            }.get(ext, 'application/octet-stream')

        payload = _read_file_bytes(file_path)
        if payload is None:
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run():
    host = '0.0.0.0'
    port = int(os.getenv('PORT', '8080'))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f'Dashboard server running on http://{host}:{port}')
    server.serve_forever()


if __name__ == '__main__':
    run()
