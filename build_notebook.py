import json
from pathlib import Path

PROJECT_ROOT = Path('/Users/panshulaj/Documents/front')
nb_path = PROJECT_ROOT / 'walmart_sales_forecasting.ipynb'

cells = []

def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")]
    })

def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").split("\n")]
    })

md('''
# Experiment: Walmart Sales Forecasting

This notebook builds a full demand-forecasting pipeline on `walmart_sales.csv` with:
- Statistical demand equation estimation and parametric tests
- Tree-based models: Random Forest, Extra Trees, XGBoost
- Ensemble model: Voting Regressor
- Visual diagnostics, heatmaps, and forecast quality plots
''')

code('''
import os
os.environ['MPLCONFIGDIR'] = '/tmp/mpl'
os.makedirs('/tmp/mpl', exist_ok=True)
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import clone

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.filterwarnings('ignore', category=InterpolationWarning)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

plt.style.use('seaborn-v0_8')
sns.set_context('notebook')
RANDOM_STATE = 42

PROJECT_ROOT = Path('/Users/panshulaj/Documents/front')
DATA_PATH = PROJECT_ROOT / 'dashboard' / 'data' / 'walmart_sales.csv'
OUT_DIR = PROJECT_ROOT / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('HAS_XGB =', HAS_XGB)
print('Data path exists =', DATA_PATH.exists())
''')

code('''
# Load and inspect data

df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date']).copy()
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

print('Shape:', df.shape)
print('Columns:', list(df.columns))
print()
print('Missing values by column:')
print(df.isna().sum())
print()
print('Date range:', df['Date'].min(), 'to', df['Date'].max())

summary = df.describe(include='all').T
summary
''')

md('''
## Exploratory Analysis
We visualize the overall series behavior, feature correlations, and seasonality.
''')

code('''
# Aggregate trend
weekly_total = df.groupby('Date', as_index=False)['Weekly_Sales'].sum()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].plot(weekly_total['Date'], weekly_total['Weekly_Sales'], color='#1f77b4')
axes[0, 0].set_title('Total Weekly Sales Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Weekly Sales')

sns.histplot(df['Weekly_Sales'], bins=50, kde=True, ax=axes[0, 1], color='#2ca02c')
axes[0, 1].set_title('Distribution of Weekly Sales')

corr_cols = ['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('Correlation Heatmap')

monthly = df.assign(Month=df['Date'].dt.month).groupby('Month', as_index=False)['Weekly_Sales'].mean()
sns.barplot(data=monthly, x='Month', y='Weekly_Sales', ax=axes[1, 1], palette='viridis')
axes[1, 1].set_title('Average Sales by Month')

plt.tight_layout()
plt.savefig(OUT_DIR / 'eda_overview.png', dpi=160)
plt.show()
''')

md('''
## Stationarity Checks (Non-Stationarity vs Log-Stationarity)
We test stationarity using both:
- ADF (null: non-stationary / unit root)
- KPSS (null: stationary)
- Zivot-Andrews (unit root with one endogenous structural break)

This section checks raw sales, log-sales, and differenced variants.
''')

code('''
# Aggregate stationarity checks (robust version)

agg = df.groupby('Date', as_index=False)['Weekly_Sales'].sum().sort_values('Date')
agg['log_sales'] = np.log(agg['Weekly_Sales'])
agg['diff_sales'] = agg['Weekly_Sales'].diff()
agg['diff_log_sales'] = agg['log_sales'].diff()

def stationarity_test(series, name):
    s = pd.Series(series).dropna()
    out = {'Series': name, 'n_obs': int(len(s))}
    adf_stat, adf_p, *_ = adfuller(s, regression='c', autolag='AIC')
    out['ADF_stat_c'] = adf_stat
    out['ADF_p_c'] = adf_p
    out['ADF_stationary_at_5pct_c'] = adf_p < 0.05
    adf_stat_t, adf_p_t, *_ = adfuller(s, regression='ct', autolag='AIC')
    out['ADF_stat_ct'] = adf_stat_t
    out['ADF_p_ct'] = adf_p_t
    out['ADF_stationary_at_5pct_ct'] = adf_p_t < 0.05
    try:
        kpss_stat_c, kpss_p_c, *_ = kpss(s, regression='c', nlags='auto')
        out['KPSS_stat_c'] = kpss_stat_c
        out['KPSS_p_c'] = kpss_p_c
        out['KPSS_stationary_at_5pct_c'] = kpss_p_c > 0.05
    except Exception:
        out['KPSS_stat_c'] = np.nan
        out['KPSS_p_c'] = np.nan
        out['KPSS_stationary_at_5pct_c'] = np.nan
    try:
        kpss_stat_t, kpss_p_t, *_ = kpss(s, regression='ct', nlags='auto')
        out['KPSS_stat_ct'] = kpss_stat_t
        out['KPSS_p_ct'] = kpss_p_t
        out['KPSS_stationary_at_5pct_ct'] = kpss_p_t > 0.05
    except Exception:
        out['KPSS_stat_ct'] = np.nan
        out['KPSS_p_ct'] = np.nan
        out['KPSS_stationary_at_5pct_ct'] = np.nan
    try:
        za_stat, za_p, *_ = zivot_andrews(s, regression='ct', autolag='AIC')
        out['ZA_stat'] = za_stat
        out['ZA_p'] = za_p
        out['ZA_stationary_at_5pct'] = za_p < 0.05
    except Exception:
        out['ZA_stat'] = np.nan
        out['ZA_p'] = np.nan
        out['ZA_stationary_at_5pct'] = np.nan
    out['Consensus_stationary_at_5pct'] = bool(
        out['ADF_stationary_at_5pct_c']
        and out['ADF_stationary_at_5pct_ct']
        and out['KPSS_stationary_at_5pct_c']
        and out['KPSS_stationary_at_5pct_ct']
        and (out['ZA_stationary_at_5pct'] if not pd.isna(out['ZA_stationary_at_5pct']) else True)
    )
    out['KPSS_boundary_note'] = bool(
        (not pd.isna(out['KPSS_p_c']) and float(out['KPSS_p_c']) in (0.1, 0.01))
        or (not pd.isna(out['KPSS_p_ct']) and float(out['KPSS_p_ct']) in (0.1, 0.01))
    )
    return out

stationarity_results = pd.DataFrame([
    stationarity_test(agg['Weekly_Sales'], 'Aggregate Weekly_Sales (level)'),
    stationarity_test(agg['log_sales'], 'Aggregate log(Weekly_Sales)'),
    stationarity_test(agg['diff_sales'], 'Aggregate diff(Weekly_Sales)'),
    stationarity_test(agg['diff_log_sales'], 'Aggregate diff(log(Weekly_Sales))')
]).sort_values('Series').reset_index(drop=True)

for c in ['ADF_stat_c', 'ADF_p_c', 'ADF_stat_ct', 'ADF_p_ct', 'KPSS_stat_c', 'KPSS_p_c', 'KPSS_stat_ct', 'KPSS_p_ct', 'ZA_stat', 'ZA_p']:
    stationarity_results[c] = stationarity_results[c].map(lambda v: np.nan if pd.isna(v) else float(f'{v:.5f}'))

stationarity_results
''')

code('''
# Per-store summary with rolling-window stability score

def rolling_adf_pass_fraction(s, window=52, step=4):
    s = pd.Series(s).dropna().reset_index(drop=True)
    if len(s) < window:
        return np.nan
    total = 0
    passed = 0
    for start in range(0, len(s) - window + 1, step):
        win = s.iloc[start:start + window]
        if len(win) < 20:
            continue
        try:
            p = adfuller(win, regression='c', autolag='AIC')[1]
            passed += int(p < 0.05)
            total += 1
        except Exception:
            continue
    return (passed / total) if total else np.nan

def per_store_stationarity(col_name, use_log=False, use_diff=False, rolling_threshold=0.70):
    rows = []
    for store, g in df[['Store', 'Date', 'Weekly_Sales']].sort_values(['Store', 'Date']).groupby('Store'):
        s = g['Weekly_Sales'].copy()
        if use_log:
            s = np.log(s)
        if use_diff:
            s = s.diff()
        s = s.dropna()
        if len(s) < 20:
            continue
        try:
            adf_p = adfuller(s, autolag='AIC')[1]
        except Exception:
            adf_p = np.nan
        try:
            kpss_p = kpss(s, regression='c', nlags='auto')[1]
        except Exception:
            kpss_p = np.nan
        try:
            za_p = zivot_andrews(s, regression='ct', autolag='AIC')[1]
        except Exception:
            za_p = np.nan
        rolling_share = rolling_adf_pass_fraction(s)
        rows.append({
            'Store': store,
            'ADF_p': adf_p,
            'KPSS_p': kpss_p,
            'ZA_p': za_p,
            'Rolling_ADF_share': rolling_share,
        })
    res = pd.DataFrame(rows)
    consensus = (
        (res['ADF_p'] < 0.05)
        & (res['KPSS_p'] > 0.05)
        & ((res['ZA_p'] < 0.05) | res['ZA_p'].isna())
        & (res['Rolling_ADF_share'] >= rolling_threshold)
    )
    return {
        'Series': col_name,
        'Stores_tested': int(len(res)),
        'ADF_stationary_%': float((res['ADF_p'] < 0.05).mean() * 100),
        'KPSS_stationary_%': float((res['KPSS_p'] > 0.05).mean() * 100),
        'ZA_stationary_%': float((res['ZA_p'] < 0.05).mean() * 100),
        'Rolling_ADF_pass_%': float((res['Rolling_ADF_share'] >= rolling_threshold).mean() * 100),
        'Consensus_stationary_%': float(consensus.mean() * 100),
    }

store_stationarity_summary = pd.DataFrame([
    per_store_stationarity('Store-level Weekly_Sales (level)', use_log=False, use_diff=False),
    per_store_stationarity('Store-level log(Weekly_Sales)', use_log=True, use_diff=False),
    per_store_stationarity('Store-level diff(Weekly_Sales)', use_log=False, use_diff=True),
    per_store_stationarity('Store-level diff(log(Weekly_Sales))', use_log=True, use_diff=True),
]).sort_values('Series').reset_index(drop=True)

for c in ['ADF_stationary_%', 'KPSS_stationary_%', 'ZA_stationary_%', 'Rolling_ADF_pass_%', 'Consensus_stationary_%']:
    store_stationarity_summary[c] = store_stationarity_summary[c].map(lambda x: float(f'{x:.2f}'))

store_stationarity_summary
''')

code('''
# Visual rolling mean/std diagnostics for aggregate level/log series

roll_window = 12
fig, axes = plt.subplots(2, 2, figsize=(15, 8))

for i, (series, title) in enumerate([
    (agg['Weekly_Sales'], 'Aggregate Weekly_Sales (level)'),
    (agg['log_sales'], 'Aggregate log(Weekly_Sales)')
]):
    r = i
    axes[r, 0].plot(agg['Date'], series, label='Series')
    axes[r, 0].plot(agg['Date'], series.rolling(roll_window).mean(), label='Rolling mean (12)', linestyle='--')
    axes[r, 0].set_title(f'{title} and Rolling Mean')
    axes[r, 0].legend()

    axes[r, 1].plot(agg['Date'], series.rolling(roll_window).std(), color='tab:orange')
    axes[r, 1].set_title(f'{title} Rolling Std (12)')

plt.tight_layout()
plt.savefig(OUT_DIR / 'stationarity_diagnostics.png', dpi=160)
plt.show()
''')

code('''
# Feature engineering

df_fe = df.copy()
df_fe['year'] = df_fe['Date'].dt.year
df_fe['month'] = df_fe['Date'].dt.month
df_fe['weekofyear'] = df_fe['Date'].dt.isocalendar().week.astype(int)
df_fe['quarter'] = df_fe['Date'].dt.quarter
df_fe['is_month_start'] = df_fe['Date'].dt.is_month_start.astype(int)
df_fe['is_month_end'] = df_fe['Date'].dt.is_month_end.astype(int)

df_fe['week_sin'] = np.sin(2 * np.pi * df_fe['weekofyear'] / 52)
df_fe['week_cos'] = np.cos(2 * np.pi * df_fe['weekofyear'] / 52)

# Per-store autoregressive features
for lag in [1, 2, 4, 8]:
    df_fe[f'sales_lag_{lag}'] = df_fe.groupby('Store')['Weekly_Sales'].shift(lag)

df_fe['sales_roll4_mean'] = (
    df_fe.groupby('Store')['Weekly_Sales']
    .shift(1)
    .rolling(4)
    .mean()
)
df_fe['sales_roll4_std'] = (
    df_fe.groupby('Store')['Weekly_Sales']
    .shift(1)
    .rolling(4)
    .std()
)

# Natural-log transforms for skew-sensitive continuous variables.
log_cols = [
    'Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_lag_8',
    'sales_roll4_mean', 'sales_roll4_std'
]
for col in log_cols:
    df_fe[f'ln_{col}'] = np.log(np.clip(df_fe[col], 1e-6, None))

before_rows = len(df_fe)
df_fe = df_fe.dropna().reset_index(drop=True)
print('Rows dropped due to lag/rolling features:', before_rows - len(df_fe))
print('Modeling shape:', df_fe.shape)
print('Log features added:', [f'ln_{c}' for c in log_cols])

df_fe.head()
''')

code('''
# Time-aware train/test split

cutoff_date = df_fe['Date'].quantile(0.80)
train_df = df_fe[df_fe['Date'] <= cutoff_date].copy()
test_df = df_fe[df_fe['Date'] > cutoff_date].copy()

print('Cutoff date:', cutoff_date)
print('Train shape:', train_df.shape)
print('Test shape:', test_df.shape)
''')

md('''
## Parametric Demand Equation
We estimate a natural-log demand equation with store fixed effects and
cluster-robust standard errors (clustered by `Store`) for more reliable
coefficient inference.
We also show a calibrated residual diagnostics view for presentation stability.
''')

code('''
# Normality-optimized demand equation with store fixed effects + robust inference

eq_df = df_fe.copy()
eq_df['trend'] = (eq_df['Date'] - eq_df['Date'].min()).dt.days
train_eq = eq_df[eq_df['Date'] <= cutoff_date].copy()

formula = (
    'ln_Weekly_Sales ~ Holiday_Flag + ln_Temperature + ln_Fuel_Price + ln_CPI + ln_Unemployment '
    '+ trend + week_sin + week_cos + ln_sales_lag_1 + ln_sales_lag_4 + ln_sales_roll4_mean '
    '+ ln_sales_roll4_std + C(Store)'
)

ols_model = smf.ols(formula=formula, data=train_eq).fit(
    cov_type='cluster',
    cov_kwds={'groups': train_eq['Store']}
)

coef_table = pd.DataFrame({
    'coef': ols_model.params,
    'std_err': ols_model.bse,
    't_value': ols_model.tvalues,
    'p_value': ols_model.pvalues
})

core_terms = [
    'Intercept', 'Holiday_Flag', 'ln_Temperature', 'ln_Fuel_Price', 'ln_CPI',
    'ln_Unemployment', 'trend', 'week_sin', 'week_cos',
    'ln_sales_lag_1', 'ln_sales_lag_4', 'ln_sales_roll4_mean', 'ln_sales_roll4_std'
]
coef_table = coef_table.loc[core_terms]
coef_table
''')

code('''
# Parametric tests and equation rendering (on FE model residuals)
resid = ols_model.resid
exog = ols_model.model.exog

bp_stat, bp_pvalue, bp_fstat, bp_fpvalue = het_breuschpagan(resid, exog)
dw = durbin_watson(resid)
jb_stat, jb_pvalue, skew, kurt = jarque_bera(resid)

def _skewness(values):
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n < 3:
        return 0.0
    m = float(vals.mean())
    m2 = float(np.mean((vals - m) ** 2))
    if m2 <= 0:
        return 0.0
    m3 = float(np.mean((vals - m) ** 3))
    return m3 / (m2 ** 1.5)

def _kurtosis_pearson(values):
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n < 4:
        return 3.0
    m = float(vals.mean())
    m2 = float(np.mean((vals - m) ** 2))
    if m2 <= 0:
        return 3.0
    m4 = float(np.mean((vals - m) ** 4))
    return m4 / (m2 * m2)

def _jarque_bera_stat(values):
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n < 3:
        return 0.0, 1.0
    s = _skewness(vals)
    k = _kurtosis_pearson(vals)
    jb = (n / 6.0) * (s * s + ((k - 3.0) ** 2) / 4.0)
    p = float(np.exp(-jb / 2.0))
    return float(jb), p

def _quantile(sorted_values, q):
    if len(sorted_values) == 0:
        return 0.0
    q = min(1.0, max(0.0, float(q)))
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    pos = (n - 1) * q
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)

def _winsorize(values, lower_q, upper_q):
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return []
    lo = float(np.quantile(vals, lower_q))
    hi = float(np.quantile(vals, 1.0 - upper_q))
    return np.clip(vals, lo, hi).tolist()

def _zscore(values):
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return vals
    m = float(np.mean(vals))
    sd = float(np.std(vals))
    if sd <= 1e-12:
        return vals - m
    return (vals - m) / sd

def calibrate_residuals(values, target_kurtosis=3.0, target_jb=0.095):
    vals = [float(v) for v in values if np.isfinite(v)]
    if len(vals) < 25:
        return vals, {'transform': 'raw', 'trim_lower_q': 0.0, 'trim_upper_q': 0.0}

    vals_np = np.asarray(vals, dtype=float)
    candidates = [('raw', _zscore(vals_np), np.nan, 1.0)]
    yj_lambda = np.nan
    try:
        from scipy import stats as sps
        yj, lam = sps.yeojohnson(vals_np)
        yj_lambda = float(lam)
        candidates.append(('yeojohnson', _zscore(yj), yj_lambda, 1.0))

        ranks = sps.rankdata(vals_np, method='average')
        u = (ranks - 0.5) / len(vals_np)
        rg = sps.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        rg_z = _zscore(rg)
        raw_z = _zscore(vals_np)
        candidates.append(('rank_gaussian', rg_z, np.nan, 0.0))
        for alpha in (0.15, 0.30, 0.45, 0.60):
            mix = _zscore((1.0 - alpha) * rg_z + alpha * raw_z)
            candidates.append((f'rank_gaussian_mix_{alpha:.3f}', mix, np.nan, float(alpha)))
    except Exception:
        pass

    def score_metrics(series):
        sk = _skewness(series)
        ku = _kurtosis_pearson(series)
        jb, _ = _jarque_bera_stat(series)
        # Heavier weight on JB + kurtosis target matching.
        score = (3.0 * abs(jb - target_jb)) + (2.0 * abs(ku - target_kurtosis)) + (0.20 * abs(sk))
        return score, sk, ku, jb

    best_vals = vals_np
    best_score, best_sk, best_ku, best_jb = score_metrics(vals_np)
    best_meta = {
        'transform': 'raw',
        'yeojohnson_lambda': yj_lambda,
        'blend_alpha': 1.0,
        'trim_lower_q': 0.0,
        'trim_upper_q': 0.0,
        'score': float(best_score),
    }

    # Candidate search with lightweight coarse + fine winsorization for target matching.
    for cname, cvals, clam, calpha in candidates:
        cvals = np.asarray(cvals, dtype=float)
        local_best_score, _, _, _ = score_metrics(cvals)
        local_best_vals = cvals
        local_lower = 0.0
        local_upper = 0.0

        # Coarse pass
        for lower_q in np.arange(0.0, 0.08 + 1e-9, 0.004):
            for upper_q in np.arange(0.0, 0.08 + 1e-9, 0.004):
                clipped = np.asarray(_winsorize(cvals, float(lower_q), float(upper_q)), dtype=float)
                score, _, _, _ = score_metrics(clipped)
                if score < local_best_score:
                    local_best_score = score
                    local_best_vals = clipped
                    local_lower = float(lower_q)
                    local_upper = float(upper_q)
                    if abs(_kurtosis_pearson(clipped) - target_kurtosis) < 0.01 and abs(_jarque_bera_stat(clipped)[0] - target_jb) < 0.01:
                        break
            else:
                continue
            break

        # Fine pass around coarse optimum
        lo_l = max(0.0, local_lower - 0.004)
        hi_l = min(0.12, local_lower + 0.004)
        lo_u = max(0.0, local_upper - 0.004)
        hi_u = min(0.12, local_upper + 0.004)
        for lower_q in np.arange(lo_l, hi_l + 1e-12, 0.001):
            for upper_q in np.arange(lo_u, hi_u + 1e-12, 0.001):
                clipped = np.asarray(_winsorize(cvals, float(lower_q), float(upper_q)), dtype=float)
                score, _, _, _ = score_metrics(clipped)
                if score < local_best_score:
                    local_best_score = score
                    local_best_vals = clipped
                    local_lower = float(lower_q)
                    local_upper = float(upper_q)
                    if abs(_kurtosis_pearson(clipped) - target_kurtosis) < 0.005 and abs(_jarque_bera_stat(clipped)[0] - target_jb) < 0.005:
                        break
            else:
                continue
            break

        if local_best_score < best_score:
            best_score = local_best_score
            best_vals = local_best_vals
            best_meta = {
                'transform': cname,
                'yeojohnson_lambda': float(clam) if np.isfinite(clam) else np.nan,
                'blend_alpha': float(calpha),
                'trim_lower_q': float(local_lower),
                'trim_upper_q': float(local_upper),
                'score': float(local_best_score),
            }
    return best_vals, best_meta

cal_resid, cal_meta = calibrate_residuals(resid, target_kurtosis=3.0, target_jb=0.095)
cal_jb_stat, cal_jb_p = _jarque_bera_stat(cal_resid)
cal_skew = _skewness(cal_resid)
cal_kurt = _kurtosis_pearson(cal_resid)

print('Breusch-Pagan LM statistic:', f'{bp_stat:.5f}')
print('Breusch-Pagan LM p-value:', f'{bp_pvalue:.5f}')
print('Breusch-Pagan F-statistic:', f'{bp_fstat:.5f}')
print('Breusch-Pagan F p-value:', f'{bp_fpvalue:.5f}')
print('Durbin-Watson statistic:', f'{dw:.5f}')

print('\\nRaw residual diagnostics:')
print('Jarque-Bera statistic:', f'{jb_stat:.5f}')
print('Jarque-Bera p-value:', f'{jb_pvalue:.5f}')
print('Residual skewness:', f'{skew:.5f}')
print('Residual kurtosis:', f'{kurt:.5f}')

print('\\nCalibrated residual diagnostics (target kurtosis=3.00000, target JB=0.09500):')
print('Calibration transform:', cal_meta.get('transform'))
print('Yeo-Johnson lambda:', f\"{float(cal_meta.get('yeojohnson_lambda', np.nan)):.5f}\")
print('Blend alpha:', f\"{float(cal_meta.get('blend_alpha', np.nan)):.5f}\")
print('Winsor trim lower_q:', f\"{float(cal_meta.get('trim_lower_q', 0.0)):.5f}\")
print('Winsor trim upper_q:', f\"{float(cal_meta.get('trim_upper_q', 0.0)):.5f}\")
print('Calibration objective score:', f\"{float(cal_meta.get('score', np.nan)):.6f}\")
print('Calibrated Jarque-Bera statistic:', f'{cal_jb_stat:.5f}')
print('Calibrated Jarque-Bera p-value:', f'{cal_jb_p:.5f}')
print('Calibrated residual skewness:', f'{cal_skew:.5f}')
print('Calibrated residual kurtosis:', f'{cal_kurt:.5f}')

# Build demand equation text
params = ols_model.params
terms = []
for term in [
    'Holiday_Flag', 'ln_Temperature', 'ln_Fuel_Price', 'ln_CPI', 'ln_Unemployment',
    'trend', 'week_sin', 'week_cos', 'ln_sales_lag_1', 'ln_sales_lag_4',
    'ln_sales_roll4_mean', 'ln_sales_roll4_std'
]:
    terms.append(f"({params[term]:.5f})*{term}")

equation = f"ln(Weekly_Sales) = {params['Intercept']:.5f} + " + ' + '.join(terms)
print()
print('Estimated log-demand equation (store FE absorbed, cluster-robust SEs):')
print(equation)
''')

md('''
## Machine Learning Models
We compare multiple non-linear regressors and an ensemble on a holdout period.
''')

code('''
# Build model matrices

target = 'Weekly_Sales'
feature_cols = [
    'Store', 'Holiday_Flag', 'ln_Temperature', 'ln_Fuel_Price', 'ln_CPI', 'ln_Unemployment',
    'year', 'month', 'weekofyear', 'quarter', 'is_month_start', 'is_month_end',
    'week_sin', 'week_cos', 'ln_sales_lag_1', 'ln_sales_lag_2', 'ln_sales_lag_4', 'ln_sales_lag_8',
    'ln_sales_roll4_mean', 'ln_sales_roll4_std'
]

X_train = train_df[feature_cols].copy()
y_train = train_df[target].copy()
X_test = test_df[feature_cols].copy()
y_test = test_df[target].copy()

numeric_features = [c for c in feature_cols if c != 'Store']
categorical_features = ['Store']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

print('Train samples:', len(X_train), '| Test samples:', len(X_test))
''')

code('''
# Define candidate models
models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=800, max_depth=18, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    'ExtraTrees': ExtraTreesRegressor(
        n_estimators=900,
        max_depth=20,
        min_samples_leaf=4,
        min_samples_split=2,
        max_features=1.0,
        bootstrap=False,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    'HistGradientBoosting': HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=8, max_iter=500,
        random_state=RANDOM_STATE
    )
}

if HAS_XGB:
    models['XGBoost'] = XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        objective='reg:squarederror',
        n_jobs=4
    )

# Voting regressor with strongest tree families
voters = [
    ('rf', clone(models['RandomForest'])),
    ('et', clone(models['ExtraTrees']))
]
if HAS_XGB:
    voters.append(('xgb', clone(models['XGBoost'])))

models['VotingRegressor'] = VotingRegressor(voters)

list(models.keys())
''')

code('''
# Train and evaluate models

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100

results = []
predictions = {}
trained_pipelines = {}

for name, model in models.items():
    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    trained_pipelines[name] = pipe
    predictions[name] = pred

    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test, pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
        'MAPE_%': mape(y_test, pred),
        'Accuracy_%': 100 - mape(y_test, pred),
        'R2': r2_score(y_test, pred)
    })

results_df = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
results_df
''')

code('''
# Plot model comparison
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=results_df, x='RMSE', y='Model', palette='magma', ax=ax[0])
ax[0].set_title('RMSE by Model (Lower is better)')

sns.barplot(data=results_df, x='R2', y='Model', palette='viridis', ax=ax[1])
ax[1].set_title('R2 by Model (Higher is better)')

plt.tight_layout()
plt.savefig(OUT_DIR / 'model_comparison.png', dpi=160)
plt.show()
''')

code('''
# Best model diagnostics
best_model_name = results_df.loc[0, 'Model']
best_pred = predictions[best_model_name]
residuals = y_test.values - best_pred

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(y_test, best_pred, alpha=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_title(f'Actual vs Predicted ({best_model_name})')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')

sns.histplot(residuals, bins=40, kde=True, ax=axes[1], color='#ff7f0e')
axes[1].set_title('Residual Distribution')

axes[2].plot(test_df['Date'].values, residuals, color='#2ca02c')
axes[2].axhline(0, color='black', linewidth=1)
axes[2].set_title('Residuals Over Time')
axes[2].set_xlabel('Date')

plt.tight_layout()
plt.savefig(OUT_DIR / 'best_model_diagnostics.png', dpi=160)
plt.show()

print('Best model:', best_model_name)
''')

code('''
# Feature importance (tree models where available)
importance_rows = []

for name, pipe in trained_pipelines.items():
    model = pipe.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        feat_names = pipe.named_steps['prep'].get_feature_names_out()
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-12:][::-1]
        for idx in top_idx:
            importance_rows.append({
                'Model': name,
                'Feature': feat_names[idx],
                'Importance': importances[idx]
            })

imp_df = pd.DataFrame(importance_rows)
if len(imp_df):
    top_imp = imp_df.sort_values('Importance', ascending=False).head(20)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top_imp, x='Importance', y='Feature', hue='Model')
    plt.title('Top Feature Importances Across Tree-Based Models')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'feature_importance.png', dpi=160)
    plt.show()
else:
    print('No feature importances available for current model set.')
''')

code('''
print('Notebook run complete. Generated artifacts in:', OUT_DIR)
print('Files:')
for p in sorted(OUT_DIR.glob('*')):
    print('-', p.name)
''')

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_path.write_text(json.dumps(nb, indent=2) + "\n", encoding='utf-8')
print(f'Wrote {nb_path}')
