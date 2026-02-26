import pandas as pd
from prophet import Prophet


class ForecastService:
    def forecast(self, data: list[dict], periods: int = 12) -> list[dict]:
        if not data:
            return []
        df = pd.DataFrame(data)
        df = df.rename(columns={"date": "ds", "revenue": "y"})
        df["ds"] = pd.to_datetime(df["ds"])
        model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df[["ds", "y"]])
        future = model.make_future_dataframe(periods=periods, freq="W")
        fc = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
        return fc.to_dict(orient="records")
