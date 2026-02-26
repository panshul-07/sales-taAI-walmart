import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyService:
    def detect(self, data: list[dict]) -> list[dict]:
        if len(data) < 10:
            return []
        df = pd.DataFrame(data)
        feats = df[["quantity", "revenue", "profit"]].fillna(0)
        clf = IsolationForest(random_state=42, contamination=0.05)
        flags = clf.fit_predict(feats)
        df["is_anomaly"] = flags == -1
        return df[df["is_anomaly"]].to_dict(orient="records")
