import joblib
import pandas as pd

from url_ml.feature_engineering import extract_url_features


class ModelService:
    def __init__(self, model_path):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.feature_stats = artifact["feature_stats"]
        self.model_type = artifact.get("training", {}).get("model_type", "rf")

    def predict(self, url: str) -> tuple[dict, int, float]:
        features = extract_url_features(url)
        feature_frame = pd.DataFrame([{name: features[name] for name in self.feature_names}])

        if self.model_type == "dl" and hasattr(self.model, "predict_with_urls"):
            predicted_class = int(self.model.predict_with_urls([url], feature_frame)[0])
            base_probability = float(self.model.predict_proba_with_urls([url], feature_frame)[0][1])
        else:
            predicted_class = int(self.model.predict(feature_frame)[0])
            if hasattr(self.model, "predict_proba"):
                base_probability = float(self.model.predict_proba(feature_frame)[0][1])
            else:
                base_probability = float(predicted_class)

        return features, predicted_class, base_probability

    def rank_features(self, features: dict, top_k: int = 8) -> list[dict]:
        importances = getattr(self.model, "feature_importances_", [0.0] * len(self.feature_names))
        ranked = []
        for feature_name, importance in zip(self.feature_names, importances):
            mean = self.feature_stats[feature_name]["mean"]
            std = self.feature_stats[feature_name]["std"] or 1.0
            value = float(features[feature_name])
            z_score = abs((value - mean) / std)
            score = float(importance) * z_score
            if not any(importances):
                score = z_score
            ranked.append(
                {
                    "feature": feature_name,
                    "value": value,
                    "importance": float(importance),
                    "debug_score": score,
                }
            )
        ranked.sort(key=lambda item: item["debug_score"], reverse=True)
        return ranked[:top_k]
