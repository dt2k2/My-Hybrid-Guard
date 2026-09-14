import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from url_ml.dl_model import DLModelConfig, DeepURLClassifier
from url_ml.feature_engineering import NUMERIC_FEATURES, extract_url_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a phishing URL classifier from a CSV dataset.",
    )
    parser.add_argument("--data", default="Dataset.csv", help="Path to the CSV dataset.")
    parser.add_argument(
        "--model",
        default="rf",
        choices=["rf", "dt", "dl"],
        help="Model type: rf = Random Forest, dt = Decision Tree, dl = deep neural URL encoder.",
    )
    parser.add_argument("--label-col", default="label", help="Target label column.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the trained model and metadata are saved.",
    )
    parser.add_argument("--n-estimators", type=int, default=300, help="RF tree count.")
    parser.add_argument("--max-depth", type=int, default=None, help="Optional max tree depth.")
    parser.add_argument("--dl-max-features", type=int, default=40000, help="Maximum character n-gram features for DL model.")
    parser.add_argument("--dl-ngram-min", type=int, default=3, help="Minimum character n-gram size for DL model.")
    parser.add_argument("--dl-ngram-max", type=int, default=5, help="Maximum character n-gram size for DL model.")
    parser.add_argument("--dl-hidden-layers", default="256,128,64", help="Comma-separated hidden layer sizes for DL model.")
    parser.add_argument("--dl-max-iter", type=int, default=80, help="Max training epochs/iterations for DL model.")
    return parser.parse_args()


def ensure_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing_features = [column for column in NUMERIC_FEATURES if column not in dataframe.columns]
    if not missing_features:
        return dataframe

    if "url" not in dataframe.columns:
        raise ValueError(
            "Dataset is missing required numeric feature columns and also has no 'url' column to regenerate them.",
        )

    generated_features = dataframe["url"].apply(extract_url_features).apply(pd.Series)
    for column in NUMERIC_FEATURES:
        dataframe[column] = generated_features[column]
    return dataframe


def build_model(args: argparse.Namespace):
    if args.model == "dl":
        hidden_layers = tuple(int(value.strip()) for value in args.dl_hidden_layers.split(",") if value.strip())
        if not hidden_layers:
            raise ValueError("--dl-hidden-layers must contain at least one positive integer.")

        config = DLModelConfig(
            max_features=args.dl_max_features,
            ngram_min=args.dl_ngram_min,
            ngram_max=args.dl_ngram_max,
            hidden_layers=hidden_layers,
            max_iter=args.dl_max_iter,
            random_state=args.random_state,
        )
        return DeepURLClassifier(config=config)

    if args.model == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

    return DecisionTreeClassifier(
        max_depth=args.max_depth,
        random_state=args.random_state,
        class_weight="balanced",
    )


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(data_path)
    if args.label_col not in dataframe.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    dataframe = ensure_features(dataframe.copy())
    dataframe = dataframe.dropna(subset=[args.label_col])

    X = dataframe[NUMERIC_FEATURES].astype(float)
    y = dataframe[args.label_col].astype(int)

    urls = dataframe["url"].astype(str)
    X_train, X_test, y_train, y_test, urls_train, urls_test = train_test_split(
        X,
        y,
        urls,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_model(args)
    if args.model == "dl":
        model.fit(urls_train.tolist(), X_train, y_train.to_numpy())
        predictions = model.predict_with_urls(urls_test.tolist(), X_test)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, average="binary")),
        "classification_report": classification_report(y_test, predictions, digits=4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "label_mapping": {"0": "benign", "1": "malicious"},
    }

    feature_stats = {
        feature: {
            "mean": float(X_train[feature].mean()),
            "std": float(X_train[feature].std(ddof=0)) if float(X_train[feature].std(ddof=0)) > 0 else 1.0,
        }
        for feature in NUMERIC_FEATURES
    }

    artifact = {
        "model": model,
        "feature_names": NUMERIC_FEATURES,
        "feature_stats": feature_stats,
        "metrics": metrics,
        "training": {
            "rows": int(len(dataframe)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "model_type": args.model,
            "random_state": args.random_state,
            "dl_config": {
                "max_features": args.dl_max_features,
                "ngram_min": args.dl_ngram_min,
                "ngram_max": args.dl_ngram_max,
                "hidden_layers": args.dl_hidden_layers,
                "max_iter": args.dl_max_iter,
            }
            if args.model == "dl"
            else None,
        },
    }

    model_path = output_dir / "url_phishing_model.joblib"
    metrics_path = output_dir / "training_metrics.json"
    importance_path = output_dir / "feature_importance.csv"

    joblib.dump(artifact, model_path)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    importance_frame = pd.DataFrame(
        {
            "feature": NUMERIC_FEATURES,
            "importance": getattr(model, "feature_importances_", np.zeros(len(NUMERIC_FEATURES))),
        }
    ).sort_values("importance", ascending=False)
    importance_frame.to_csv(importance_path, index=False)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved importances to: {importance_path}")
    print("\nEvaluation summary")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(metrics["classification_report"])
    print("Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)


if __name__ == "__main__":
    main()