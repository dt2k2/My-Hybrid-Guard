import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from url_ml.feature_engineering import NUMERIC_FEATURES, extract_url_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark URL detector and export FP/FN reports.")
    parser.add_argument("--data", default="Dataset.csv", help="Path to labeled CSV dataset.")
    parser.add_argument("--url-col", default="url", help="URL column name.")
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--model-path", default="artifacts/url_phishing_model.joblib", help="Trained model artifact path.")
    parser.add_argument("--mode", choices=["ml", "hybrid"], default="ml", help="ml: model-only, hybrid: full analyzer.")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional number of rows to evaluate.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--suspicious-as-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In hybrid mode, treat suspicious as positive when computing TP/FP.",
    )
    parser.add_argument("--output-dir", default="artifacts/benchmark", help="Output directory for reports.")
    return parser.parse_args()


def normalize_label(value) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer, float, np.floating)):
        return 1 if float(value) > 0 else 0

    text = str(value).strip().lower()
    positive_values = {"1", "true", "malicious", "phishing", "malware", "defacement", "bad"}
    negative_values = {"0", "false", "benign", "safe", "good", "legitimate"}
    if text in positive_values:
        return 1
    if text in negative_values:
        return 0
    return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = int(tp + tn + fp + fn)
    accuracy = float((tp + tn) / total) if total else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0

    return {
        "total": total,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
    }


def run_ml_mode(dataframe: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    artifact = joblib.load(args.model_path)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    model_type = artifact.get("training", {}).get("model_type", "rf")

    data = dataframe.copy()
    missing_features = [name for name in feature_names if name not in data.columns]
    if missing_features:
        generated = data[args.url_col].apply(extract_url_features).apply(pd.Series)
        for name in feature_names:
            data[name] = generated[name]

    X = data[feature_names].astype(float)

    if model_type == "dl" and hasattr(model, "predict_with_urls"):
        urls = data[args.url_col].astype(str).tolist()
        data["pred_binary"] = model.predict_with_urls(urls, X).astype(int)
        data["pred_score"] = model.predict_proba_with_urls(urls, X)[:, 1].astype(float)
    else:
        data["pred_binary"] = model.predict(X).astype(int)
        if hasattr(model, "predict_proba"):
            data["pred_score"] = model.predict_proba(X)[:, 1].astype(float)
        else:
            data["pred_score"] = data["pred_binary"].astype(float)

    data["pred_label"] = data["pred_binary"].map({0: "benign", 1: "malicious"})
    return data


def run_hybrid_mode(dataframe: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    from webapp.app.config import CACHE_PATH, MODEL_PATH
    from webapp.app.services.analyzer_service import UrlAnalyzerService
    from webapp.app.services.cache_store import JsonCacheStore
    from webapp.app.services.model_service import ModelService

    analyzer = UrlAnalyzerService(model_service=ModelService(MODEL_PATH), cache_store=JsonCacheStore(CACHE_PATH))

    records = []
    for index, row in dataframe.iterrows():
        result = analyzer.analyze(str(row[args.url_col]))
        prediction = result["final_prediction"]
        pred_binary = 1 if prediction == "malicious" else 0
        if args.suspicious_as_positive and prediction == "suspicious":
            pred_binary = 1

        records.append(
            {
                "row_index": int(index),
                "url": row[args.url_col],
                "pred_label": prediction,
                "pred_binary": pred_binary,
                "pred_score": float(result["final_malicious_probability"]),
            }
        )

    merged = dataframe.copy()
    prediction_frame = pd.DataFrame(records).set_index("row_index")
    merged = merged.merge(prediction_frame, left_index=True, right_index=True, how="left")
    return merged


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(args.data)
    if args.url_col not in dataframe.columns:
        raise ValueError(f"Missing URL column: {args.url_col}")
    if args.label_col not in dataframe.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    dataframe["true_binary"] = dataframe[args.label_col].apply(normalize_label)
    dataframe = dataframe.dropna(subset=["true_binary"]).copy()
    dataframe["true_binary"] = dataframe["true_binary"].astype(int)

    if args.sample_size and args.sample_size > 0 and args.sample_size < len(dataframe):
        dataframe = dataframe.sample(n=args.sample_size, random_state=args.random_state)

    if args.mode == "ml":
        evaluated = run_ml_mode(dataframe, args)
    else:
        evaluated = run_hybrid_mode(dataframe, args)

    y_true = evaluated["true_binary"].to_numpy(dtype=int)
    y_pred = evaluated["pred_binary"].to_numpy(dtype=int)
    metrics = compute_metrics(y_true, y_pred)

    summary = {
        "mode": args.mode,
        "dataset": str(Path(args.data).resolve()),
        "rows_evaluated": int(len(evaluated)),
        "url_column": args.url_col,
        "label_column": args.label_col,
        "suspicious_as_positive": bool(args.suspicious_as_positive),
        "metrics": metrics,
    }

    evaluated["error_type"] = "correct"
    evaluated.loc[(evaluated["true_binary"] == 0) & (evaluated["pred_binary"] == 1), "error_type"] = "false_positive"
    evaluated.loc[(evaluated["true_binary"] == 1) & (evaluated["pred_binary"] == 0), "error_type"] = "false_negative"

    summary_path = output_dir / "benchmark_summary.json"
    results_path = output_dir / "benchmark_predictions.csv"
    fp_path = output_dir / "false_positives.csv"
    fn_path = output_dir / "false_negatives.csv"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    evaluated.to_csv(results_path, index=False)
    evaluated[evaluated["error_type"] == "false_positive"].to_csv(fp_path, index=False)
    evaluated[evaluated["error_type"] == "false_negative"].to_csv(fn_path, index=False)

    print("Benchmark completed")
    print(f"Mode: {args.mode}")
    print(f"Rows: {len(evaluated)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved predictions: {results_path}")
    print(f"Saved false positives: {fp_path}")
    print(f"Saved false negatives: {fn_path}")


if __name__ == "__main__":
    main()
