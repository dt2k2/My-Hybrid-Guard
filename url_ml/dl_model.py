from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from url_ml.feature_engineering import normalize_url


@dataclass
class DLModelConfig:
    max_features: int = 40000
    ngram_min: int = 3
    ngram_max: int = 5
    hidden_layers: tuple[int, ...] = (256, 128, 64)
    max_iter: int = 80
    random_state: int = 42


class DeepURLClassifier:
    """Character-level URL encoder + neural classifier.

    This classifier learns URL patterns dynamically from data instead of relying
    on domain/hosting hardcoded rules.
    """

    def __init__(self, config: DLModelConfig | None = None) -> None:
        self.config = config or DLModelConfig()
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(self.config.ngram_min, self.config.ngram_max),
            max_features=self.config.max_features,
            min_df=2,
            lowercase=True,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.classifier = MLPClassifier(
            hidden_layer_sizes=self.config.hidden_layers,
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=self.config.max_iter,
            early_stopping=True,
            n_iter_no_change=6,
            random_state=self.config.random_state,
        )

    def _normalize_urls(self, urls: Iterable[str]) -> list[str]:
        return [normalize_url(url or "") for url in urls]

    def _build_matrix(self, urls: Sequence[str], numeric_frame: pd.DataFrame, fit: bool) -> sparse.csr_matrix:
        normalized_urls = self._normalize_urls(urls)
        numeric_values = numeric_frame.astype(float).to_numpy(copy=True)

        if fit:
            url_matrix = self.vectorizer.fit_transform(normalized_urls)
            numeric_scaled = self.scaler.fit_transform(numeric_values)
        else:
            url_matrix = self.vectorizer.transform(normalized_urls)
            numeric_scaled = self.scaler.transform(numeric_values)

        numeric_sparse = sparse.csr_matrix(numeric_scaled)
        return sparse.hstack([url_matrix, numeric_sparse], format="csr")

    def fit(self, urls: Sequence[str], numeric_frame: pd.DataFrame, labels: Sequence[int]) -> "DeepURLClassifier":
        matrix = self._build_matrix(urls, numeric_frame, fit=True)
        y = np.asarray(labels, dtype=int)
        self.classifier.fit(matrix, y)
        return self

    def predict_with_urls(self, urls: Sequence[str], numeric_frame: pd.DataFrame) -> np.ndarray:
        matrix = self._build_matrix(urls, numeric_frame, fit=False)
        return self.classifier.predict(matrix)

    def predict_proba_with_urls(self, urls: Sequence[str], numeric_frame: pd.DataFrame) -> np.ndarray:
        matrix = self._build_matrix(urls, numeric_frame, fit=False)
        return self.classifier.predict_proba(matrix)

    def predict(self, feature_frame: pd.DataFrame) -> np.ndarray:
        if "__url__" not in feature_frame.columns:
            raise ValueError("DeepURLClassifier requires '__url__' column in feature frame.")
        numeric_frame = feature_frame.drop(columns=["__url__"])
        return self.predict_with_urls(feature_frame["__url__"].astype(str).tolist(), numeric_frame)

    def predict_proba(self, feature_frame: pd.DataFrame) -> np.ndarray:
        if "__url__" not in feature_frame.columns:
            raise ValueError("DeepURLClassifier requires '__url__' column in feature frame.")
        numeric_frame = feature_frame.drop(columns=["__url__"])
        return self.predict_proba_with_urls(feature_frame["__url__"].astype(str).tolist(), numeric_frame)
