from typing import Iterable, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

from .constants import DATA_SOURCE_URL


def load_data(*args: list, **kwargs: dict) -> pd.DataFrame:
    """Load 'German Credit Scoring dataset'."""
    print(f"Loading dataset...")
    df = pd.read_csv(DATA_SOURCE_URL, *args, sep=',', engine="python", **kwargs)
    print(f"Dataset has been loaded! Shape: {df.shape}")
    return df


def get_confusion_matrix(estimator: Pipeline, X: Iterable, y: Iterable) -> ConfusionMatrixDisplay:
    """Create confusion matrix from a given fitted estimator and data."""
    disp = ConfusionMatrixDisplay.from_estimator(estimator=estimator, X=X, y=y, cmap="Blues", normalize="true")
    disp.ax_.set_title("Normalized confusion matrix")
    disp.ax_.grid(False)
    disp.figure_.set_size_inches(6, 6)

    return disp


def eval_metrics(actual: Iterable, pred: Iterable) -> Tuple[float, float]:
    """Calculate accuracy anf F1 scores."""
    acc = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred, average="macro")
    return acc, f1
