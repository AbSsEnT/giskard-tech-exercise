from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_val_score

from .constants import DATA_SOURCE_URL


def load_data(*args: list, **kwargs: dict) -> pd.DataFrame:
    """Load 'German Credit Scoring dataset'."""
    print(f"Loading dataset...")
    df = pd.read_csv(DATA_SOURCE_URL, *args, sep=',', engine="python", **kwargs)
    print(f"Dataset has been loaded! Shape: {df.shape}")
    return df


def plot_confusion_matrix(estimator: Pipeline, X: Iterable, y: Iterable) -> None:
    """Create confusion matrix from a given fitted estimator and data."""
    _, ax = plt.subplots(figsize=(5, 5))

    ConfusionMatrixDisplay.from_estimator(estimator=estimator, X=X, y=y, cmap="Blues", normalize="true", ax=ax)

    ax.set_title("Normalized Confusion-Matrix Plot")
    ax.grid(False)

    plt.plot()


def eval_metrics(actual: Iterable, pred: Iterable) -> None:
    """Calculate classification report."""
    print(f"-" * 60)
    print(classification_report(actual, pred, digits=3))
    print(f"-" * 60)


def plot_classes_distribution(Y: pd.Series) -> None:
    """Plot classes distribution."""
    plt.figure()
    plt.hist(Y)
    plt.title(f"Samples in total: {len(Y)}")
    plt.show()


def cross_validation_scoring(pipeline: Pipeline, X: Iterable, Y: Iterable) -> None:
    """Perform cross-validation repeated-cross validation to get less biased estimate of generalization loss."""
    folds_generator = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
    scores = cross_val_score(pipeline, X, Y, cv=folds_generator, scoring="f1_macro", n_jobs=-1, verbose=0)

    print(f"Cross-Validation F1-macro: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")


def cross_validation_grid_search(pipeline: Pipeline, params_grid: dict, X: Iterable, Y: Iterable) -> None:
    """Perform grid-search repeated-cross validation to get the best hyperparameters."""
    folds_generator = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
    searcher = GridSearchCV(pipeline, params_grid, cv=folds_generator, scoring="f1_macro", refit=False, n_jobs=-1)
    searcher.fit(X, Y)

    print(f"Best params: {searcher.best_params_}")
    print(f"Best F1-macro score: {searcher.best_score_: .3f}")
