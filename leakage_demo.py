"""Reproducible demonstration of target leakage in feature selection."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_demo(random_state: int = 42) -> dict[str, float]:
    X, y = make_classification(
        n_samples=1200, n_features=1000, n_informative=10,
        n_redundant=10, n_classes=2, weights=[0.5, 0.5],
        class_sep=0.5, flip_y=0.1, random_state=random_state,
    )
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.30, stratify=y, random_state=random_state
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # INCORRECT: feature selection sees all labels, including holdout labels.
    leaked_selector = SelectKBest(score_func=f_classif, k=25)
    X_selected_all = leaked_selector.fit_transform(X, y)
    leaked_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, random_state=random_state)),
    ])
    leaked_model.fit(X_selected_all[train_idx], y_train)
    leaked_prob = leaked_model.predict_proba(X_selected_all[test_idx])[:, 1]
    leaked_pred = leaked_model.predict(X_selected_all[test_idx])

    # CORRECT: each learned transformation is fitted on training data only.
    safe_model = Pipeline([
        ("feature_selection", SelectKBest(score_func=f_classif, k=25)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, random_state=random_state)),
    ])
    safe_model.fit(X_train, y_train)
    safe_prob = safe_model.predict_proba(X_test)[:, 1]
    safe_pred = safe_model.predict(X_test)

    return {
        "leakage_prone_accuracy": float(accuracy_score(y_test, leaked_pred)),
        "leakage_prone_roc_auc": float(roc_auc_score(y_test, leaked_prob)),
        "leakage_free_accuracy": float(accuracy_score(y_test, safe_pred)),
        "leakage_free_roc_auc": float(roc_auc_score(y_test, safe_prob)),
    }


if __name__ == "__main__":
    for name, value in run_demo().items():
        print(f"{name.replace('_', ' ').title():28s}: {value:.4f}")
    print("\nThe leakage-prone result is biased because feature selection used")
    print("labels from the holdout set before the train/test boundary was respected.")
