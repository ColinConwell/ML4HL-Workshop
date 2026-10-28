"""Precompute scikit-learn decision surfaces for the ``classifier_zoo`` widget.

The widget runs in static pages with no Python kernel, so every model is fit
here across a sweep of its main hyperparameter and the resulting decision
regions are shipped to the browser as compact base64 grids.
"""

from __future__ import annotations

import base64

import numpy as np


def _models(n_train: int = 200):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    scaled = lambda est: make_pipeline(StandardScaler(), est)  # noqa: E731
    k_vals = [min(k, max(1, n_train - 1)) for k in (1, 3, 9, 25, 75)]
    return {
        "logistic": dict(
            label="Logistic regression", param="C (inverse penalty)",
            values=[0.001, 0.01, 0.1, 1, 100],
            make=lambda v: scaled(LogisticRegression(C=v, max_iter=5000)),
        ),
        "knn": dict(
            label="K-nearest neighbors", param="k (neighbors)",
            values=k_vals,
            make=lambda v: scaled(KNeighborsClassifier(n_neighbors=min(int(v), max(1, n_train - 1)))),
        ),
        "svm": dict(
            label="Support vector machine (RBF)", param="gamma (wiggliness)",
            values=[0.05, 0.3, 1.5, 8, 40],
            make=lambda v: CalibratedClassifierCV(scaled(SVC(C=1.0, gamma=v)), ensemble=False),
        ),
        "tree": dict(
            label="Decision tree", param="max depth",
            values=[1, 2, 4, 8, None],
            make=lambda v: DecisionTreeClassifier(max_depth=v, random_state=0),
        ),
        "gbm": dict(
            label="Gradient-boosted trees", param="boosting rounds",
            values=[1, 5, 20, 100, 400],
            make=lambda v: HistGradientBoostingClassifier(max_iter=v, learning_rate=0.15, random_state=0),
        ),
    }


def decision_grids(
    X,
    y,
    *,
    name: str = "dataset",
    class_names: list[str] | None = None,
    xlabel: str = "feature 1",
    ylabel: str = "feature 2",
    res: int = 72,
    test_size: float = 0.3,
    models: list[str] | None = None,
    seed: int = 0,
) -> dict:
    """Fit each model family across its hyperparameter sweep on a 2D dataset.

    Returns a JSON-ready dict for ``ml4hl.show("classifier_zoo", data=...)``.
    Pass a list of these dicts to let the widget switch between datasets.
    """
    from sklearn.model_selection import train_test_split

    X = np.asarray(X, dtype=float)
    y_raw = np.asarray(y)
    classes, y_idx = np.unique(y_raw, return_inverse=True)
    names = class_names or [str(c) for c in classes]

    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y_idx, random_state=seed)
    is_test = np.zeros(len(X), dtype=int)
    is_test[te] = 1

    pad = 0.08
    (x0, y0), (x1, y1) = X.min(0), X.max(0)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    xr, yr = [x0 - px, x1 + px], [y0 - py, y1 + py]
    gx = np.linspace(xr[0], xr[1], res)
    gy = np.linspace(yr[1], yr[0], res)  # top row first
    G = np.array([[a, b] for b in gy for a in gx])

    out_models = {}
    for key, spec in _models(len(tr)).items():
        if models and key not in models:
            continue
        grids, tr_acc, te_acc = [], [], []
        for v in spec["values"]:
            est = spec["make"](v).fit(X[tr], y_idx[tr])
            proba = est.predict_proba(G)
            cls = proba.argmax(1).astype(np.uint8)
            conf = np.clip((proba.max(1) - 1 / len(classes)) / (1 - 1 / len(classes)), 0, 1)
            packed = np.empty(res * res * 2, dtype=np.uint8)
            packed[0::2] = cls
            packed[1::2] = np.round(conf * 255).astype(np.uint8)
            grids.append(base64.b64encode(packed.tobytes()).decode("ascii"))
            tr_acc.append(round(float(est.score(X[tr], y_idx[tr])), 4))
            te_acc.append(round(float(est.score(X[te], y_idx[te])), 4))
        out_models[key] = dict(
            label=spec["label"], param=spec["param"],
            values=["none" if v is None else v for v in spec["values"]],
            grids=grids, train=tr_acc, test=te_acc,
        )

    return dict(
        name=name, classes=names, xlabel=xlabel, ylabel=ylabel,
        x=[float(xr[0]), float(xr[1])], y=[float(yr[0]), float(yr[1])], res=res,
        X=np.round(X, 4).tolist(), labels=y_idx.tolist(), test=is_test.tolist(),
        models=out_models,
    )
