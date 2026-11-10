"""Initial sampling, classifier training, and simulated active learning."""

import hashlib
import inspect
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tqdm.auto import tqdm

from .data import (
    REPOSITORY_ROOT,
    build_representation,
    load_20newsgroups,
    load_banking77,
    select_dataset_classes,
)
from .policies import (
    acquire_batch,
    initialize_policy_state,
    resolve_round_policy,
    update_adaptive_state,
)

_TFIDF_PARAMETERS = frozenset(inspect.signature(TfidfVectorizer).parameters)


def create_initial_pool(
    df, initial_size: int, method: str = "stratified_oracle", random_state: int = 42
):
    """Return positional indices for a labeled set and remaining training pool.

    ``stratified_oracle`` and ``coverage`` use true labels and are simulation
    only. Stratification approximates class proportions; coverage selects one
    example per class, then fills randomly. ``random`` uses no class labels.
    """
    if "split" in df and not df["split"].isin(["train", "unspecified"]).all():
        raise ValueError("Initial pool must contain training rows only.")
    if not 1 <= initial_size < len(df):
        raise ValueError(
            "initial_size must be positive and smaller than the training set."
        )
    rng = np.random.default_rng(random_state)
    positions = np.arange(len(df))
    if method == "random":
        selected = rng.choice(positions, initial_size, replace=False)
    elif method in {"stratified_oracle", "coverage"}:
        labels = df["label"].to_numpy()
        groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        if initial_size < len(groups):
            raise ValueError(
                "initial_size must cover every class for oracle initialization."
            )
        selected = [int(rng.choice(group)) for group in groups]
        remaining = positions[~np.isin(positions, selected)]
        if method == "stratified_oracle":
            # Allocate the remaining slots proportionally to remaining class sizes.
            sizes = np.array([len(group) - 1 for group in groups])
            ideal = (initial_size - len(groups)) * sizes / sizes.sum()
            counts = np.floor(ideal).astype(int)
            for index in np.argsort(-(ideal - counts), kind="stable")[
                : initial_size - len(groups) - counts.sum()
            ]:
                counts[index] += 1
            for group, count in zip(groups, counts):
                candidates = group[~np.isin(group, selected)]
                selected.extend(
                    rng.choice(candidates, int(count), replace=False).tolist()
                )
        else:
            selected.extend(
                rng.choice(
                    remaining, initial_size - len(selected), replace=False
                ).tolist()
            )
        selected = np.asarray(selected, dtype=int)
    else:
        raise ValueError(f"Unknown initialization method: {method!r}")
    return selected, positions[~np.isin(positions, selected)]


def initialize_until_coverage(
    X,
    labels,
    candidates,
    known_classes,
    method="random",
    min_per_class=2,
    total_budget=200,
    random_state=42,
):
    """Query labels sequentially until coverage; selection uses features only.

    NNC greedily minimizes summed Euclidean distance to the selected set. Its
    exact pairwise distance matrix requires O(n²) memory and is costly for large
    pools. For a fixed candidate order, NNC is deterministic across seeds;
    ties follow candidate order. K-means++ samples in proportion to squared
    Euclidean distance. Labels only determine when coverage is complete.
    """
    if method not in {"random", "kmeans++", "nnc"}:
        raise ValueError("Coverage initialization requires random, kmeans++, or nnc.")
    if min_per_class < 1 or total_budget < 0:
        raise ValueError("min_per_class must be positive and total_budget nonnegative.")
    candidates = np.asarray(candidates, dtype=int)
    labels = np.asarray(labels)
    classes = np.asarray(known_classes)
    if not len(classes) or len(np.unique(classes)) != len(classes):
        raise ValueError("known_classes must contain unique class labels.")
    rng = np.random.default_rng(random_state)
    counts = {label: 0 for label in classes}
    remaining = np.ones(len(candidates), dtype=bool)
    nearest = None
    distances = (
        euclidean_distances(X[candidates])
        if method == "nnc" and len(candidates)
        else None
    )
    selected = []
    while (
        remaining.any()
        and len(selected) < total_budget
        and min(counts.values()) < min_per_class
    ):
        available = np.flatnonzero(remaining)
        if method == "nnc":
            current = np.full(len(candidates), np.inf) if nearest is None else nearest
            # Each column is a possible new center; all candidates contribute.
            costs = np.minimum(current[:, None], distances[:, available]).sum(axis=0)
            choice = int(available[np.argmin(costs)])
        elif method == "random" or nearest is None:
            choice = int(rng.choice(available))
        else:
            weights = nearest[available] ** 2
            choice = (
                int(rng.choice(available, p=weights / weights.sum()))
                if weights.sum() > 0
                else int(rng.choice(available))
            )
        selected.append(int(candidates[choice]))
        remaining[choice] = False
        label = labels[
            candidates[choice]
        ]  # Simulated oracle; never used for selection.
        if label not in counts:
            raise ValueError("An observed label is absent from known_classes.")
        counts[label] += 1
        if method != "random":
            distance = (
                distances[:, choice]
                if method == "nnc"
                else euclidean_distances(
                    X[candidates], X[candidates[choice : choice + 1]]
                ).ravel()
            )
            nearest = distance if nearest is None else np.minimum(nearest, distance)
    return (
        np.asarray(selected, dtype=int),
        candidates[remaining],
        counts,
        min(counts.values()) >= min_per_class,
    )


def _check_nonnegative(X):
    """Reject signed features without altering sparse or dense representations."""
    values = X.data if sparse.issparse(X) else np.asarray(X)
    if np.any(values < 0):
        raise ValueError(
            "MultinomialNB requires non-negative features; the provided "
            "representation contains negative values."
        )


def train_classifier(
    X,
    labels,
    labeled_idx,
    model_type: str = "logistic_regression",
    random_state: int = 42,
):
    """Fit a named classifier, or clone a supplied scikit-learn estimator."""
    labels = np.asarray(labels)
    if len(np.unique(labels[np.asarray(labeled_idx, dtype=int)])) < 2:
        raise ValueError("Classifier training requires at least two labeled classes.")
    if isinstance(model_type, str):
        if model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_type == "multinomial_nb":
            model = MultinomialNB()
        else:
            raise ValueError(f"Unknown classifier: {model_type!r}.")
    else:
        model = clone(model_type)
    if isinstance(model, MultinomialNB):
        _check_nonnegative(X[np.asarray(labeled_idx, dtype=int)])
    model.fit(
        X[np.asarray(labeled_idx, dtype=int)],
        labels[np.asarray(labeled_idx, dtype=int)],
    )
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The classifier must provide predict_proba for uncertainty policies."
        )
    return model


def run_active_learning(
    X_train,
    y_train,
    X_eval,
    y_eval,
    labeled_idx,
    pool_idx,
    policy: dict,
    budget: int = 200,
    batch_size: int = 20,
    random_state: int = 42,
    sample_ids=None,
    model_type="logistic_regression",
    policy_name=None,
    dataset_id="unspecified",
    representation=None,
    initial_method="provided",
    data_signature=None,
    total_budget=None,
    stopping=None,
    monitoring_idx=None,
    monitoring_seed=None,
    initialization_complete=True,
):
    """Train, evaluate, acquire and reveal labels until the budget is exhausted.

    All indices are positional in X_train; evaluation data is separate. Pool
    labels are accessed only after selection to simulate annotation. Returns a
    dict containing configuration, per-round metrics, and acquisition records.
    """
    if total_budget is not None:
        if budget is not None:
            raise ValueError("Specify budget or total_budget, not both.")
        if total_budget < len(labeled_idx):
            raise ValueError("Initial labels exceed total_budget.")
        budget = total_budget - len(labeled_idx)
    if budget is None or budget < 0 or batch_size <= 0:
        raise ValueError("budget must be nonnegative and batch_size positive.")
    if (isinstance(model_type, str) and model_type == "multinomial_nb") or isinstance(
        model_type, MultinomialNB
    ):
        _check_nonnegative(X_train)
        _check_nonnegative(X_eval)
    y_train, y_eval = np.asarray(y_train), np.asarray(y_eval)
    labeled = np.asarray(labeled_idx, dtype=int).copy()
    pool = np.asarray(pool_idx, dtype=int).copy()
    if (
        len(y_train) != X_train.shape[0]
        or len(y_eval) != X_eval.shape[0]
        or np.intersect1d(labeled, pool).size
        or len(np.unique(labeled)) != len(labeled)
        or len(np.unique(pool)) != len(pool)
        or np.any(labeled < 0)
        or np.any(pool < 0)
        or np.any(labeled >= len(y_train))
        or np.any(pool >= len(y_train))
    ):
        raise ValueError("Train/evaluation sizes or labeled/pool indices are invalid.")
    ids = np.asarray(sample_ids if sample_ids is not None else np.arange(len(y_train)))
    if len(ids) != len(y_train) or len(np.unique(ids)) != len(ids):
        raise ValueError("sample_ids must uniquely identify all training rows.")
    monitoring = np.asarray(
        monitoring_idx if monitoring_idx is not None else [], dtype=int
    )
    if (
        len(np.unique(monitoring)) != len(monitoring)
        or np.any(monitoring < 0)
        or np.any(monitoring >= len(y_train))
        or np.intersect1d(monitoring, np.concatenate([labeled, pool])).size
    ):
        raise ValueError(
            "Monitoring indices must be unique and excluded from training and acquisition."
        )
    stopping = stopping or {}
    if set(stopping) - {"target_performance", "prediction_stability"}:
        raise ValueError("Unknown stopping criterion.")
    target = stopping.get("target_performance")
    stability = stopping.get("prediction_stability")
    if target is not None and (
        target.get("metric") not in {"accuracy", "macro_f1"}
        or not 0 <= target.get("threshold", -1) <= 1
    ):
        raise ValueError(
            "Target stopping requires accuracy or macro_f1 and a threshold in [0, 1]."
        )
    if stability is not None and (
        stability.get("metric", "cohen_kappa") != "cohen_kappa"
        or not -1 <= stability.get("threshold", 0.99) <= 1
        or not isinstance(stability.get("window", 3), int)
        or stability.get("window", 3) < 1
        or len(monitoring) < 2
    ):
        raise ValueError(
            "Stability stopping requires Cohen kappa, a positive window, and at least two monitoring examples."
        )
    config = {
        "policy": dict(policy),
        "policy_name": policy_name or policy["type"],
        "budget": budget,
        "batch_size": batch_size,
        "random_state": random_state,
        "initial_labeled_size": len(labeled),
        "initial_class_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(y_train[labeled], return_counts=True))
        },
        "model_type": str(model_type),
        "dataset_id": dataset_id,
        "representation": _representation_metadata(representation),
        "initial_method": initial_method,
        "data_signature": data_signature
        or _data_signature(X_train, y_train, X_eval, y_eval, ids),
    }
    if total_budget is not None:
        config.update(
            total_budget=total_budget,
            initialization_complete=initialization_complete,
            stopping=stopping,
            monitoring_size=len(monitoring),
            monitoring_seed=monitoring_seed,
        )
    rng = np.random.default_rng(random_state)
    total_rounds = (min(budget, len(pool)) + batch_size - 1) // batch_size
    policy_state = initialize_policy_state(policy, total_rounds)
    history, acquired, policy_history = [], [], []
    round_number = 0
    cumulative = 0
    previous_predictions = None
    kappas = []
    stop_reasons = []
    if total_budget is not None and len(np.unique(y_train[labeled])) < 2:
        stop_reasons = [
            "initialization_incomplete"
            if not initialization_complete
            else "insufficient_classes"
        ]
        if cumulative >= budget:
            stop_reasons.append("max_budget")
        if not len(pool):
            stop_reasons.append("pool_exhausted")
    while True:
        if stop_reasons:
            break
        started = time.perf_counter()
        model = train_classifier(X_train, y_train, labeled, model_type, random_state)
        train_seconds = time.perf_counter() - started
        prediction = model.predict(X_eval)
        row = {
            "policy": config["policy_name"],
            "seed": random_state,
            "round": round_number,
            "initial_labeled_size": config["initial_labeled_size"],
            "acquired_labels": cumulative,
            "training_size": len(labeled),
            "accuracy": accuracy_score(y_eval, prediction),
            "macro_f1": f1_score(y_eval, prediction, average="macro", zero_division=0),
            "training_seconds": train_seconds,
            "acquisition_seconds": 0.0,
        }
        history.append(row)
        if stability is not None:
            current_predictions = model.predict(X_train[monitoring])
            if previous_predictions is not None:
                kappa = cohen_kappa_score(previous_predictions, current_predictions)
                # Identical constant predictions have undefined kappa but perfect agreement.
                if np.isnan(kappa) and np.array_equal(
                    previous_predictions, current_predictions
                ):
                    kappa = 1.0
                kappas.append(float(kappa))
                row["cohen_kappa"] = kappas[-1]
                if len(kappas) >= stability.get("window", 3):
                    row["stability_mean"] = float(
                        np.mean(kappas[-stability.get("window", 3) :])
                    )
            previous_predictions = current_predictions
        if total_budget is not None:
            if target is not None and row[target["metric"]] >= target["threshold"]:
                stop_reasons.append("target_performance")
            if stability is not None and row.get(
                "stability_mean", -np.inf
            ) >= stability.get("threshold", 0.99):
                stop_reasons.append("prediction_stability")
            if cumulative >= budget:
                stop_reasons.append("max_budget")
            if not len(pool):
                stop_reasons.append("pool_exhausted")
            if not initialization_complete:
                stop_reasons.append("initialization_incomplete")
            if stop_reasons:
                break
        elif cumulative >= budget or not len(pool):
            break
        count = min(batch_size, budget - cumulative, len(pool))
        started = time.perf_counter()
        round_policy = resolve_round_policy(policy, round_number + 1, policy_state)
        beta_before = policy_state.get("beta")
        # No hidden labels or evaluation data are passed to the policy.
        selected, details = acquire_batch(
            round_policy, model, X_train, labeled, pool, count, rng, return_info=True
        )
        if policy["type"] == "adaptive":
            # Capture the current model's predictions before the oracle reveals
            # the batch labels and before any training on those examples.
            probabilities = model.predict_proba(X_train[selected])
        row["acquisition_seconds"] = time.perf_counter() - started
        batch_labels = y_train[selected]  # Simulated oracle reveals labels here.
        decision = {"round": round_number + 1, "strategy": round_policy["type"]}
        if policy["type"] == "adaptive":
            class_positions = {label: pos for pos, label in enumerate(model.classes_)}
            # Random initialization may omit a class. Its probability is zero;
            # clipping keeps the negative log-likelihood finite and explicit.
            true_probabilities = np.array(
                [
                    probabilities[i, class_positions[label]]
                    if label in class_positions
                    else 0.0
                    for i, label in enumerate(batch_labels)
                ]
            )
            batch_loss = float(-np.log(np.clip(true_probabilities, 1e-12, 1.0)).mean())
            policy_state, normalized = update_adaptive_state(
                policy, policy_state, batch_loss
            )
            decision.update(
                {
                    "beta_before": beta_before,
                    "batch_loss": batch_loss,
                    "normalized_loss": normalized,
                    "beta_after": policy_state["beta"],
                    "window_losses": list(policy_state["losses"]),
                }
            )
        if policy["type"] in {"adaptive", "scheduled"}:
            policy_history.append(decision)
        for index, detail in zip(selected, details):
            acquired.append(
                {
                    "policy": config["policy_name"],
                    "seed": random_state,
                    "round": round_number + 1,
                    "sample_idx": int(index),
                    "sample_id": ids[index],
                    "label": y_train[index],
                    **detail,
                }
            )
        labeled = np.concatenate([labeled, selected])
        pool = pool[~np.isin(pool, selected)]
        cumulative += len(selected)
        round_number += 1
    config["completed_rounds"] = round_number
    config["actual_acquired_labels"] = cumulative
    if total_budget is not None:
        config.update(
            actual_labels_consumed=len(labeled),
            acquired_labels=cumulative,
            training_size=len(labeled),
            stop_reasons=stop_reasons,
            stop_reason=stop_reasons[0],
            labels_saved=total_budget - len(labeled),
            finished_early=len(labeled) < total_budget,
        )
    result = {
        "configuration": config,
        "history": pd.DataFrame(history),
        "acquisitions": pd.DataFrame(acquired),
        "initial_labeled_idx": np.asarray(labeled_idx, dtype=int).copy(),
        "initial_labeled_ids": ids[np.asarray(labeled_idx, dtype=int)].copy(),
        "initial_pool_ids": ids[np.asarray(pool_idx, dtype=int)].copy(),
        "policy_history": pd.DataFrame(policy_history),
    }
    if total_budget is not None:
        result["monitoring_ids"] = ids[monitoring].copy()
    return result


def _representation_metadata(representation):
    """Record representation identity without retaining its fitted model."""
    representation = representation or {"method": "unspecified"}
    metadata = {
        key: representation[key]
        for key in ("method", "model_name")
        if key in representation
    }
    if metadata.get("method") == "tfidf":
        metadata.update(
            {
                key: value
                for key, value in representation.items()
                if key in _TFIDF_PARAMETERS
            }
        )
    if metadata.get("method") == "embeddings" and not metadata.get("model_name"):
        metadata["model_name"] = "glove-wiki-gigaword-50"
    return metadata


def _data_signature(X_train, y_train, X_eval, y_eval, sample_ids):
    """Fingerprint experiment inputs so comparisons cannot mix data splits."""
    digest = hashlib.sha256()
    for matrix in (X_train, X_eval):
        digest.update(str(matrix.shape).encode())
        if sparse.issparse(matrix):
            matrix = matrix.tocsr()
            for part in (matrix.data, matrix.indices, matrix.indptr):
                digest.update(np.ascontiguousarray(part).tobytes())
        else:
            digest.update(np.ascontiguousarray(matrix).tobytes())
    for values in (y_train, y_eval, sample_ids):
        digest.update(
            pd.util.hash_pandas_object(pd.Series(values), index=False)
            .to_numpy()
            .tobytes()
        )
    return digest.hexdigest()


def _prepare_runner_inputs(experiment):
    """Convert a dataset configuration to the inputs used by both runners."""
    required = {"X_train", "y_train", "X_eval", "y_eval", "train_df"}
    present = required.intersection(experiment)
    if present == required:
        return experiment
    if present:
        raise ValueError(
            "Provide all prepared inputs or use a declarative dataset configuration; "
            f"missing: {', '.join(sorted(required - present))}."
        )
    if "dataset" not in experiment:
        raise ValueError("Provide prepared inputs or a dataset configuration.")
    representation = experiment.get("representation", {"method": "tfidf"})
    if isinstance(representation, str):
        representation = {"method": representation}
    if not isinstance(representation, dict):
        raise TypeError(
            "representation must be a method name or a configuration dictionary."
        )
    train, evaluation, split_id = prepare_dataset_experiment(experiment)
    if (
        experiment.get("classes") is not None
        or experiment.get("max_classes") is not None
    ) and "known_classes" in experiment:
        selected_ids = {item["id"] for item in train.attrs["selected_classes"]}
        if set(experiment["known_classes"]) != selected_ids:
            raise ValueError("known_classes must match the selected class IDs.")
    X_train, X_eval, metadata = build_representation(
        train["text"], evaluation["text"], **representation
    )
    prepared = dict(experiment)
    prepared.update(
        X_train=X_train,
        y_train=train["label"].to_numpy(),
        X_eval=X_eval,
        y_eval=evaluation["label"].to_numpy(),
        train_df=train,
        representation=_representation_metadata(metadata),
        split_id=split_id,
    )
    prepared.setdefault("dataset_id", experiment["dataset"])
    if "classifier" in experiment:
        prepared.setdefault("model_type", experiment["classifier"])
    return prepared


def run_experiments(experiment: dict, policies: dict, show_progress=True):
    """Run every named policy from one matched initial split per seed.

    ``experiment`` may specify a dataset and representation, or provide
    ``X_train``, ``y_train``, ``X_eval``, ``y_eval`` and aligned ``train_df``.
    Optional keys are ``seeds``,
    ``initial_size``, ``initial_method``, ``budget``, ``batch_size``,
    ``classifier`` (or legacy ``model_type``), ``dataset_id`` and
    ``representation``. If both classifier names are supplied, ``classifier``
    takes precedence. Policies remain a
    separate mapping from display names to policy configurations.

    Each policy run creates its own seeded RNG, so changing dictionary order
    does not affect its selections. Results store an input fingerprint rather
    than retaining feature matrices or fitted representation models.
    Set ``show_progress=False`` to suppress the policy-by-seed progress bar.
    """
    if "total_budget" in experiment:
        if "budget" in experiment:
            raise ValueError("Specify budget or total_budget, not both.")
        raise ValueError("Use run_conditions for total-budget experiments.")
    experiment = _prepare_runner_inputs(experiment)
    X_train = experiment["X_train"]
    y_train = experiment["y_train"]
    X_eval = experiment["X_eval"]
    y_eval = experiment["y_eval"]
    train_df = experiment["train_df"]
    seeds = experiment.get("seeds", (42, 43, 44, 45, 46))
    initial_size = experiment.get("initial_size", 90)
    initial_method = experiment.get("initial_method", "stratified_oracle")
    budget = experiment.get("budget", 200)
    batch_size = experiment.get("batch_size", 20)
    model_type = experiment.get(
        "classifier", experiment.get("model_type", "logistic_regression")
    )
    dataset_id = experiment.get("dataset_id", "unspecified")
    representation = experiment.get("representation")
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not policies:
        raise ValueError("Provide at least one policy and unique experiment seeds.")
    if len(train_df) != X_train.shape[0] or not np.array_equal(
        train_df["label"].to_numpy(), np.asarray(y_train)
    ):
        raise ValueError("train_df must align with X_train and y_train by row.")
    sample_ids = train_df["sample_id"].to_numpy()
    representation = _representation_metadata(representation)
    signature = _data_signature(X_train, y_train, X_eval, y_eval, sample_ids)
    results = []
    with tqdm(
        total=len(seeds) * len(policies),
        desc="Experiments",
        unit="run",
        disable=not show_progress,
    ) as progress:
        for seed in seeds:
            labeled, pool = create_initial_pool(
                train_df,
                initial_size=initial_size,
                method=initial_method,
                random_state=seed,
            )
            for name, policy in policies.items():
                if show_progress:
                    progress.set_postfix(policy=name, seed=seed)
                result = run_active_learning(
                    X_train,
                    y_train,
                    X_eval,
                    y_eval,
                    labeled,
                    pool,
                    policy,
                    budget=budget,
                    batch_size=batch_size,
                    random_state=seed,
                    sample_ids=sample_ids,
                    model_type=model_type,
                    policy_name=name,
                    dataset_id=dataset_id,
                    representation=representation,
                    initial_method=initial_method,
                    data_signature=signature,
                )
                if "selected_classes" in train_df.attrs:
                    result["configuration"]["selected_classes"] = train_df.attrs[
                        "selected_classes"
                    ]
                results.append(result)
                progress.update(1)
    return results


def run_conditions(experiment, conditions, policy, show_progress=True):
    """Compare named total-budget conditions with matched seeded inputs.

    A monitoring subset is held out for every condition if any condition uses
    stability stopping. Monitoring examples are never labeled or trained on.
    The known class set defaults to the dataset's class metadata (unique labels).
    ``candidate_size`` optionally limits initialization to a seeded, label-free
    sample; all other eligible examples remain available for later acquisition.
    Inputs may be prepared matrices or a dataset and representation configuration.
    """
    if "budget" in experiment or any("budget" in c for c in conditions.values()):
        raise ValueError("Specify total_budget without the additional-label budget.")
    if "total_budget" not in experiment or not conditions:
        raise ValueError("Provide a shared total_budget and named conditions.")
    experiment = _prepare_runner_inputs(experiment)
    allowed = {
        "initial_method",
        "initial_size",
        "candidate_size",
        "min_per_class",
        "batch_size",
        "stopping",
    }
    for overrides in conditions.values():
        if set(overrides) - allowed:
            raise ValueError(
                "Conditions may override initialization, batch size, and stopping only."
            )
    total_budget = experiment["total_budget"]
    if not isinstance(total_budget, int) or total_budget < 1:
        raise ValueError("total_budget must be a positive integer.")
    seeds = tuple(experiment.get("seeds", (42, 43, 44, 45, 46)))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one unique seed.")
    X, y = experiment["X_train"], np.asarray(experiment["y_train"])
    X_eval, y_eval = experiment["X_eval"], experiment["y_eval"]
    df = experiment["train_df"]
    if len(df) != X.shape[0] or not np.array_equal(df["label"].to_numpy(), y):
        raise ValueError("train_df must align with X_train and y_train by row.")
    if "split" in df and not df["split"].isin(["train", "unspecified"]).all():
        raise ValueError("Training candidates must contain training rows only.")
    ids = df["sample_id"].to_numpy()
    settings = {
        name: {**experiment, **overrides} for name, overrides in conditions.items()
    }
    uses_monitor = any(
        (cfg.get("stopping") or {}).get("prediction_stability") is not None
        for cfg in settings.values()
    )
    monitor_size = (
        experiment.get("monitoring_size", min(200, len(df) // 10))
        if uses_monitor
        else 0
    )
    if uses_monitor and (
        not isinstance(monitor_size, int) or not 2 <= monitor_size < len(df)
    ):
        raise ValueError(
            "monitoring_size must be at least two and smaller than the training set."
        )
    # Prefer dataset class metadata; never inspect monitoring labels when it is supplied.
    classes = (
        experiment["known_classes"] if "known_classes" in experiment else np.unique(y)
    )
    results = []
    with tqdm(
        total=len(seeds) * len(conditions),
        desc="Conditions",
        unit="run",
        disable=not show_progress,
    ) as progress:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            monitor = (
                np.sort(rng.choice(len(df), monitor_size, replace=False))
                if monitor_size
                else np.array([], dtype=int)
            )
            candidates = np.setdiff1d(np.arange(len(df)), monitor)
            # Monitoring labels play no role, including in the input fingerprint.
            signature_labels = np.empty_like(y)
            signature_labels[:] = y[candidates[0]]
            signature_labels[candidates] = y[candidates]
            signature = _data_signature(X, signature_labels, X_eval, y_eval, ids)
            for name, cfg in settings.items():
                if show_progress:
                    progress.set_postfix(condition=name, seed=seed)
                method = cfg.get("initial_method", "random")
                candidate_size = cfg.get("candidate_size")
                if candidate_size is None:
                    initial_candidates = candidates
                else:
                    if (
                        isinstance(candidate_size, bool)
                        or not isinstance(candidate_size, (int, np.integer))
                        or not 1 <= candidate_size <= len(candidates)
                    ):
                        raise ValueError(
                            "candidate_size must be a positive integer no larger "
                            "than the eligible training pool."
                        )
                    initial_candidates = np.sort(
                        np.random.default_rng(seed).choice(
                            candidates, size=candidate_size, replace=False
                        )
                    )
                if cfg.get("min_per_class") is not None:
                    labeled, _, counts, complete = initialize_until_coverage(
                        X,
                        y,
                        initial_candidates,
                        classes,
                        method,
                        cfg["min_per_class"],
                        total_budget,
                        seed,
                    )
                else:
                    size = cfg.get("initial_size", 90)
                    if size > total_budget:
                        raise ValueError("initial_size exceeds total_budget.")
                    if candidate_size is not None and size == len(initial_candidates):
                        if method not in {"random", "stratified_oracle", "coverage"}:
                            raise ValueError(
                                f"Unknown initialization method: {method!r}"
                            )
                        labeled = initial_candidates.copy()
                    else:
                        chosen, _ = create_initial_pool(
                            df.iloc[initial_candidates], size, method, seed
                        )
                        labeled = initial_candidates[chosen]
                    counts = {
                        label: int(np.sum(y[labeled] == label)) for label in classes
                    }
                    complete = True
                # Restore every unannotated eligible example, including those
                # outside the initialization subset; monitoring stays excluded.
                pool = candidates[~np.isin(candidates, labeled)]
                result = run_active_learning(
                    X,
                    y,
                    X_eval,
                    y_eval,
                    labeled,
                    pool,
                    dict(policy),
                    budget=None,
                    total_budget=total_budget,
                    batch_size=cfg.get("batch_size", 20),
                    random_state=seed,
                    sample_ids=ids,
                    model_type=cfg.get(
                        "classifier", cfg.get("model_type", "logistic_regression")
                    ),
                    policy_name=cfg.get("policy_name", policy["type"]),
                    dataset_id=cfg.get("dataset_id", "unspecified"),
                    representation=cfg.get("representation"),
                    initial_method=method,
                    data_signature=signature,
                    stopping=cfg.get("stopping"),
                    monitoring_idx=monitor,
                    monitoring_seed=seed if uses_monitor else None,
                    initialization_complete=complete,
                )
                result["configuration"].update(
                    condition=name,
                    min_per_class=cfg.get("min_per_class"),
                    initialization_candidate_size=len(initial_candidates),
                    initialization_incomplete_reason=(
                        "budget_exhausted"
                        if len(labeled) >= total_budget
                        else "candidate_subset_exhausted"
                        if candidate_size is not None
                        else "eligible_pool_exhausted"
                    )
                    if not complete
                    else None,
                    initial_class_counts={str(k): int(v) for k, v in counts.items()},
                )
                if "selected_classes" in df.attrs:
                    result["configuration"]["selected_classes"] = df.attrs[
                        "selected_classes"
                    ]
                results.append(result)
                progress.update(1)
    return results


def _stratified_limit(frame, limit, minimum, seed, name):
    """Sample a fixed maximum with per-class coverage and proportional remainder."""
    if limit is None:
        return frame.reset_index(drop=True)
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit < 1:
        raise ValueError(f"{name} must be a positive integer or None.")
    if limit >= len(frame):
        return frame.reset_index(drop=True)
    groups = [group for _, group in frame.groupby("label", sort=True)]
    if limit < minimum * len(groups) or any(len(group) < minimum for group in groups):
        raise ValueError(f"{name} is too small for the required per-class coverage.")
    capacity = np.array([len(group) - minimum for group in groups])
    extra = limit - minimum * len(groups)
    ideal = extra * capacity / capacity.sum()
    allocation = np.floor(ideal).astype(int)
    for i in np.argsort(-(ideal - allocation), kind="stable")[
        : extra - allocation.sum()
    ]:
        allocation[i] += 1
    rng = np.random.default_rng(seed)
    chosen = np.concatenate(
        [
            rng.choice(group.index.to_numpy(), minimum + count, replace=False)
            for group, count in zip(groups, allocation)
        ]
    )
    return frame.loc[np.sort(chosen)].reset_index(drop=True)


def prepare_dataset_experiment(config):
    """Load one dataset and create a fixed stratified train/validation split.

    Optional limits are applied before feature construction. Official test rows
    are never used. Returns aligned dataframes and a split fingerprint.
    """
    dataset = config["dataset"]
    loaders = {"banking77": load_banking77, "20newsgroups": load_20newsgroups}
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset!r}.")
    per_class = config.get("initial_per_class", 5)
    if (
        isinstance(per_class, bool)
        or not isinstance(per_class, (int, np.integer))
        or per_class < 1
    ):
        raise ValueError("initial_per_class must be a positive integer.")
    data = loaders[dataset](config.get("cache_dir", "data"))
    data = select_dataset_classes(
        data,
        classes=config.get("classes"),
        max_classes=config.get("max_classes"),
        random_state=config.get("split_seed", 42),
    )
    selected_classes = data.attrs["selected_classes"]
    source = data[data["split"] == "train"].reset_index(drop=True)
    if source.empty or source["sample_id"].duplicated().any():
        raise ValueError(
            "The source training split must contain uniquely identified examples."
        )
    classes = np.asarray([item["id"] for item in selected_classes])
    counts = source["label"].value_counts().reindex(classes, fill_value=0)
    if (counts < per_class + 1).any():
        raise ValueError(
            "Each known class needs enough training and validation examples."
        )
    split_seed = config.get("split_seed", 42)
    try:
        train, evaluation = train_test_split(
            source,
            test_size=config.get("validation_fraction", 0.2),
            random_state=split_seed,
            stratify=source["label"],
        )
    except ValueError as exc:
        raise ValueError(
            f"Cannot create a stratified train/validation split: {exc}"
        ) from exc
    train = train.reset_index(drop=True)
    evaluation = evaluation.reset_index(drop=True)
    train = _stratified_limit(
        train, config.get("train_size"), per_class, split_seed, "train_size"
    )
    evaluation = _stratified_limit(
        evaluation, config.get("eval_size"), 1, split_seed, "eval_size"
    )
    if (train["label"].value_counts().reindex(classes, fill_value=0) < per_class).any():
        raise ValueError("Training split has too few examples for initial_per_class.")
    train.attrs["selected_classes"] = selected_classes
    evaluation.attrs["selected_classes"] = selected_classes
    digest = hashlib.sha256()
    for frame in (train, evaluation):
        digest.update(
            pd.util.hash_pandas_object(frame["sample_id"], index=False)
            .to_numpy()
            .tobytes()
        )
    return train, evaluation, digest.hexdigest()


def run_dataset_experiment(config, policies, show_progress=True):
    """Run matched policies across representations for one dataset experiment.

    Initialization uses the simulated oracle to draw exactly the configured
    number of examples per class. GloVe must already exist on disk; no model
    downloads are initiated. Results retain metadata, not feature matrices.
    """
    if not policies:
        raise ValueError("Provide at least one acquisition policy.")
    representations = config.get("representations", [{"method": "tfidf"}])
    seeds = tuple(config.get("seeds", (42, 43, 44, 45, 46)))
    if not representations or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide representations and unique nonempty seeds.")
    train, evaluation, split_id = prepare_dataset_experiment(config)
    labels = train["label"].to_numpy()
    eval_labels = evaluation["label"].to_numpy()
    ids = train["sample_id"].to_numpy()
    per_class = config.get("initial_per_class", 5)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    initial_count = len(groups) * per_class
    total_budget = config["total_budget"]
    if (
        isinstance(total_budget, bool)
        or not isinstance(total_budget, (int, np.integer))
        or total_budget < initial_count
    ):
        raise ValueError("total_budget must cover all initial per-class labels.")
    if total_budget > len(train):
        raise ValueError("total_budget exceeds the number of training candidates.")
    batch_size = config.get("batch_size", 20)
    initial_sets = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        initial_sets[seed] = np.concatenate(
            [rng.choice(group, per_class, replace=False) for group in groups]
        )
    results = []
    with tqdm(
        total=len(representations) * len(seeds) * len(policies),
        desc="Dataset experiment",
        unit="run",
        disable=not show_progress,
    ) as progress:
        for specification in representations:
            rep = dict(specification)
            classifier = rep.pop(
                "classifier",
                config.get(
                    "classifier", config.get("model_type", "logistic_regression")
                ),
            )
            method = rep.get("method", "tfidf")
            if method == "embeddings":
                model_name = rep.get("model_name") or "glove-wiki-gigaword-50"
                if model_name != "glove-wiki-gigaword-50":
                    raise ValueError(
                        "Experiment 4 supports the existing GloVe model only."
                    )
                models_dir = Path(
                    rep.get("models_dir", REPOSITORY_ROOT / "models")
                ).expanduser()
                if not (models_dir / model_name / f"{model_name}.gz").is_file():
                    raise FileNotFoundError(
                        f"Local GloVe vectors not found in {models_dir / model_name}; no download was attempted."
                    )
            elif method != "tfidf":
                raise ValueError(f"Unknown representation: {method!r}.")
            X, X_eval, metadata = build_representation(
                train["text"], evaluation["text"], **rep
            )
            signature = _data_signature(X, labels, X_eval, eval_labels, ids)
            for seed in seeds:
                initial = initial_sets[seed]
                pool = np.flatnonzero(~np.isin(np.arange(len(train)), initial))
                for name, policy in policies.items():
                    if show_progress:
                        progress.set_postfix(
                            representation=method, policy=name, seed=seed
                        )
                    result = run_active_learning(
                        X,
                        labels,
                        X_eval,
                        eval_labels,
                        initial,
                        pool,
                        dict(policy),
                        budget=None,
                        total_budget=total_budget,
                        batch_size=batch_size,
                        random_state=seed,
                        sample_ids=ids,
                        model_type=classifier,
                        policy_name=name,
                        dataset_id=config["dataset"],
                        representation=metadata,
                        initial_method="per_class_oracle",
                        data_signature=signature,
                    )
                    result["configuration"].update(
                        split_id=split_id,
                        split_seed=config.get("split_seed", 42),
                        initial_per_class=per_class,
                        train_size=len(train),
                        eval_size=len(evaluation),
                        final_accuracy=float(result["history"].iloc[-1]["accuracy"]),
                        final_macro_f1=float(result["history"].iloc[-1]["macro_f1"]),
                        selected_classes=train.attrs["selected_classes"],
                    )
                    results.append(result)
                    progress.update(1)
    return results
