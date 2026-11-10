"""Individual acquisition functions. Indices are positions in the train matrix."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize


def _ordered(scores, count):
    """Higher scores first; ties retain candidate-pool order."""
    return np.argsort(-np.asarray(scores), kind="stable")[:count]


def acquire_random(pool_idx, batch_size, rng, return_scores=False):
    """Select uniformly without replacement from the candidate positions."""
    pool = np.asarray(pool_idx, dtype=int)
    selected = rng.choice(pool, size=min(batch_size, len(pool)), replace=False)
    return (selected, None) if return_scores else selected


def uncertainty_scores(model, X, pool_idx, method="least_confidence"):
    """One uncertainty score per pool position; higher means more uncertain."""
    probabilities = model.predict_proba(X[np.asarray(pool_idx, dtype=int)])
    if method == "least_confidence":
        return 1 - probabilities.max(axis=1)
    if method == "margin":
        ordered = np.sort(probabilities, axis=1)
        return 1 - (ordered[:, -1] - ordered[:, -2])
    if method == "entropy":
        return -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1)), axis=1)
    raise ValueError(f"Unknown uncertainty method: {method!r}")


def acquire_uncertainty(
    model, X, pool_idx, batch_size, method="least_confidence", return_scores=False
):
    """Select least-confidence, smallest-margin, or highest-entropy examples."""
    pool = np.asarray(pool_idx, dtype=int)
    if not len(pool) or batch_size <= 0:
        empty = np.array([], dtype=int)
        return (empty, np.array([])) if return_scores else empty
    scores = uncertainty_scores(model, X, pool, method)
    selected = pool[_ordered(scores, min(batch_size, len(pool)))]
    return (selected, scores) if return_scores else selected


def novelty_scores(X, labeled_idx, pool_idx):
    """Cosine distance to the nearest labeled example."""
    return 1 - cosine_similarity(
        X[np.asarray(pool_idx, dtype=int)], X[np.asarray(labeled_idx, dtype=int)]
    ).max(axis=1)


def acquire_novelty(X, labeled_idx, pool_idx, batch_size, return_scores=False):
    pool = np.asarray(pool_idx, dtype=int)
    if not len(pool) or batch_size <= 0:
        empty = np.array([], dtype=int)
        return (empty, np.array([])) if return_scores else empty
    scores = novelty_scores(X, labeled_idx, pool)
    selected = pool[_ordered(scores, min(batch_size, len(pool)))]
    return (selected, scores) if return_scores else selected


def acquire_diversity(
    X, labeled_idx, pool_idx, batch_size, selected_idx=None, return_scores=False
):
    """Farthest-first, initially anchored to labeled and preselected examples.

    With a preselected query batch (as in budget splitting), every new point
    remains distant from both that batch and the existing labeled set. With no
    preselected batch, retain Lab 04's standalone behavior: the first point is
    most novel relative to labeled data, then the batch spreads from that point.
    """
    pool = np.asarray(pool_idx, dtype=int)
    combined_reference = selected_idx is not None
    seeds = np.asarray(selected_idx if selected_idx is not None else [], dtype=int)
    count = min(batch_size, len(pool))
    if count <= 0:
        empty = np.array([], dtype=int)
        return (empty, []) if return_scores else empty
    if np.intersect1d(pool, seeds).size:
        raise ValueError("Existing batch selections must be excluded from pool_idx.")
    distances = (
        1
        - cosine_similarity(
            X[pool], X[np.concatenate([np.asarray(labeled_idx, dtype=int), seeds])]
        ).max(axis=1)
        if combined_reference
        else novelty_scores(X, labeled_idx, pool)
    )
    selected, selection_scores = [], []
    for _ in range(count):
        position = int(np.argmax(distances))
        selected.append(int(pool[position]))
        selection_scores.append(float(distances[position]))
        new_distances = (
            1 - cosine_similarity(X[pool], X[pool[position : position + 1]]).ravel()
        )
        distances = np.minimum(distances, new_distances)
        distances[np.isin(pool, selected)] = -np.inf
    result = np.asarray(selected, dtype=int)
    return (result, selection_scores) if return_scores else result


def cluster_candidates(X, pool_idx, n_clusters, random_state=42):
    """Return k-means membership and distances for both policy orders."""
    pool = np.asarray(pool_idx, dtype=int)
    kmeans = KMeans(
        n_clusters=min(n_clusters, len(pool)), random_state=random_state, n_init=5
    )
    labels = kmeans.fit_predict(X[pool])
    return labels, euclidean_distances(X[pool], kmeans.cluster_centers_)


def acquire_clustering(X, pool_idx, batch_size, random_state=42, return_info=False):
    """Select a centroid-nearest pool member from each nonempty cluster."""
    pool = np.asarray(pool_idx, dtype=int)
    count = min(batch_size, len(pool))
    if count <= 0:
        empty = np.array([], dtype=int)
        return (empty, {}) if return_info else empty
    labels, distances = cluster_candidates(X, pool, count, random_state)
    selected, clusters = [], []
    for cluster in range(count):
        members = np.flatnonzero(labels == cluster)
        if len(members):
            position = members[np.argmin(distances[members, cluster])]
            selected.append(int(pool[position]))
            clusters.append(cluster)
    # Identical points can leave clusters empty. Fill deterministically and
    # record that these extra selections are not cluster representatives.
    for candidate in pool:
        if len(selected) == count:
            break
        if candidate not in selected:
            selected.append(int(candidate))
            clusters.append(None)
    selected = np.asarray(selected, dtype=int)
    info = {
        "cluster_labels": labels,
        "distances": distances,
        "selected_clusters": clusters,
    }
    return (selected, info) if return_info else selected


def acquire_representativeness(
    X, labeled_idx, pool_idx, batch_size, return_scores=False
):
    """Select high pool support minus labeled support, using cosine similarity."""
    pool = np.asarray(pool_idx, dtype=int)
    if not len(pool) or batch_size <= 0:
        empty = np.array([], dtype=int)
        return (empty, np.array([])) if return_scores else empty
    # Normalize sparse or dense inputs. Subtract the actual diagonal so the
    # relative-support calculation also handles zero vectors correctly.
    pool_vectors = normalize(X[pool])
    labeled_vectors = normalize(X[np.asarray(labeled_idx, dtype=int)])
    pool_mean = np.asarray(pool_vectors.mean(axis=0)).ravel()
    labeled_mean = np.asarray(labeled_vectors.mean(axis=0)).ravel()
    total_support = np.asarray(pool_vectors @ pool_mean).ravel() * len(pool)
    self_similarity = (
        np.asarray(pool_vectors.multiply(pool_vectors).sum(axis=1)).ravel()
        if hasattr(pool_vectors, "multiply")
        else np.einsum("ij,ij->i", pool_vectors, pool_vectors)
    )
    pool_support = (
        (total_support - self_similarity) / (len(pool) - 1)
        if len(pool) > 1
        else np.zeros(1)
    )
    labeled_support = np.asarray(pool_vectors @ labeled_mean).ravel()
    scores = pool_support - labeled_support
    selected = pool[_ordered(scores, min(batch_size, len(pool)))]
    return (selected, scores) if return_scores else selected
