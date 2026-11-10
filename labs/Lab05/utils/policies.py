"""Compose acquisition policies using positions in the training matrix."""

import numpy as np

from . import acquisition as acq


def initialize_policy_state(policy, total_rounds):
    """Validate and initialize state separately for one experimental run."""
    kind = policy["type"]
    if kind == "scheduled":
        schedule = policy.get("schedule", [])
        supported = {
            "random",
            "uncertainty",
            "novelty",
            "diversity",
            "clustering",
            "representativeness",
        }
        if (
            not isinstance(schedule, (list, tuple))
            or not schedule
            or any(item not in supported for item in schedule)
        ):
            raise ValueError(
                "A schedule must contain supported individual strategy names."
            )
        if not policy.get("repeat", False) and len(schedule) < total_rounds:
            raise ValueError(
                f"Non-repeating schedule has {len(schedule)} entries but needs {total_rounds} acquisition rounds."
            )
        return {}
    if kind == "adaptive":
        if (
            policy.get("explore", "diversity") != "diversity"
            or policy.get("exploit", "uncertainty") != "uncertainty"
        ):
            raise ValueError(
                "Adaptive policy uses diversity for exploration and uncertainty for exploitation."
            )
        beta = float(policy.get("initial_beta", 0.5))
        epsilon = float(policy.get("epsilon", 0.1))
        decay = float(policy.get("lambda", 0.9))
        threshold = float(policy.get("threshold", 0.5))
        window = policy.get("window", 3)
        if (
            not 0 < epsilon < 0.5
            or not epsilon <= beta <= 1 - epsilon
            or not np.isfinite(decay)
            or decay <= 0
            or not 0 <= threshold <= 1
            or not isinstance(window, int)
            or isinstance(window, bool)
            or window < 1
        ):
            raise ValueError(
                "Invalid adaptive beta, epsilon, lambda, threshold, or window."
            )
        return {"beta": beta, "losses": []}
    return {}


def resolve_round_policy(policy, round_number, state):
    """Resolve a fixed schedule or threshold-based adaptive choice for a round."""
    kind = policy["type"]
    if kind == "scheduled":
        schedule = policy["schedule"]
        index = round_number - 1
        if policy.get("repeat", False):
            index %= len(schedule)
        elif index >= len(schedule):
            raise ValueError("Non-repeating schedule does not cover this round.")
        return {"type": schedule[index]}
    if kind == "adaptive":
        strategy = (
            policy.get("explore", "diversity")
            if state["beta"] <= float(policy.get("threshold", 0.5))
            else policy.get("exploit", "uncertainty")
        )
        return {"type": strategy, "method": "least_confidence"}
    return policy


def update_adaptive_state(policy, state, batch_loss):
    """Update beta using the current batch and preceding raw window losses.

    A window with fewer than two distinct losses uses normalized feedback 0.
    This course adaptation omits the paper's additional moving-average step.
    """
    if not np.isfinite(batch_loss) or batch_loss < 0:
        raise ValueError("Batch loss must be finite and nonnegative.")
    window = int(policy.get("window", 3))
    losses = (state["losses"] + [float(batch_loss)])[-window:]
    low, high = min(losses), max(losses)
    normalized = (batch_loss - low) / (high - low) if high > low else 0.0
    epsilon = float(policy.get("epsilon", 0.1))
    next_beta = float(
        np.clip(
            float(policy.get("lambda", 0.9)) * state["beta"] * np.exp(normalized),
            epsilon,
            1 - epsilon,
        )
    )
    return {"beta": next_beta, "losses": losses}, float(normalized)


def _budget_split_settings(policy):
    """Validate the two component strategies and their batch weights."""
    strategies = policy.get("strategies", ["uncertainty", "diversity"])
    supported = {
        "random",
        "uncertainty",
        "diversity",
        "novelty",
        "clustering",
        "representativeness",
    }
    if (
        not isinstance(strategies, (list, tuple))
        or len(strategies) != 2
        or any(
            not isinstance(item, str) or item not in supported for item in strategies
        )
    ):
        raise ValueError(
            "Budget split requires two supported strategies: "
            + ", ".join(sorted(supported))
            + "."
        )
    try:
        weights = np.asarray(policy.get("weights", [0.5, 0.5]), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Provide two finite nonnegative weights with positive total."
        ) from exc
    if (
        weights.shape != (2,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.any(weights > 0)
    ):
        raise ValueError("Provide two finite nonnegative weights with positive total.")
    return strategies, weights


def acquire_batch(
    policy: dict,
    model,
    X,
    labeled_idx,
    pool_idx,
    batch_size: int,
    rng,
    return_info=False,
):
    """Return selected training-matrix positions, optionally with per-item info.

    Ranking is descending priority with pool-order tie breaks. Rank aggregation
    minimizes the weighted sum of 1-based ranks. Budget-split counts use the
    largest-remainder rule (ties favor the first strategy).
    """
    pool = np.asarray(pool_idx, dtype=int)
    count = min(batch_size, len(pool))
    if batch_size < 0 or len(np.unique(pool)) != len(pool):
        raise ValueError("batch_size must be nonnegative and pool indices unique.")
    if np.intersect1d(pool, np.asarray(labeled_idx, dtype=int)).size:
        raise ValueError("The candidate pool and labeled set must be disjoint.")
    if policy["type"] == "budget_split":
        strategies, weights = _budget_split_settings(policy)
    if count == 0:
        result = np.array([], dtype=int)
        return (result, []) if return_info else result

    kind = policy["type"]
    method = policy.get("method", "least_confidence")
    random_state = int(rng.integers(0, 2**31 - 1))
    info = []

    if kind == "random":
        selected = acq.acquire_random(pool, count, rng)
        info = [{"strategy": kind} for _ in selected]
    elif kind in {"uncertainty", "novelty", "representativeness"}:
        if kind == "uncertainty":
            selected, scores = acq.acquire_uncertainty(
                model, X, pool, count, method, True
            )
        elif kind == "novelty":
            selected, scores = acq.acquire_novelty(X, labeled_idx, pool, count, True)
        else:
            selected, scores = acq.acquire_representativeness(
                X, labeled_idx, pool, count, True
            )
        score_map = dict(zip(pool, scores))
        info = [
            {"strategy": kind, "score": float(score_map[index])} for index in selected
        ]
    elif kind == "diversity":
        selected, scores = acq.acquire_diversity(
            X, labeled_idx, pool, count, return_scores=True
        )
        info = [{"strategy": kind, "score": score} for score in scores]
    elif kind == "clustering":
        selected, details = acq.acquire_clustering(X, pool, count, random_state, True)
        info = [
            {"strategy": kind, "cluster": cluster}
            for cluster in details["selected_clusters"]
        ]
    elif kind == "sequential":
        if (
            policy.get("first", "uncertainty") != "uncertainty"
            or policy.get("second", "clustering") != "clustering"
        ):
            raise ValueError(
                "Sequential policy supports uncertainty followed by clustering."
            )
        candidate_count = min(
            len(pool), max(count, int(policy.get("candidate_size", 100)))
        )
        candidates, scores = acq.acquire_uncertainty(
            model, X, pool, candidate_count, method, True
        )
        selected, details = acq.acquire_clustering(
            X, candidates, count, random_state, True
        )
        score_map = dict(zip(pool, scores))
        info = [
            {
                "strategy": "uncertainty_then_clustering",
                "uncertainty": float(score_map[index]),
                "candidate_size": candidate_count,
                "cluster": cluster,
            }
            for index, cluster in zip(selected, details["selected_clusters"])
        ]
    elif kind == "cluster_then_uncertainty":
        if "n_clusters" in policy:
            raise ValueError(
                "cluster_then_uncertainty uses one cluster per selected item; "
                "remove the redundant n_clusters parameter."
            )
        labels, _ = acq.cluster_candidates(X, pool, count, random_state)
        scores = acq.uncertainty_scores(model, X, pool, method)
        selected_list = []
        for cluster in range(count):
            members = np.flatnonzero(labels == cluster)
            if len(members):
                pos = members[np.argmax(scores[members])]
                selected_list.append(int(pool[pos]))
                info.append(
                    {
                        "strategy": kind,
                        "cluster": cluster,
                        "cluster_fallback": False,
                        "uncertainty": float(scores[pos]),
                    }
                )
        for pos in acq._ordered(scores, len(pool)):
            if len(selected_list) == count:
                break
            if pool[pos] not in selected_list:
                selected_list.append(int(pool[pos]))
                info.append(
                    {
                        "strategy": kind,
                        "cluster": None,
                        "cluster_fallback": True,
                        "uncertainty": float(scores[pos]),
                    }
                )
        selected = np.asarray(selected_list, dtype=int)
    elif kind == "rank_aggregation":
        if policy.get("strategies", ["uncertainty", "novelty"]) != [
            "uncertainty",
            "novelty",
        ]:
            raise ValueError("Rank aggregation supports uncertainty and novelty.")
        weights = np.asarray(policy.get("weights", [0.5, 0.5]), dtype=float)
        if len(weights) != 2 or np.any(weights < 0) or weights.sum() == 0:
            raise ValueError("Provide two nonnegative weights with positive total.")
        uncertainty = acq.uncertainty_scores(model, X, pool, method)
        novelty = acq.novelty_scores(X, labeled_idx, pool)
        ranks = []
        for scores in (uncertainty, novelty):
            order = np.argsort(-scores, kind="stable")
            rank = np.empty(len(pool), dtype=int)
            rank[order] = np.arange(1, len(pool) + 1)
            ranks.append(rank)
        aggregate = np.average(np.vstack(ranks), axis=0, weights=weights)
        positions = np.argsort(aggregate, kind="stable")[:count]
        selected = pool[positions]
        info = [
            {
                "strategy": kind,
                "aggregate_rank": float(aggregate[p]),
                "uncertainty": float(uncertainty[p]),
                "novelty": float(novelty[p]),
                "uncertainty_rank": int(ranks[0][p]),
                "novelty_rank": int(ranks[1][p]),
            }
            for p in positions
        ]
    elif kind == "budget_split":
        # Scale only extreme weights, keeping the existing rounding for ordinary inputs.
        if np.max(weights) > np.finfo(float).max / max(2, count):
            weights = weights / np.max(weights)
        ideal = count * weights / weights.sum()
        allocation = np.floor(ideal).astype(int)
        for index in np.argsort(-(ideal - allocation), kind="stable")[
            : count - allocation.sum()
        ]:
            allocation[index] += 1

        def select_component(strategy, candidates, quota, previous=None):
            if quota == 0:
                return np.array([], dtype=int), []
            if strategy == "random":
                chosen = acq.acquire_random(candidates, quota, rng)
                return chosen, [{"strategy": strategy} for _ in chosen]
            if strategy == "diversity":
                chosen, scores = acq.acquire_diversity(
                    X,
                    labeled_idx,
                    candidates,
                    quota,
                    selected_idx=previous,
                    return_scores=True,
                )
                return chosen, [
                    {"strategy": strategy, "score": score} for score in scores
                ]
            if strategy == "clustering":
                chosen, details = acq.acquire_clustering(
                    X, candidates, quota, random_state, True
                )
                return chosen, [
                    {"strategy": strategy, "cluster": cluster}
                    for cluster in details["selected_clusters"]
                ]
            if strategy == "uncertainty":
                chosen, scores = acq.acquire_uncertainty(
                    model, X, candidates, quota, method, True
                )
            elif strategy == "novelty":
                chosen, scores = acq.acquire_novelty(
                    X, labeled_idx, candidates, quota, True
                )
            else:
                chosen, scores = acq.acquire_representativeness(
                    X, labeled_idx, candidates, quota, True
                )
            score_map = dict(zip(candidates, scores))
            return chosen, [
                {"strategy": strategy, "score": float(score_map[index])}
                for index in chosen
            ]

        first, first_info = select_component(strategies[0], pool, int(allocation[0]))
        remaining = pool[~np.isin(pool, first)]
        second, second_info = select_component(
            strategies[1], remaining, int(allocation[1]), first
        )
        selected = np.concatenate([first, second])
        info = first_info + second_info
    else:
        raise ValueError(f"Unknown policy type: {kind!r}")
    if (
        len(selected) != count
        or len(np.unique(selected)) != count
        or not np.isin(selected, pool).all()
    ):
        raise RuntimeError("Policy returned invalid or duplicate candidates.")
    return (selected, info) if return_info else selected
