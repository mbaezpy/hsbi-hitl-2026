"""Inspect acquisitions and compare active-learning experiments."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.integrate import trapezoid
from sklearn.metrics.pairwise import cosine_similarity


def inspect_acquired_samples(result, dataset, round_number: int, n: int = 10):
    """Return annotated messages and selection information for one round."""
    acquired = result["acquisitions"]
    if acquired.empty:
        return acquired.copy()
    rows = acquired[acquired["round"] == round_number].head(n)
    columns = [
        column for column in ("sample_id", "text", "label_name") if column in dataset
    ]
    return rows.merge(
        dataset[columns], on="sample_id", how="left", validate="many_to_one"
    )


def compute_batch_redundancy(X, selected_idx):
    """Mean pairwise cosine similarity over distinct selected examples."""
    indices = np.asarray(selected_idx, dtype=int)
    if len(indices) < 2:
        return np.nan
    similarities = cosine_similarity(X[indices])
    return float(similarities[np.triu_indices(len(indices), k=1)].mean())


def plot_policy_dynamics(result):
    """Plot pre-acquisition beta and the adaptive strategy selected each round."""
    if result["configuration"]["policy"]["type"] != "adaptive":
        raise ValueError("Policy dynamics require an adaptive experiment result.")
    decisions = result["policy_history"]
    if decisions.empty:
        raise ValueError("The adaptive run has no completed acquisition rounds.")
    threshold = result["configuration"]["policy"].get("threshold", 0.5)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(decisions["round"], decisions["beta_before"], color="gray", alpha=0.7)
    for strategy, group in decisions.groupby("strategy", sort=False):
        ax.scatter(group["round"], group["beta_before"], label=strategy, zorder=3)
    ax.axhline(
        threshold, color="black", linestyle="--", label=f"Threshold ({threshold:g})"
    )
    ax.set(
        xlabel="Acquisition round",
        ylabel="Beta before acquisition",
        title="Feedback-adaptive policy decisions",
        ylim=(0, 1),
    )
    ax.set_xticks(decisions["round"])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_learning_curves(
    results,
    metric: str = "macro_f1",
    x_axis="acquired_labels",
    aggregate=None,
    grid_step=10,
    ax=None,
):
    """Plot trajectories or condition means and sample SD on an interpolated grid.

    ``aggregate=True`` interpolates only between a run's observed endpoints.
    The contributing seeds can therefore change across the curve, especially
    after early stopping. Grid points are estimates, not new evaluations.
    The default preserves the original policy and total-label plots.
    """
    if metric not in {"accuracy", "macro_f1"}:
        raise ValueError("metric must be 'accuracy' or 'macro_f1'.")
    if x_axis not in {"acquired_labels", "training_size"}:
        raise ValueError("x_axis must be 'acquired_labels' or 'training_size'.")
    if aggregate not in (None, True, False):
        raise ValueError("aggregate must be True, False, or None.")
    if ax is not None and not aggregate:
        raise ValueError("An existing axis is supported for aggregated curves only.")
    results = [results] if isinstance(results, dict) else list(results)
    if aggregate:
        if not np.isscalar(grid_step) or not np.isfinite(grid_step) or grid_step <= 0:
            raise ValueError("grid_step must be a positive finite number.")
        groups = {}
        seen = set()
        for result in results:
            cfg = result["configuration"]
            name = cfg.get("condition", cfg.get("policy_name"))
            seed = cfg.get("random_state")
            if name is None:
                raise ValueError("Each result must have a condition or policy name.")
            if seed is not None and (name, seed) in seen:
                raise ValueError("Each condition and seed must identify one run.")
            seen.add((name, seed))
            history = result["history"]
            if history.empty:
                continue
            x = history[x_axis].to_numpy(dtype=float)
            y = history[metric].to_numpy(dtype=float)
            order = np.argsort(x)
            x, y = x[order], y[order]
            if (
                not np.isfinite(x).all()
                or not np.isfinite(y).all()
                or np.any(np.diff(x) <= 0)
            ):
                raise ValueError(
                    "Evaluation checkpoints must be finite and unique within each run."
                )
            groups.setdefault(name, []).append((x, y))
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            fig = ax.figure
        for name, runs in groups.items():
            low = min(x[0] for x, _ in runs)
            high = max(x[-1] for x, _ in runs)
            grid = np.arange(
                np.ceil(low / grid_step) * grid_step, high + grid_step, grid_step
            )
            grid = grid[grid <= high]
            # Include endpoints so short or off-grid runs remain visible.
            grid = np.unique(
                np.concatenate([grid, [v for x, _ in runs for v in (x[0], x[-1])]])
            )
            # Insert a missing point in uncovered gaps, even when the grid is
            # coarser than the gap, so plotted lines cannot bridge those gaps.
            midpoints = (grid[:-1] + grid[1:]) / 2
            gaps = [
                point
                for point in midpoints
                if not any(x[0] <= point <= x[-1] for x, _ in runs)
            ]
            if gaps:
                grid = np.sort(np.concatenate([grid, gaps]))
            values = np.vstack(
                [np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in runs]
            )
            summary = pd.DataFrame(values).agg(["mean", "std"])
            mean = summary.loc["mean"].to_numpy()
            std = summary.loc["std"].to_numpy()
            (line,) = ax.plot(grid, mean, label=name)
            ax.fill_between(
                grid,
                mean - std,
                mean + std,
                where=np.isfinite(std),
                color=line.get_color(),
                alpha=0.15,
            )
        ax.set(
            xlabel="Total labeled examples"
            if x_axis == "training_size"
            else "Cumulative acquired labels",
            ylabel=metric.replace("_", " ").title(),
            title="Condition means ± SD (linearly interpolated; no extrapolation)",
        )
        ax.grid(alpha=0.3)
        if groups:
            ax.legend()
        fig.tight_layout()
        return fig, ax
    if aggregate is False or x_axis == "training_size":
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {}
        for result in results:
            cfg = result["configuration"]
            name = cfg.get("condition", cfg.get("policy_name"))
            history = result["history"]
            if history.empty:
                continue
            color = colors.get(name)
            (line,) = ax.plot(
                history[x_axis],
                history[metric],
                marker="o",
                alpha=0.6,
                color=color,
                label=name if name not in colors else "_nolegend_",
            )
            colors[name] = line.get_color()
        ax.set(
            xlabel="Total labeled examples"
            if x_axis == "training_size"
            else "Cumulative acquired labels",
            ylabel=metric.replace("_", " ").title(),
            title="Active learning: observed checkpoints by run",
        )
        ax.grid(alpha=0.3)
        if colors:
            ax.legend()
        fig.tight_layout()
        return fig, ax
    history = pd.concat([result["history"] for result in results], ignore_index=True)
    summary = (
        history.groupby(["policy", "acquired_labels"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    for policy, rows in summary.groupby("policy", sort=False):
        x = rows["acquired_labels"].to_numpy()
        mean = rows["mean"].to_numpy()
        (line,) = ax.plot(x, mean, marker="o", label=policy)
        std = rows["std"].to_numpy()
        if np.isfinite(std).any():
            ax.fill_between(
                x, mean - std, mean + std, color=line.get_color(), alpha=0.15
            )
    ax.set(
        xlabel="Cumulative acquired labels",
        ylabel=metric.replace("_", " ").title(),
        title="Active learning: acquisition policies",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def summarize_conditions(results):
    """Return actual per-run outcomes and a condition-level mean/SD summary."""
    results = [results] if isinstance(results, dict) else list(results)
    rows = []
    for result in results:
        cfg = result["configuration"]
        history = result["history"]
        final = history.iloc[-1] if not history.empty else {}
        rows.append(
            {
                "condition": cfg["condition"],
                "seed": cfg["random_state"],
                "initial_labels": cfg["initial_labeled_size"],
                "acquired_labels": cfg["actual_acquired_labels"],
                "total_labels": cfg["actual_labels_consumed"],
                "accuracy": final.get("accuracy", np.nan),
                "macro_f1": final.get("macro_f1", np.nan),
                "labels_saved": cfg["labels_saved"],
                "stop_reason": cfg["stop_reason"],
                "stop_reasons": tuple(cfg["stop_reasons"]),
                "initialization_complete": cfg["initialization_complete"],
                "completed_rounds": cfg["completed_rounds"],
            }
        )
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise ValueError("Provide at least one condition result.")
    grouped = runs.groupby("condition", sort=False)
    metrics = [
        "initial_labels",
        "total_labels",
        "accuracy",
        "macro_f1",
        "labels_saved",
        "completed_rounds",
    ]
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    summary["n_runs"] = grouped.size()
    summary["initialization_complete_count"] = grouped["initialization_complete"].sum()
    summary["stop_reason_frequencies"] = grouped["stop_reasons"].agg(
        lambda reasons: (
            pd.Series([reason for item in reasons for reason in item])
            .value_counts()
            .to_dict()
        )
    )
    return runs, summary.reset_index()


def plot_initialization_cost(results):
    """Plot mean initialization labels with standard deviation across seeds."""
    _, summary = summarize_conditions(results)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        summary["condition"],
        summary["initial_labels_mean"],
        yerr=summary["initial_labels_std"].fillna(0),
        capsize=3,
    )
    ax.set(ylabel="Labels used for initialization", title="Initialization cost")
    fig.tight_layout()
    return fig, ax


def plot_stopping_outcomes(results):
    """Show labels consumed per run, colored by the actual primary stop reason."""
    results = [results] if isinstance(results, dict) else list(results)
    runs, _ = summarize_conditions(results)
    runs = pd.concat(
        [
            group.sort_values("seed")
            for _, group in runs.groupby("condition", sort=False)
        ],
        ignore_index=True,
    )
    colors = {
        "max_budget": "#4c78a8",
        "target_performance": "#54a24b",
        "prediction_stability": "#f58518",
    }
    other_color = "#8b8b8b"
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(runs) + 1.5)))
    positions = np.arange(len(runs))
    ax.barh(
        positions,
        runs["total_labels"],
        color=[colors.get(reason, other_color) for reason in runs["stop_reason"]],
    )
    for pos, row in runs.iterrows():
        secondary = [
            reason for reason in row["stop_reasons"] if reason != row["stop_reason"]
        ]
        annotations = []
        if row["stop_reason"] not in colors:
            annotations.append(row["stop_reason"])
        if secondary:
            annotations.append("also: " + ", ".join(secondary))
        if annotations:
            ax.annotate(
                "; ".join(annotations),
                (row["total_labels"], pos),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
    ax.set_yticks(
        positions, [f"{row.condition} — seed {row.seed}" for row in runs.itertuples()]
    )
    ax.invert_yaxis()
    budgets = sorted({r["configuration"]["total_budget"] for r in results})
    for budget in budgets:
        ax.axvline(
            budget,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Total budget ({budget})",
        )
    right_edge = max(max(budgets), runs["total_labels"].max())
    ax.set_xlim(0, max(1, right_edge * (1.6 if ax.texts else 1.08)))
    reason_handles = [
        Patch(color=color, label=reason) for reason, color in colors.items()
    ]
    reason_handles.append(Patch(color=other_color, label="Other (shown beside bar)"))
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=reason_handles + handles, loc="upper left", bbox_to_anchor=(1.01, 1)
    )
    ax.set(
        xlabel="Total labels consumed",
        ylabel="Condition and seed",
        title="Stopping outcomes by run",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_stopping_diagnostics(results, seed=42):
    """Plot recorded threshold diagnostics for each applicable condition in a seed."""
    results = [results] if isinstance(results, dict) else list(results)
    selected = [
        result for result in results if result["configuration"]["random_state"] == seed
    ]
    if not selected:
        raise ValueError(f"No results for seed {seed}.")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for ax, key, title in zip(
        axes,
        ("target_performance", "prediction_stability"),
        ("Validation performance", "Prediction stability"),
    ):
        applicable = 0
        metrics = set()
        for result in selected:
            cfg = result["configuration"]
            criterion = (cfg.get("stopping") or {}).get(key)
            if criterion is None:
                continue
            applicable += 1
            name = cfg.get("condition", cfg.get("policy_name", "run"))
            metric = (
                criterion["metric"] if key == "target_performance" else "stability_mean"
            )
            threshold = criterion.get("threshold", 0.99)
            metrics.add(metric)
            history = result["history"]
            if metric in history and "training_size" in history:
                observed = (
                    history[["training_size", metric]]
                    .dropna()
                    .sort_values("training_size")
                )
            else:
                observed = pd.DataFrame(columns=["training_size", metric])
            reached = observed[observed[metric] >= threshold]
            status = "reached" if not reached.empty else "not reached"
            (line,) = ax.plot(
                observed["training_size"],
                observed[metric],
                marker="o",
                label=f"{name} ({metric}; {status})",
            )
            ax.axhline(
                threshold,
                color=line.get_color(),
                linestyle="--",
                alpha=0.7,
                label=f"{name} threshold ({threshold:g})",
            )
            if not reached.empty:
                first = reached.iloc[0]
                ax.scatter(
                    first["training_size"],
                    first[metric],
                    color=line.get_color(),
                    marker="*",
                    s=120,
                    zorder=4,
                    label=f"{name}: first reached",
                )
        if not applicable:
            ax.text(
                0.5,
                0.5,
                "No condition configured",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            ax.legend(fontsize="small", loc="best")
        ax.set(
            xlabel="Total labeled examples",
            title=f"{title} — seed {seed}",
            ylabel=(
                "Rolling mean Cohen's κ"
                if key == "prediction_stability"
                else ", ".join(sorted(metrics))
                if metrics
                else "Validation performance"
            ),
        )
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes


def summarize_stopping(results):
    """Summarize consumption and final performance without unrelated metrics."""
    results = [results] if isinstance(results, dict) else list(results)
    _, summary = summarize_conditions(results)
    early = (
        pd.DataFrame(
            {
                "condition": r["configuration"]["condition"],
                "early": r["configuration"]["finished_early"],
            }
            for r in results
        )
        .groupby("condition", sort=False)["early"]
        .sum()
    )
    summary["early_stops"] = summary["condition"].map(early).astype(int)
    return summary[
        [
            "condition",
            "total_labels_mean",
            "total_labels_std",
            "macro_f1_mean",
            "macro_f1_std",
            "labels_saved_mean",
            "early_stops",
            "stop_reason_frequencies",
        ]
    ]


def _representation_name(config):
    representation = config["representation"]
    if representation.get("method") == "embeddings":
        return representation.get("model_name", "glove-wiki-gigaword-50")
    method = representation.get("method", "unspecified")
    parameters = {
        key: value
        for key, value in representation.items()
        if key not in {"method", "model_name"}
    }
    if parameters:
        return f"{method} ({', '.join(f'{key}={value}' for key, value in sorted(parameters.items()))})"
    return method


def plot_setting_learning_curves(results, ax=None):
    """Plot policy/representation means within one dataset, on total labels."""
    results = [results] if isinstance(results, dict) else list(results)
    if not results or len({r["configuration"]["dataset_id"] for r in results}) != 1:
        raise ValueError("Provide results from exactly one dataset.")
    labeled = []
    classifiers = {}
    for result in results:
        cfg = result["configuration"]
        classifiers.setdefault(_representation_name(cfg), set()).add(cfg["model_type"])
    for result in results:
        cfg = result["configuration"]
        representation = _representation_name(cfg)
        name = f"{representation} + {cfg['policy_name']}"
        if len(classifiers[representation]) > 1:
            name += f" ({cfg['model_type']})"
        labeled.append({**result, "configuration": {**cfg, "condition": name}})
    fig, ax = plot_learning_curves(
        labeled, x_axis="training_size", aggregate=True, ax=ax
    )
    ax.set_title(
        f"{results[0]['configuration']['dataset_id']}: mean macro-F1 (interpolated)"
    )
    return fig, ax


def plot_dataset_learning_curves(*datasets):
    """Show separate within-dataset policy/representation curves side by side."""
    if not datasets:
        raise ValueError("Provide at least one dataset's results.")
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(8 * len(datasets), 5), squeeze=False
    )
    for ax, results in zip(axes[0], datasets):
        plot_setting_learning_curves(results, ax=ax)
    fig.tight_layout()
    return fig, axes[0]


def compare_policy_advantages(results, baseline="Random", policy="Uncertainty"):
    """Pair final macro-F1 at the requested total budget within each setting."""
    results = [results] if isinstance(results, dict) else list(results)
    settings = {}
    for result in results:
        cfg = result["configuration"]
        if cfg["policy_name"] not in {baseline, policy}:
            continue
        key = (cfg["dataset_id"], _representation_name(cfg), cfg["model_type"])
        run_key = (cfg["random_state"], cfg["policy_name"])
        group = settings.setdefault(key, {})
        if run_key in group:
            raise ValueError(f"Duplicate policy/seed in setting {key}: {run_key}.")
        group[run_key] = result
    rows = []
    for (dataset, representation, classifier), runs in settings.items():
        seeds = {seed for seed, name in runs if name == policy} & {
            seed for seed, name in runs if name == baseline
        }
        for seed in sorted(seeds):
            reference, selected = runs[(seed, baseline)], runs[(seed, policy)]
            a, b = reference["configuration"], selected["configuration"]
            if (
                not _compatible(reference, selected)
                or a["total_budget"] != b["total_budget"]
                or a.get("split_id") != b.get("split_id")
            ):
                raise ValueError(
                    f"Unmatched configuration or initialization for {dataset}, {representation}, seed {seed}."
                )
            budget = a["total_budget"]
            left = reference["history"]
            right = selected["history"]
            left = left[left["training_size"] == budget]
            right = right[right["training_size"] == budget]
            if len(left) != 1 or len(right) != 1:
                raise ValueError(
                    f"Both runs must have an evaluation at total budget {budget}."
                )
            rows.append(
                {
                    "dataset": dataset,
                    "representation": representation,
                    "classifier": classifier,
                    "seed": seed,
                    "total_budget": budget,
                    "difference": right.iloc[0]["macro_f1"] - left.iloc[0]["macro_f1"],
                }
            )
    if not rows:
        raise ValueError("No matched policy pairs found.")
    pairs = pd.DataFrame(rows)
    summary = (
        pairs.groupby(["dataset", "representation", "classifier"], sort=False)[
            "difference"
        ]
        .agg(mean_difference="mean", std_difference="std", n_pairs="count")
        .reset_index()
    )
    return pairs, summary


def plot_policy_advantages(results, baseline="Random", policy="Uncertainty"):
    """Plot paired macro-F1 differences, with one bar per dataset setting."""
    _, summary = compare_policy_advantages(results, baseline=baseline, policy=policy)
    fig, ax = plt.subplots(figsize=(9, max(4, len(summary) * 0.5 + 1)))
    positions = np.arange(len(summary))
    ax.barh(positions, summary["mean_difference"])
    repeated = summary["n_pairs"].to_numpy() > 1
    if repeated.any():
        ax.errorbar(
            summary.loc[repeated, "mean_difference"],
            positions[repeated],
            xerr=summary.loc[repeated, "std_difference"],
            fmt="none",
            color="black",
            capsize=3,
        )
    ax.set_yticks(
        positions,
        [
            f"{row.dataset} / {row.representation} / {row.classifier}"
            for row in summary.itertuples()
        ],
    )
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1)
    ax.set(
        xlabel=f"Paired macro-F1 difference ({policy} − {baseline})",
        title="Policy advantage by dataset and representation",
    )
    fig.tight_layout()
    return fig, ax


def summarize_experiments(results):
    """Return one row per policy and seed, including final metrics and times."""
    results = [results] if isinstance(results, dict) else list(results)
    rows = []
    for result in results:
        final = result["history"].iloc[-1]
        rows.append(
            {
                "policy": final["policy"],
                "seed": final["seed"],
                "initial_labeled_size": final["initial_labeled_size"],
                "budget_used": final["acquired_labels"],
                "training_size": final["training_size"],
                "accuracy": final["accuracy"],
                "macro_f1": final["macro_f1"],
                "training_seconds": result["history"]["training_seconds"].sum(),
                "acquisition_seconds": result["history"]["acquisition_seconds"].sum(),
                "completed_rounds": result["configuration"].get(
                    "completed_rounds", final["round"]
                ),
                "target_budget": result["configuration"]["budget"],
                "accuracy_alc": area_under_learning_curve(result, "accuracy"),
                "macro_f1_alc": area_under_learning_curve(result, "macro_f1"),
            }
        )
    return pd.DataFrame(rows)


def area_under_learning_curve(result, metric="macro_f1", budget=None, normalize=True):
    """Integrate the learning curve over acquired labels, including zero.

    By default divide by the requested budget. Return NaN when a run ends
    before that budget; never extrapolate. The target must be a logged round.
    """
    if metric not in {"accuracy", "macro_f1"}:
        raise ValueError("metric must be 'accuracy' or 'macro_f1'.")
    history = result["history"].sort_values("acquired_labels")
    target = result["configuration"]["budget"] if budget is None else budget
    x = history["acquired_labels"].to_numpy()
    if not len(x) or x[0] != 0 or np.any(np.diff(x) <= 0) or target < 0:
        raise ValueError(
            "History must start at zero and have unique increasing budgets."
        )
    if x[-1] < target or target == 0:
        return np.nan
    if target not in x:
        raise ValueError(f"Budget {target} is not a recorded evaluation point.")
    included = x <= target
    area = float(trapezoid(history.loc[included, metric].to_numpy(), x=x[included]))
    return area / target if normalize else area


def summarize_policies(results):
    """Aggregate final metrics and normalized ALC across seeds by policy.

    Standard deviations are NaN for one run. ALC is NaN for runs ending before
    the configured target; ``alc_runs`` shows how many reached that target.
    All results must have the same configured annotation budget.
    """
    results = [results] if isinstance(results, dict) else list(results)
    if results and any(
        not _same_conditions(results[0], result) for result in results[1:]
    ):
        raise ValueError("Policy summary requires equivalent experimental conditions.")
    runs = summarize_experiments(results)
    if runs.empty:
        raise ValueError("Provide at least one experiment result.")
    if runs.duplicated(["policy", "seed"]).any():
        raise ValueError("Each policy and seed must identify one run.")
    if runs["target_budget"].nunique() != 1:
        raise ValueError("Policy ALC comparisons require the same target budget.")
    metrics = [
        "accuracy",
        "macro_f1",
        "budget_used",
        "training_seconds",
        "acquisition_seconds",
        "accuracy_alc",
        "macro_f1_alc",
    ]
    grouped = runs.groupby("policy", sort=False)
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary["n_runs"] = grouped.size()
    summary["alc_runs"] = grouped["macro_f1_alc"].count()
    return summary.reset_index()


def _same_conditions(left, right):
    """Check shared settings while allowing different seeds and policies."""
    keys = (
        "dataset_id",
        "representation",
        "model_type",
        "initial_method",
        "initial_labeled_size",
        "budget",
        "batch_size",
        "data_signature",
    )
    a, b = left["configuration"], right["configuration"]
    return all(key in a and key in b and a[key] == b[key] for key in keys)


def _compatible(left, right):
    """Check matched settings, labeled examples, and acquisition pool."""
    return (
        _same_conditions(left, right)
        and np.array_equal(left["initial_labeled_ids"], right["initial_labeled_ids"])
        and np.array_equal(left["initial_pool_ids"], right["initial_pool_ids"])
    )


def compare_to_baseline(results, baseline="Random", metric="macro_f1"):
    """Return per-seed final differences and mean/SD for matched conditions.

    Seeds without a baseline are omitted; incompatible matched runs raise an
    error. Final acquired budgets must match; no performance is extrapolated.
    """
    if metric not in {"accuracy", "macro_f1"}:
        raise ValueError("metric must be 'accuracy' or 'macro_f1'.")
    results = [results] if isinstance(results, dict) else list(results)
    lookup = {}
    for result in results:
        cfg = result["configuration"]
        key = (cfg["policy_name"], cfg["random_state"])
        if key in lookup:
            raise ValueError(f"Duplicate policy/seed result: {key}")
        lookup[key] = result
    if not any(name == baseline for name, _ in lookup):
        raise ValueError(f"No baseline results named {baseline!r}.")
    pairs = []
    for (name, seed), result in lookup.items():
        if name == baseline or (baseline, seed) not in lookup:
            continue
        reference = lookup[(baseline, seed)]
        if not _compatible(result, reference):
            raise ValueError(
                f"Incompatible conditions for {name!r} and {baseline!r}, seed {seed}."
            )
        final, base = result["history"].iloc[-1], reference["history"].iloc[-1]
        if final["acquired_labels"] != base["acquired_labels"]:
            raise ValueError(
                f"Unequal final budgets for {name!r} and {baseline!r}, seed {seed}."
            )
        pairs.append(
            {
                "policy": name,
                "baseline": baseline,
                "seed": seed,
                "budget_used": final["acquired_labels"],
                "metric": metric,
                "policy_value": final[metric],
                "baseline_value": base[metric],
                "difference": final[metric] - base[metric],
            }
        )
    per_seed = pd.DataFrame(
        pairs,
        columns=[
            "policy",
            "baseline",
            "seed",
            "budget_used",
            "metric",
            "policy_value",
            "baseline_value",
            "difference",
        ],
    )
    if per_seed.empty:
        return per_seed, pd.DataFrame(
            columns=["policy", "mean_difference", "std_difference", "n_pairs"]
        )
    summary = (
        per_seed.groupby("policy", sort=False)["difference"]
        .agg(mean_difference="mean", std_difference="std", n_pairs="count")
        .reset_index()
    )
    return per_seed, summary


def plot_final_performance(results, metric="macro_f1", budget=None, sort=True):
    """Plot final scores at an exact cumulative additional-label budget.

    One run has no error bar; repeated runs show sample standard deviation.
    Missing requested budgets raise an error rather than substituting a round.
    """
    if metric not in {"accuracy", "macro_f1"}:
        raise ValueError("metric must be 'accuracy' or 'macro_f1'.")
    results = [results] if isinstance(results, dict) else list(results)
    if not results:
        raise ValueError("Provide at least one experiment result.")
    if budget is None:
        final_budgets = {
            int(result["history"].iloc[-1]["acquired_labels"]) for result in results
        }
        if len(final_budgets) != 1:
            raise ValueError(
                "Runs have different final budgets; specify a common recorded budget."
            )
        budget = final_budgets.pop()
    rows = []
    for result in results:
        matching = result["history"][result["history"]["acquired_labels"] == budget]
        if len(matching) != 1:
            cfg = result["configuration"]
            raise ValueError(
                f"No unique result at budget {budget} for {cfg['policy_name']} seed {cfg['random_state']}."
            )
        rows.append(
            {
                "policy": result["configuration"]["policy_name"],
                "value": matching.iloc[0][metric],
            }
        )
    grouped = (
        pd.DataFrame(rows)
        .groupby("policy", sort=False)["value"]
        .agg(["mean", "std", "count"])
    )
    if sort:
        grouped = grouped.sort_values("mean", ascending=False, kind="stable")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(grouped))))
    positions = np.arange(len(grouped))
    ax.barh(positions, grouped["mean"].to_numpy())
    multi = grouped["count"].to_numpy() > 1
    if multi.any():
        ax.errorbar(
            grouped["mean"].to_numpy()[multi],
            positions[multi],
            xerr=grouped["std"].to_numpy()[multi],
            fmt="none",
            color="black",
            capsize=3,
        )
    ax.set_yticks(positions, grouped.index)
    ax.invert_yaxis()
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Final performance after {budget} acquired labels")
    fig.tight_layout()
    return fig, ax
