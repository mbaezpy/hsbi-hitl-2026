"""Inspect individual Label Studio text-classification annotations.

The parser targets a single Label Studio Choices control with one selected
choice. It does not aggregate annotations or assess annotator reliability.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_label_studio(path):
    """Read a raw Label Studio JSON list of task objects."""
    with Path(path).open(encoding="utf-8") as stream:
        tasks = json.load(stream)
    if not isinstance(tasks, list) or any(not isinstance(task, dict) for task in tasks):
        raise ValueError("Expected a raw Label Studio JSON list of task objects.")
    return tasks


def normalize_label_studio(
    tasks,
    id_field="example_id",
    text_field="text",
    control_name="label",
    annotator_id_field="id",
    include_annotator=True,
):
    """Return task and annotation tables; keep one row per annotation.

    Only a single-label Choices result from ``control_name`` is supported.
    Empty/missing results are incomplete; cancelled records remain visible but
    are excluded from submitted counts and reference comparisons.
    """
    task_rows, records = [], []
    for task in tasks:
        data = task.get("data") or {}
        if not isinstance(data, dict):
            raise TypeError("Each task's data must be an object.")
        identifier = data.get(id_field)
        text = data.get(text_field)
        if identifier is None or (
            isinstance(identifier, str) and not identifier.strip()
        ):
            identifier = None
            warnings.warn(
                f"Task {task.get('id')!r} has no {id_field!r}; it cannot be joined to the manifest.",
                stacklevel=2,
            )
        task_rows.append(
            {
                "task_id": task.get("id"),
                "example_id": str(identifier) if identifier is not None else None,
                "text": text,
                "draft_count": len(task.get("drafts") or []),
            }
        )
        annotations = task.get("annotations") or []
        if not isinstance(annotations, list):
            raise TypeError("Task annotations must be a list or null.")
        for annotation in annotations:
            if annotation is None:
                annotation = {}
            if not isinstance(annotation, dict):
                raise TypeError("Each annotation must be an object or null.")
            results = annotation.get("result") or []
            if not isinstance(results, list) or any(
                not isinstance(result, dict) for result in results
            ):
                raise ValueError("Annotation results must be a list of objects.")
            matching = [
                result for result in results if result.get("from_name") == control_name
            ]
            if len(matching) > 1:
                raise ValueError(
                    f"Multiple results for {control_name!r}; only one Choices result is supported."
                )
            choice = None
            if matching:
                result = matching[0]
                if result.get("type") != "choices" or not isinstance(
                    result.get("value", {}), dict
                ):
                    raise ValueError(
                        f"Control {control_name!r} must contain a single-label Choices result."
                    )
                choices = result.get("value", {}).get("choices", [])
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ValueError(
                        "Only one selected choice per annotation is supported."
                    )
                if choices and not isinstance(choices[0], str):
                    raise ValueError("Selected choices must be strings.")
                choice = choices[0] if choices and choices[0].strip() else None
            elif results:
                warnings.warn(
                    f"No result for control {control_name!r} in annotation {annotation.get('id')!r}.",
                    stacklevel=2,
                )
            cancelled = bool(annotation.get("was_cancelled", False))
            person = annotation.get("completed_by") if include_annotator else None
            if isinstance(person, dict):
                person = person.get(annotator_id_field)
            records.append(
                {
                    "task_id": task.get("id"),
                    "example_id": str(identifier) if identifier is not None else None,
                    "text": text,
                    "annotation_id": annotation.get("id"),
                    "annotator_id": person,
                    "label": choice,
                    "status": "cancelled"
                    if cancelled
                    else "submitted"
                    if choice
                    else "incomplete",
                    "was_cancelled": cancelled,
                    "lead_time": annotation.get("lead_time"),
                    "created_at": annotation.get("created_at"),
                    "updated_at": annotation.get("updated_at"),
                    "completed_at": annotation.get("completed_at")
                    or annotation.get("created_at"),
                }
            )
    task_columns = ["task_id", "example_id", "text", "draft_count"]
    columns = [
        "task_id",
        "example_id",
        "text",
        "annotation_id",
        "annotator_id",
        "label",
        "status",
        "was_cancelled",
        "lead_time",
        "created_at",
        "updated_at",
        "completed_at",
    ]
    return pd.DataFrame(task_rows, columns=task_columns), pd.DataFrame(
        records, columns=columns
    )


def load_reference(
    path, id_col="example_id", label_col="reference_label", text_col="text"
):
    """Load an original pilot manifest or reference file with one row per item."""
    frame = pd.read_csv(path, dtype={id_col: "string"})
    if id_col not in frame or label_col not in frame:
        raise ValueError(f"Reference file must contain {id_col!r} and {label_col!r}.")
    if (
        frame[id_col].isna().any()
        or frame[id_col].str.strip().eq("").any()
        or frame[id_col].duplicated().any()
    ):
        raise ValueError("Reference IDs must be nonempty and unique.")
    result = frame.rename(
        columns={id_col: "example_id", label_col: "reference_label"}
    ).copy()
    if text_col in frame and text_col not in (id_col, label_col):
        result = result.rename(columns={text_col: "reference_text"})
    return result


def flag_timing(annotations, threshold=None):
    """Keep all timing values, marking invalid values and optional slow cases."""
    if threshold is not None and (not np.isfinite(threshold) or threshold <= 0):
        raise ValueError(
            "Timing threshold must be a positive number of seconds or None."
        )
    result = annotations.copy()
    result["lead_time_raw"] = result["lead_time"]
    result["lead_time"] = pd.to_numeric(result["lead_time"], errors="coerce")
    valid = (
        result["lead_time"].notna()
        & np.isfinite(result["lead_time"])
        & result["lead_time"].ge(0)
    )
    result["time_issue"] = np.where(
        result["lead_time_raw"].isna(), "missing", np.where(valid, "", "invalid")
    )
    result["time_outlier"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    if threshold is not None:
        result.loc[valid, "time_outlier"] = result.loc[valid, "lead_time"] > threshold
    return result


def add_annotation_sequence(annotations):
    """Number timestamped submitted annotations independently for each worker.

    Rows without a worker or a valid completion timestamp remain unnumbered.
    Ties retain export order; missing duration does not remove a sequence step.
    """
    result = annotations.copy()
    result["completed_at"] = pd.to_datetime(
        result["completed_at"], errors="coerce", utc=True
    )
    result["sequence"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    valid = (
        result["status"].eq("submitted")
        & result["annotator_id"].notna()
        & result["completed_at"].notna()
    )
    # Use positional indices so duplicate DataFrame indices cannot mix workers.
    positions = pd.DataFrame(
        {
            "position": np.arange(len(result)),
            "worker": result["annotator_id"].to_numpy(),
            "completed_at": result["completed_at"].to_numpy(),
            "valid": valid.to_numpy(),
        }
    )
    for _, group in positions[positions["valid"]].groupby("worker", sort=False):
        ordered = group.sort_values(["completed_at", "position"], kind="stable")
        result.iloc[
            ordered["position"].to_numpy(), result.columns.get_loc("sequence")
        ] = np.arange(1, len(ordered) + 1)
    return result


def load_pilot(
    annotations_file,
    reference_file=None,
    label_mapping=None,
    task_id_field="example_id",
    text_field="text",
    reference_label_col="reference_label",
    control_name="label",
    reference_id_col=None,
    reference_text_col=None,
    annotator_id_field="id",
    include_annotator=True,
):
    """Load one row per manifest item and one per raw annotation.

    ``reference_file`` is the full task manifest and may omit reference labels.
    Without it, only exported items are known and a warning makes this limit
    explicit. Label mapping is applied to both sides for comparison. Label
    Studio's ``completed_at`` is used when present; otherwise ``created_at``
    is the recorded submission timestamp. ``updated_at`` is not a substitute.
    """
    raw_tasks = read_label_studio(annotations_file)
    results = [result for task in raw_tasks for annotation in (task.get("annotations") or [])
               if isinstance(annotation, dict) for result in (annotation.get("result") or [])
               if isinstance(result, dict)]
    if results and not any(result.get("from_name") == control_name for result in results):
        raise ValueError(
            f"No results for classification control {control_name!r}; "
            "check the control name or use load_ie_pilot for extraction exports."
        )
    tasks, annotations = normalize_label_studio(
        raw_tasks,
        id_field=task_id_field,
        text_field=text_field,
        control_name=control_name,
        annotator_id_field=annotator_id_field,
        include_annotator=include_annotator,
    )
    task_rows = tasks.dropna(subset=["example_id"])
    if task_rows["example_id"].duplicated().any():
        warnings.warn(
            "Multiple exported tasks share an example ID; annotations remain separate.",
            stacklevel=2,
        )
    task_rows = task_rows.drop_duplicates("example_id")
    if reference_file is None:
        warnings.warn(
            "No task manifest supplied; coverage describes exported items only and may be incomplete.",
            stacklevel=2,
        )
        items = task_rows[["example_id", "text"]].copy()
        items["reference_label"] = pd.NA
    else:
        reference_id_col = reference_id_col or task_id_field
        reference_text_col = reference_text_col or text_field
        frame = pd.read_csv(reference_file, dtype={reference_id_col: "string"})
        if reference_id_col not in frame:
            raise ValueError(f"Reference manifest must contain {reference_id_col!r}.")
        ids = frame[reference_id_col]
        if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
            raise ValueError("Reference IDs must be nonempty and unique.")
        items = pd.DataFrame({"example_id": ids.astype(str)})
        items["text"] = (
            frame[reference_text_col].astype("string")
            if reference_text_col in frame
            else pd.NA
        )
        if reference_label_col is not None and reference_label_col in frame:
            items["reference_label"] = frame[reference_label_col]
        else:
            if reference_label_col is not None:
                warnings.warn(
                    f"Reference label column {reference_label_col!r} is absent; comparisons are unavailable.",
                    stacklevel=2,
                )
            items["reference_label"] = pd.NA
    known_ids = set(items["example_id"])
    exported_ids = set(task_rows["example_id"])
    outside = exported_ids - known_ids
    if outside:
        warnings.warn(
            f"{len(outside)} exported item(s) are absent from the supplied manifest; excluded from coverage.",
            stacklevel=2,
        )
    text_by_id = task_rows.set_index("example_id")["text"]
    items["text"] = (
        items["text"]
        .astype("string")
        .fillna(items["example_id"].map(text_by_id).astype("string"))
    )
    items["task_exported"] = items["example_id"].isin(exported_ids)
    annotations = flag_timing(annotations)
    annotations = annotations.merge(
        items[["example_id", "text", "reference_label"]],
        on="example_id",
        how="left",
        suffixes=("", "_item"),
        validate="many_to_one",
    )
    annotations["in_manifest"] = annotations["example_id"].isin(known_ids)
    annotations["text"] = (
        annotations["text"]
        .astype("string")
        .fillna(annotations.pop("text_item").astype("string"))
    )
    mapping = label_mapping or {}
    annotations["reference_normalized"] = annotations["reference_label"].map(
        lambda value: mapping.get(value, value) if pd.notna(value) else value
    )
    annotations["label_normalized"] = annotations["label"].map(
        lambda value: mapping.get(value, value) if pd.notna(value) else value
    )
    comparable = (
        annotations["status"].eq("submitted")
        & annotations["reference_normalized"].notna()
        & annotations["label_normalized"].notna()
    )
    annotations["reference_match"] = pd.Series(
        pd.NA, index=annotations.index, dtype="boolean"
    )
    annotations.loc[comparable, "reference_match"] = annotations.loc[
        comparable, "reference_normalized"
    ].eq(annotations.loc[comparable, "label_normalized"])
    submitted = annotations[
        annotations["status"].eq("submitted") & annotations["in_manifest"]
    ]
    counts = submitted.groupby("example_id").size()
    items["annotation_count"] = items["example_id"].map(counts).fillna(0).astype(int)
    items = items.reset_index(drop=True)
    items.attrs["manifest_supplied"] = reference_file is not None
    return items, add_annotation_sequence(annotations)


def pilot_summary(items, annotations, num_ratings_per_example):
    """Return a compact pilot overview; only submitted choices count for coverage."""
    if (
        isinstance(num_ratings_per_example, bool)
        or not isinstance(num_ratings_per_example, (int, np.integer))
        or num_ratings_per_example < 1
    ):
        raise ValueError("num_ratings_per_example must be a positive integer.")
    submitted = annotations[annotations["status"].eq("submitted")]
    item_label = (
        "Examples in manifest"
        if items.attrs.get("manifest_supplied", True)
        else "Examples in export (manifest unavailable)"
    )
    summary = {
        item_label: len(items),
        "Submitted annotations": len(submitted),
        "Annotators with known ID": submitted["annotator_id"].nunique()
        if submitted["annotator_id"].notna().any()
        else pd.NA,
        "Examples with required annotations": int(
            items["annotation_count"].ge(num_ratings_per_example).sum()
        ),
    }
    if "in_manifest" in submitted and (~submitted["in_manifest"]).any():
        summary["Submitted annotations outside manifest"] = int(
            (~submitted["in_manifest"]).sum()
        )
    if "task_exported" in items and (~items["task_exported"]).any():
        summary["Examples absent from export (coverage unknown)"] = int(
            (~items["task_exported"]).sum()
        )
    return pd.Series(summary, name="count")


def plot_annotation_coverage(items, num_ratings_per_example):
    """Plot submitted annotation counts over the supplied manifest items."""
    if (
        isinstance(num_ratings_per_example, bool)
        or not isinstance(num_ratings_per_example, (int, np.integer))
        or num_ratings_per_example < 1
    ):
        raise ValueError("num_ratings_per_example must be a positive integer.")
    maximum = max(
        num_ratings_per_example,
        int(items["annotation_count"].max()) if len(items) else 0,
    )
    counts = (
        items["annotation_count"]
        .value_counts()
        .reindex(range(maximum + 1), fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color="#3d7770")
    ax.axvline(
        num_ratings_per_example - 0.5,
        color="#53616b",
        linestyle="--",
        linewidth=1,
        label=f"Required: {num_ratings_per_example}",
    )
    ax.set(
        xlabel="Submitted annotations observed per example",
        ylabel="Examples",
        title="Observed annotation coverage",
    )
    if "task_exported" in items and (~items["task_exported"]).any():
        ax.text(
            0.99,
            0.98,
            "Items absent from export: coverage unknown",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize="small",
        )
    ax.set_xticks(counts.index)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_annotator_progress(annotations, show_reference_matches=False):
    """Plot recorded lead times by worker sequence, without inferring learning.

    Optionally show individual reference matches; this is not an accuracy or
    agreement metric. Missing timestamps, IDs, and times are not invented.
    """
    sequenced = add_annotation_sequence(annotations)
    submitted = sequenced[sequenced["status"].eq("submitted")]
    data = sequenced[sequenced["sequence"].notna()]
    panels = 2 if show_reference_matches else 1
    fig, axes = plt.subplots(
        panels, 1, figsize=(9, 4 if panels == 1 else 7), squeeze=False
    )
    ax = axes[0, 0]
    for worker, group in data.groupby("annotator_id", sort=False):
        worker_label = (
            str(int(worker))
            if isinstance(worker, (int, float, np.integer, np.floating))
            and float(worker).is_integer()
            else str(worker)
        )
        group = group.sort_values("sequence")
        times = pd.to_numeric(group["lead_time"], errors="coerce")
        times = times.where(np.isfinite(times) & times.ge(0))
        (line,) = ax.plot(
            group["sequence"], times, marker="o", label=f"Annotator {worker_label}"
        )
        if show_reference_matches:
            observed = group[group["reference_match"].notna()]
            axes[1, 0].scatter(
                observed["sequence"],
                observed["reference_match"].astype(int),
                color=line.get_color(),
                label=f"Annotator {worker_label}",
            )
    ax.set(
        xlabel="Annotation sequence",
        ylabel="Recorded time (seconds)",
        title="Annotation time over each annotator’s sequence",
    )
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.grid(alpha=0.25)
    if data.empty:
        ax.text(
            0.5,
            0.5,
            "No submitted annotations with a worker and timestamp",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.legend(title="Worker")
    missing_order = int(submitted["sequence"].isna().sum())
    plotted_times = pd.to_numeric(data["lead_time"], errors="coerce")
    missing_times = int(
        (plotted_times.isna() | ~np.isfinite(plotted_times) | plotted_times.lt(0)).sum()
    )
    if missing_order or missing_times:
        ax.text(
            0.01,
            0.98,
            f"Unordered: {missing_order} · Missing/invalid times: {missing_times}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize="small",
        )
    if show_reference_matches:
        match_ax = axes[1, 0]
        match_ax.set(
            xlabel="Annotation sequence",
            ylabel="Reference comparison",
            title="Individual reference comparisons",
        )
        match_ax.set_yticks([0, 1], ["Different", "Match"])
        match_ax.xaxis.get_major_locator().set_params(integer=True)
        match_ax.grid(alpha=0.25)
        if data["reference_match"].notna().sum() == 0:
            match_ax.text(
                0.5,
                0.5,
                "No reference comparisons available",
                ha="center",
                va="center",
                transform=match_ax.transAxes,
            )
    fig.tight_layout()
    return fig, axes[:, 0] if panels > 1 else ax


def diagnostic_table(
    tasks, annotations, reference, label_mapping=None, timing_threshold=None
):
    """Join all reference items with individual annotation rows for inspection.

    Unexported items and exported items without annotations remain visible.
    Annotations outside the manifest are retained and marked. One task can have
    multiple rows; no labels are combined or used to score annotators.
    """
    if (
        reference["example_id"].isna().any()
        or reference["example_id"].astype(str).duplicated().any()
    ):
        raise ValueError(
            "The reference manifest must have unique nonmissing example IDs."
        )
    refs = reference.copy()
    refs["example_id"] = refs["example_id"].astype(str)
    refs["in_manifest"] = True
    ann = flag_timing(annotations, timing_threshold)
    joined = refs.merge(
        ann, on="example_id", how="outer", validate="one_to_many", indicator=True
    )
    joined["in_manifest"] = joined["_merge"].ne("right_only")
    joined = joined.drop(columns="_merge")
    observed = set(tasks["example_id"].dropna().astype(str))
    joined["task_exported"] = joined["example_id"].isin(observed)
    missing = joined["status"].isna()
    joined.loc[missing & joined["task_exported"], "status"] = "no_annotations"
    joined.loc[missing & ~joined["task_exported"], "status"] = "not_exported"
    task_texts = tasks.dropna(subset=["example_id"])[["example_id", "text"]]
    if task_texts["example_id"].duplicated().any():
        warnings.warn(
            "Multiple exported tasks share an example ID; annotation rows remain separate.",
            stacklevel=2,
        )
    task_texts = task_texts.drop_duplicates("example_id").set_index("example_id")[
        "text"
    ]
    joined["text"] = (
        joined["text"]
        .astype("string")
        .fillna(joined["example_id"].map(task_texts).astype("string"))
    )
    if "reference_text" in joined:
        joined["text"] = joined["text"].fillna(
            joined["reference_text"].astype("string")
        )
    mapping = label_mapping or {}
    joined["reference_normalized"] = joined["reference_label"].map(
        lambda value: mapping.get(value, value) if pd.notna(value) else value
    )
    joined["label_normalized"] = joined["label"].map(
        lambda value: mapping.get(value, value) if pd.notna(value) else value
    )
    comparable = (
        joined["status"].eq("submitted")
        & joined["reference_normalized"].notna()
        & joined["label_normalized"].notna()
    )
    joined["reference_match"] = pd.Series(pd.NA, index=joined.index, dtype="boolean")
    joined.loc[comparable, "reference_match"] = joined.loc[
        comparable, "reference_normalized"
    ].eq(joined.loc[comparable, "label_normalized"])
    submitted = joined[joined["status"].eq("submitted") & joined["example_id"].notna()]
    varied_ids = submitted.groupby("example_id")["label_normalized"].nunique()
    joined["different_observed_labels"] = joined["example_id"].isin(
        varied_ids[varied_ids > 1].index
    )
    joined["text_length"] = joined["text"].str.len()
    return joined


def pilot_overview(tasks, annotations, reference):
    """Return descriptive counts using the complete reference manifest denominator."""
    ids = set(reference["example_id"].dropna().astype(str))
    if len(ids) != len(reference):
        raise ValueError("The reference manifest needs unique nonmissing IDs.")
    exported_ids = set(tasks["example_id"].dropna().astype(str))
    submitted = annotations[annotations["status"].eq("submitted")]
    completed_ids = set(submitted["example_id"].dropna().astype(str)) & ids
    in_manifest = submitted[submitted["example_id"].isin(ids)]
    per_task = (
        in_manifest.groupby("example_id").size().reindex(sorted(ids), fill_value=0)
    )
    counts = pd.Series(
        {
            "pilot_pool_items": len(ids),
            "tasks_in_export": len(tasks),
            "manifest_items_in_export": len(exported_ids & ids),
            "manifest_items_not_in_export": len(ids - exported_ids),
            "manifest_items_with_submitted_annotations": len(completed_ids),
            "manifest_items_without_submitted_annotations": len(ids - completed_ids),
            "submitted_annotations": len(submitted),
            "submitted_annotations_outside_manifest": int(
                (~submitted["example_id"].isin(ids)).sum()
            ),
            "distinct_annotators": submitted["annotator_id"].nunique()
            if submitted["annotator_id"].notna().any()
            else pd.NA,
            "submitted_annotations_missing_annotator": int(
                submitted["annotator_id"].isna().sum()
            ),
            "cancelled_annotations": int(annotations["status"].eq("cancelled").sum()),
            "incomplete_annotations": int(annotations["status"].eq("incomplete").sum()),
            "tasks_with_drafts": int(tasks["draft_count"].gt(0).sum()),
        },
        name="count",
    )
    distribution = (
        per_task.value_counts()
        .sort_index()
        .rename_axis("submitted_annotations_per_item")
        .rename("items")
    )
    labels = (
        submitted["label"]
        .value_counts(dropna=False)
        .rename_axis("label")
        .rename("annotations")
    )
    return counts, distribution, labels


def reference_comparison(diagnostics):
    """Summarize matches only where both submitted and reference labels exist."""
    matches = diagnostics["reference_match"].dropna()
    return pd.Series(
        {
            "comparable_annotations": len(matches),
            "matches": int(matches.sum()),
            "mismatches": int((~matches).sum()),
            "descriptive_match_rate": float(matches.mean()) if len(matches) else np.nan,
        }
    )


def plot_annotation_times(annotations):
    """Plot recorded submission times, retaining long observations."""
    rows = annotations[annotations["status"].eq("submitted")]
    times = pd.to_numeric(rows["lead_time"], errors="coerce")
    times = times[times.notna() & np.isfinite(times) & times.ge(0)]
    fig, ax = plt.subplots(figsize=(8, 4))
    if times.empty:
        ax.text(
            0.5,
            0.5,
            "No valid times available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        ax.hist(
            times, bins=min(15, max(3, len(times))), color="#3d7770", edgecolor="white"
        )
    ax.set(
        xlabel="Annotation time (seconds)",
        ylabel="Annotations",
        title="Annotation time distribution",
    )
    ax.yaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    return fig, ax
