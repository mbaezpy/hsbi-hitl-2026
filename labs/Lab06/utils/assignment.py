"""Sample classification examples and prepare balanced worker assignments."""

import numbers

import numpy as np
import pandas as pd


def prepare_for_assignment(
    df,
    id_col="example_id",
    text_col="text",
    sample_size=18,
    num_workers=3,
    num_ratings_per_example=2,
    seed=42,
):
    """Sample examples and create balanced, reproducible worker assignments.

    Returns ``(selected_examples, label_studio_tasks, assignment_manifest)``.
    Selected examples retain all original columns for local reference. The
    label-free tasks have ``example_id``, ``text``, and a delimiter-wrapped
    ``assigned_workers`` string; the manifest has one ``example_id`` and
    ``worker_id`` row per assignment. Worker IDs are organizational labels,
    not Label Studio accounts or access restrictions.

    IDs are preserved as supplied. When importing CSV into Label Studio, use
    nonnumeric string IDs (for example ``train:001``) if leading zeroes matter,
    because CSV type inference can interpret numeric-looking IDs as numbers.

    A seeded worker permutation is traversed round-robin across all slots;
    workloads therefore differ by at most one and workers do not repeat on an
    item. This function is for single-label text classification.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not df.columns.is_unique:
        raise ValueError("DataFrame column names must be unique.")
    if id_col == text_col or id_col not in df or text_col not in df:
        raise ValueError("Provide distinct existing ID and text columns.")
    for name, value in (
        ("sample_size", sample_size),
        ("num_workers", num_workers),
        ("num_ratings_per_example", num_ratings_per_example),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Integral)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    if sample_size > len(df):
        raise ValueError("sample_size cannot exceed the number of examples.")
    if num_ratings_per_example > num_workers:
        raise ValueError("num_workers must be at least num_ratings_per_example.")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, numbers.Integral)
        or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer.")
    ids = df[id_col]
    if (
        ids.isna().any()
        or ids.astype(str).str.strip().eq("").any()
        or ids.astype(str).duplicated().any()
    ):
        raise ValueError(
            "Example IDs must be nonempty and unique, including when written as text."
        )
    texts = df[text_col]
    if (
        texts.isna().any()
        or not texts.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all()
    ):
        raise ValueError("Text values must be nonempty strings.")

    rng = np.random.default_rng(seed)
    selected = (
        df.iloc[rng.choice(len(df), size=sample_size, replace=False)]
        .copy()
        .reset_index(drop=True)
    )
    width = max(2, len(str(num_workers)))
    workers = [f"worker_{i:0{width}d}" for i in range(1, num_workers + 1)]
    order = rng.permutation(workers).tolist()
    assignment_rows = []
    task_rows = []
    slot = 0
    for example_id, text in zip(selected[id_col], selected[text_col]):
        assigned = [
            order[(slot + offset) % num_workers]
            for offset in range(num_ratings_per_example)
        ]
        slot += num_ratings_per_example
        task_rows.append(
            {
                "example_id": example_id,
                "text": text,
                "assigned_workers": "|" + "|".join(assigned) + "|",
            }
        )
        assignment_rows.extend(
            {"example_id": example_id, "worker_id": worker} for worker in assigned
        )
    return selected, pd.DataFrame(task_rows), pd.DataFrame(assignment_rows)


def prepare_image_for_assignment(
    df,
    id_col="example_id",
    image_col="image",
    sample_size=18,
    num_workers=3,
    num_ratings_per_example=2,
    seed=42,
):
    """Use the same assignment logic for image-classification URL strings.

    Returns selected source rows, label-free tasks with an ``image`` field,
    and the same long-format worker manifest as the text helper.
    """
    selected, tasks, assignments = prepare_for_assignment(
        df,
        id_col=id_col,
        text_col=image_col,
        sample_size=sample_size,
        num_workers=num_workers,
        num_ratings_per_example=num_ratings_per_example,
        seed=seed,
    )
    return selected, tasks.rename(columns={"text": "image"}), assignments
