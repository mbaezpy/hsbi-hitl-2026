"""Load text classification data and build fixed acquisition representations."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

BANKING_INTENTS = {
    "passcode_forgotten": "account_access_passcode",
    "pending_transfer": "bank_transfer_pending",
    "transfer_not_received_by_recipient": "bank_transfer_not_received",
    "pending_card_payment": "card_payment_pending",
    "declined_card_payment": "card_payment_declined",
    "transaction_charged_twice": "card_payment_duplicated",
    "pending_cash_withdrawal": "cash_withdrawal_pending",
    "cash_withdrawal_not_recognised": "cash_withdrawal_unrecognized",
    "card_acceptance": "card_acceptance",
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def select_dataset_classes(data, classes=None, max_classes=None, random_state=42):
    """Select classes across all splits, preserving their original label IDs."""
    if classes is not None and max_classes is not None:
        raise ValueError("Specify classes or max_classes, not both.")
    available = np.sort(data["label"].unique())
    if not len(available):
        raise ValueError("The dataset has no classes.")
    if "label_name" in data:
        mapping = data[["label", "label_name"]].drop_duplicates()
        if (
            mapping["label"].duplicated().any()
            or mapping["label_name"].duplicated().any()
        ):
            raise ValueError("Class IDs and names must map one-to-one.")
        names = dict(zip(mapping["label"], mapping["label_name"]))
    else:
        names = {label: str(label) for label in available}
    if max_classes is not None:
        if (
            isinstance(max_classes, bool)
            or not isinstance(max_classes, (int, np.integer))
            or max_classes < 1
        ):
            raise ValueError("max_classes must be a positive integer.")
        chosen = (
            np.random.default_rng(random_state).choice(
                available, size=max_classes, replace=False
            )
            if max_classes < len(available)
            else available
        )
    elif classes is not None:
        if not isinstance(classes, (list, tuple)) or not classes:
            raise ValueError("classes must be a nonempty list of class IDs or names.")
        by_name = {name: label for label, name in names.items()}
        chosen = []
        for item in classes:
            if isinstance(item, str):
                if item not in by_name:
                    raise ValueError(f"Unknown class name: {item!r}.")
                label = by_name[item]
            elif (
                isinstance(item, (int, np.integer))
                and not isinstance(item, (bool, np.bool_))
                and item in names
            ):
                label = item
            else:
                raise ValueError(f"Unknown class ID: {item!r}.")
            if label in chosen:
                raise ValueError(f"Duplicate selected class: {item!r}.")
            chosen.append(label)
    else:
        chosen = available
    chosen = sorted(chosen)
    filtered = data[data["label"].isin(chosen)].copy()
    filtered.attrs["selected_classes"] = [
        {"id": int(label), "name": str(names[label])} for label in chosen
    ]
    return filtered


def prepare_text_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    label_name_col: str | None = None,
) -> pd.DataFrame:
    """Standardize text and labels while preserving stable IDs and splits.

    Existing ``sample_id`` values are retained. Otherwise use the source index,
    prefixed with its split. Pass all splits together for consistent class IDs.
    """
    if df[text_col].isna().any() or df[label_col].isna().any():
        raise ValueError("Text and class labels must not be missing.")
    if not df.index.is_unique and "sample_id" not in df:
        raise ValueError("Provide sample_id when the source index is not unique.")
    labels = df[label_col]
    names = df[label_name_col] if label_name_col else labels.astype(str)
    mapping = pd.DataFrame({"source_label": labels, "name": names})
    if mapping.groupby("source_label")["name"].nunique().gt(1).any():
        raise ValueError("Each source label must have one consistent label name.")
    classes = sorted(labels.unique(), key=lambda value: (str(type(value)), str(value)))
    label_ids = {label: index for index, label in enumerate(classes)}
    if "sample_id" in df:
        sample_ids = df["sample_id"].astype(str)
    elif "split" in df:
        sample_ids = df["split"].astype(str) + ":" + df.index.astype(str)
    else:
        sample_ids = pd.Series(df.index.astype(str), index=df.index)
    result = pd.DataFrame(
        {
            "sample_id": sample_ids.to_numpy(),
            "text": df[text_col].astype(str).to_numpy(),
            "label": labels.map(label_ids).astype(int).to_numpy(),
            "label_name": names.astype(str).to_numpy(),
            "split": df["split"].astype(str).to_numpy()
            if "split" in df
            else "unspecified",
        }
    )
    if result["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique across all splits.")
    return result


def load_banking77(cache_dir: str = "data") -> pd.DataFrame:
    """Load the locally downloaded BANKING77 train/test splits from Lab 04."""
    from datasets import load_from_disk

    path = Path(cache_dir).expanduser()
    if cache_dir == "data":
        path = REPOSITORY_ROOT / "data"
    if not (path / "dataset_dict.json").is_file():
        path = path / "PolyAI_banking77"
    if not (path / "dataset_dict.json").is_file():
        raise FileNotFoundError(
            f"BANKING77 DatasetDict not found at {path.resolve()}. "
            "From the repository root run: python scripts/download_dataset.py "
            "PolyAI/banking77 data/PolyAI_banking77"
        )
    dataset = load_from_disk(str(path.resolve()))
    names = dataset["train"].features["label"].names
    frames = []
    for split in ("train", "test"):
        frame = dataset[split].to_pandas()
        frame["source_intent"] = frame["label"].map(lambda index: names[index])
        frame = frame[frame["source_intent"].isin(BANKING_INTENTS)].copy()
        frame["sample_id"] = split + ":" + frame.index.astype(str)
        frame["label_name"] = frame["source_intent"].map(BANKING_INTENTS)
        frame["split"] = split
        frames.append(frame)
    result = prepare_text_dataset(
        pd.concat(frames, ignore_index=True), label_name_col="label_name"
    )
    result["intent"] = result["label_name"]
    return result


def load_20newsgroups(cache_dir: str = "data") -> pd.DataFrame:
    """Load the locally downloaded SetFit 20 Newsgroups train/test splits.

    ``cache_dir`` may be the dataset directory or its parent. This dataset
    stores human-readable categories in ``label_text`` rather than ClassLabel
    metadata. Standardizing those names produces the original 0–19 ordering
    for this dataset and keeps the mapping consistent across both splits.
    """
    from datasets import load_from_disk

    path = Path(cache_dir).expanduser()
    if cache_dir == "data":
        path = REPOSITORY_ROOT / "data"
    if not (path / "dataset_dict.json").is_file():
        path = path / "SetFit_20newsgroups"
    if not (path / "dataset_dict.json").is_file():
        raise FileNotFoundError(
            f"20 Newsgroups DatasetDict not found at {path.resolve()}. "
            "From the repository root run: python scripts/download_dataset.py "
            "SetFit/20_newsgroups data/SetFit_20newsgroups"
        )
    dataset = load_from_disk(str(path.resolve()))
    frames = []
    for split in ("train", "test"):
        frame = dataset[split].to_pandas()
        frame["sample_id"] = split + ":" + frame.index.astype(str)
        frame["split"] = split
        frames.append(frame)
    result = prepare_text_dataset(
        pd.concat(frames, ignore_index=True),
        label_col="label_text",
        label_name_col="label_text",
    )
    result.attrs["dataset_id"] = "SetFit/20_newsgroups"
    return result


def build_representation(
    train_texts,
    eval_texts,
    method: str = "tfidf",
    model_name: str | None = None,
    **kwargs,
):
    """Build fixed vectors; fit TF-IDF on training-side text only.

    Embeddings default to Lab 04's mean-pooled GloVe vectors. A different
    model name uses the optional sentence-transformers package instead. All
    nonzero embeddings are L2-normalized. Returns train, eval, and model info.
    """
    train_texts, eval_texts = list(train_texts), list(eval_texts)
    if method == "tfidf":
        if model_name is not None:
            raise ValueError("model_name is not supported for TF-IDF.")
        vectorizer = TfidfVectorizer(**kwargs)
        return (
            vectorizer.fit_transform(train_texts),
            vectorizer.transform(eval_texts),
            {"method": method, **kwargs, "vectorizer": vectorizer},
        )
    if method != "embeddings":
        raise ValueError(f"Unknown representation: {method!r}")
    model_name = model_name or "glove-wiki-gigaword-50"
    if model_name == "glove-wiki-gigaword-50":
        import os

        os.environ["GENSIM_DATA_DIR"] = str(
            Path(kwargs.pop("models_dir", REPOSITORY_ROOT / "models")).resolve()
        )
        import gensim.downloader as api
        import spacy

        vectors = api.load(model_name)
        nlp = spacy.load("en_core_web_sm")

        def encode(texts):
            result = []
            for doc in nlp.pipe(texts):
                words = [
                    token.lemma_.lower()
                    for token in doc
                    if not token.is_stop and not token.is_punct and token.is_alpha
                ]
                known = [
                    vectors[word] for word in words if word in vectors.key_to_index
                ]
                result.append(
                    np.mean(known, axis=0) if known else np.zeros(vectors.vector_size)
                )
            return (
                normalize(np.vstack(result))
                if result
                else np.empty((0, vectors.vector_size))
            )

        return (
            encode(train_texts),
            encode(eval_texts),
            {"method": method, "model_name": model_name, "model": vectors},
        )
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, **kwargs)
    return (
        normalize(model.encode(train_texts, convert_to_numpy=True)),
        normalize(model.encode(eval_texts, convert_to_numpy=True)),
        {"method": method, "model_name": model_name, "model": model},
    )
