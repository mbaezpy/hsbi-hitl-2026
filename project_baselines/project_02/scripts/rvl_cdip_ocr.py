"""Shared, tested OCR primitives for the RVL-CDIP G/J2 pipeline."""

from __future__ import annotations

import math
import re
import shlex
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import pytesseract
from PIL import Image

OCR_PIPELINE_ID = "rvl_cdip_g_j2_v1"
OCR_ENGINE_VERSION = "1"
POLICY_VERSION = "3"
COMMON_PUNCTUATION = set(".,;:!?\"'`-_/\\()[]{}%$€£¥@&#*+=<>|~^")


@dataclass(frozen=True)
class PolicyParameters:
    """Parameters used by configuration G and the tested J2 policy."""

    upscale_min_dimension: int = 1800
    upscale_max_factor: float = 2.0
    mean_confidence_threshold: float = 60.0
    median_confidence_threshold: float = 60.0
    low_confidence_fraction_threshold: float = 0.40
    nearly_empty_word_count: int = 3
    min_alphabetic_fraction: float = 0.45
    max_unusual_fraction: float = 0.08
    max_short_token_fraction: float = 0.35
    min_box_coverage: float = 0.001
    confidence_gain: float = 5.0
    low_confidence_gain: float = 0.10
    word_gain_fraction: float = 0.25
    min_word_gain: int = 8
    plausibility_gain: float = 0.05
    coverage_gain: float = 0.002

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_POLICY = PolicyParameters()


def split_primary_config(raw_config: str) -> tuple[str, ...]:
    """Return non-PSM options, rejecting a PSM that conflicts with G."""
    try:
        tokens = shlex.split(raw_config)
    except ValueError as exc:
        raise ValueError(f"Invalid --tesseract-config: {exc}") from exc
    remaining: list[str] = []
    explicit_psm: list[int] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"--psm", "-psm"}:
            if index + 1 >= len(tokens):
                raise ValueError(f"{token} requires an integer value")
            try:
                explicit_psm.append(int(tokens[index + 1]))
            except ValueError as exc:
                raise ValueError(f"Invalid PSM value {tokens[index + 1]!r}") from exc
            index += 2
            continue
        if token.startswith("--psm="):
            try:
                explicit_psm.append(int(token.split("=", 1)[1]))
            except ValueError as exc:
                raise ValueError(f"Invalid PSM option {token!r}") from exc
            index += 1
            continue
        remaining.append(token)
        index += 1
    if any(psm != 3 for psm in explicit_psm):
        raise ValueError(
            "The integrated G/J2 pipeline requires primary PSM 3. Remove the "
            "conflicting PSM or use --tesseract-config with --psm 3."
        )
    return tuple(remaining)


def config_for_psm(base_options: tuple[str, ...], psm: int) -> str:
    return shlex.join((*base_options, "--psm", str(psm)))


def upscale_image(
    grayscale: Image.Image, parameters: PolicyParameters = DEFAULT_POLICY
) -> tuple[Image.Image, float]:
    image = grayscale.convert("L")
    scale_factor = 1.0
    smallest_dimension = min(image.size)
    if smallest_dimension < parameters.upscale_min_dimension:
        scale_factor = min(
            parameters.upscale_max_factor,
            parameters.upscale_min_dimension / smallest_dimension,
        )
        new_size = tuple(max(1, round(value * scale_factor)) for value in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image, scale_factor


def rotate_clockwise(image: Image.Image, rotation: int) -> Image.Image:
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f"Unsupported rotation {rotation}")
    return (
        image if rotation == 0 else image.rotate(-rotation, expand=True, fillcolor=255)
    )


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def text_metrics(text: str) -> dict[str, Any]:
    visible = [character for character in text if not character.isspace()]
    tokens = re.findall(r"\S+", text)
    word_tokens = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    short_tokens = [
        token
        for token in tokens
        if len("".join(character for character in token if character.isalnum())) <= 1
    ]
    unusual = [
        character
        for character in visible
        if not character.isalnum() and character not in COMMON_PUNCTUATION
    ]
    return {
        "ocr_char_count": len(text),
        "ocr_word_count": len(word_tokens),
        "alphabetic_char_fraction": (
            sum(character.isalpha() for character in visible) / len(visible)
            if visible
            else None
        ),
        "unusual_char_fraction": len(unusual) / len(visible) if visible else None,
        "short_token_fraction": len(short_tokens) / len(tokens) if tokens else None,
    }


def data_metrics(data: dict[str, list[Any]], width: int, height: int) -> dict[str, Any]:
    confidences: list[float] = []
    box_area = 0
    recognized_words = 0
    for index, raw_text in enumerate(data.get("text", [])):
        if not str(raw_text).strip():
            continue
        recognized_words += 1
        raw_confidences = data.get("conf", [])
        confidence = (
            safe_float(raw_confidences[index]) if index < len(raw_confidences) else None
        )
        if confidence is not None and confidence >= 0:
            confidences.append(confidence)
        try:
            box_area += int(data["width"][index]) * int(data["height"][index])
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    return {
        "mean_word_confidence": statistics.fmean(confidences) if confidences else None,
        "median_word_confidence": (
            statistics.median(confidences) if confidences else None
        ),
        "low_confidence_word_fraction": (
            sum(value < 50 for value in confidences) / len(confidences)
            if confidences
            else None
        ),
        "recognized_word_count": recognized_words,
        "text_box_count": recognized_words,
        "text_box_coverage_fraction": (
            min(1.0, box_area / (width * height)) if width and height else None
        ),
    }


def empty_ocr_metrics() -> dict[str, Any]:
    return {
        "ocr_text": "",
        "mean_word_confidence": None,
        "median_word_confidence": None,
        "low_confidence_word_fraction": None,
        "recognized_word_count": 0,
        "ocr_word_count": 0,
        "ocr_char_count": 0,
        "text_box_count": 0,
        "text_box_coverage_fraction": None,
        "alphabetic_char_fraction": None,
        "unusual_char_fraction": None,
        "short_token_fraction": None,
    }


def run_ocr_candidate(
    image: Image.Image,
    lang: str,
    config: str,
    timeout: float,
) -> dict[str, Any]:
    """Run the same text and confidence calls used by the adaptive benchmark."""
    started = time.perf_counter()
    result = empty_ocr_metrics()
    errors: list[str] = []
    text_succeeded = False
    data_succeeded = False
    try:
        text = pytesseract.image_to_string(
            image, lang=lang, config=config, timeout=timeout
        ).strip()
        result.update({"ocr_text": text, **text_metrics(text)})
        text_succeeded = True
    except (RuntimeError, pytesseract.TesseractError) as exc:
        errors.append(f"text: {str(exc).replace(chr(10), ' ')}")
    try:
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            config=config,
            timeout=timeout,
            output_type=pytesseract.Output.DICT,
        )
        result.update(data_metrics(data, image.width, image.height))
        data_succeeded = True
    except (RuntimeError, pytesseract.TesseractError) as exc:
        errors.append(f"confidence: {str(exc).replace(chr(10), ' ')}")
    if text_succeeded and data_succeeded:
        status = "ok" if result["ocr_text"] else "empty"
    elif text_succeeded or data_succeeded:
        status = "partial"
    else:
        status = "error"
    result.update(
        {
            "ocr_status": status,
            "ocr_error": " | ".join(errors),
            "ocr_seconds": time.perf_counter() - started,
            "text_succeeded": text_succeeded,
            "data_succeeded": data_succeeded,
        }
    )
    return result


def value(row: dict[str, Any], field: str) -> float | None:
    return safe_float(row.get(field))


def quality_assessment(
    row: dict[str, Any], parameters: PolicyParameters = DEFAULT_POLICY
) -> dict[str, Any]:
    reasons: list[str] = []
    families: set[str] = set()
    words = int(float(row.get("ocr_word_count") or 0))
    empty = not str(row.get("ocr_text") or "").strip()
    mean_conf = value(row, "mean_word_confidence")
    median_conf = value(row, "median_word_confidence")
    low_conf = value(row, "low_confidence_word_fraction")
    alpha = value(row, "alphabetic_char_fraction")
    unusual = value(row, "unusual_char_fraction")
    short = value(row, "short_token_fraction")
    coverage = value(row, "text_box_coverage_fraction")
    if empty:
        reasons.append("empty_output")
        families.add("extraction")
    if words <= parameters.nearly_empty_word_count:
        reasons.append("nearly_empty")
        families.add("quantity")
    if mean_conf is None or mean_conf < parameters.mean_confidence_threshold:
        reasons.append("low_mean_confidence")
        families.add("confidence")
    if median_conf is None or median_conf < parameters.median_confidence_threshold:
        reasons.append("low_median_confidence")
        families.add("confidence")
    if low_conf is None or low_conf > parameters.low_confidence_fraction_threshold:
        reasons.append("many_low_confidence_words")
        families.add("confidence")
    if alpha is not None and alpha < parameters.min_alphabetic_fraction:
        reasons.append("low_alphabetic_fraction")
        families.add("plausibility")
    if unusual is not None and unusual > parameters.max_unusual_fraction:
        reasons.append("high_unusual_character_fraction")
        families.add("plausibility")
    if short is not None and short > parameters.max_short_token_fraction:
        reasons.append("high_short_token_fraction")
        families.add("plausibility")
    if coverage is not None and coverage < parameters.min_box_coverage:
        reasons.append("low_text_box_coverage")
        families.add("layout")
    suspicious = (
        empty
        or "confidence" in families
        or "plausibility" in families
        or {"quantity", "layout"}.issubset(families)
    )
    return {
        "suspicious": suspicious,
        "reasons": reasons if suspicious else [],
        "observed_reasons": reasons,
        "signal_families": sorted(families),
    }


def plausible(
    row: dict[str, Any], parameters: PolicyParameters = DEFAULT_POLICY
) -> bool:
    alpha = value(row, "alphabetic_char_fraction")
    unusual = value(row, "unusual_char_fraction")
    short = value(row, "short_token_fraction")
    return (
        (alpha is None or alpha >= parameters.min_alphabetic_fraction)
        and (unusual is None or unusual <= parameters.max_unusual_fraction)
        and (short is None or short <= parameters.max_short_token_fraction)
    )


def compare_candidate(
    primary: dict[str, Any],
    candidate: dict[str, Any],
    parameters: PolicyParameters = DEFAULT_POLICY,
) -> dict[str, Any]:
    improvements: list[str] = []
    regressions: list[str] = []
    improvement_families: set[str] = set()
    regression_families: set[str] = set()
    primary_words = int(float(primary.get("ocr_word_count") or 0))
    candidate_words = int(float(candidate.get("ocr_word_count") or 0))
    primary_empty = not str(primary.get("ocr_text") or "").strip()
    candidate_plausible = plausible(candidate, parameters)
    if (
        primary_empty
        and candidate_words >= parameters.min_word_gain
        and candidate_plausible
    ):
        improvements.append("recovered_missing_plausible_text")
        improvement_families.update(("extraction", "quantity"))
    word_delta = candidate_words - primary_words
    needed_words = max(
        parameters.min_word_gain,
        math.ceil(primary_words * parameters.word_gain_fraction),
    )
    if word_delta >= needed_words and candidate_plausible:
        improvements.append("meaningful_plausible_word_recovery")
        improvement_families.add("quantity")
    if word_delta <= -needed_words:
        regressions.append("meaningful_word_loss")
        regression_families.add("quantity")
    directional = (
        (
            "mean_word_confidence",
            parameters.confidence_gain,
            True,
            "mean_confidence",
            "confidence",
        ),
        (
            "low_confidence_word_fraction",
            parameters.low_confidence_gain,
            False,
            "low_confidence_fraction",
            "confidence",
        ),
        (
            "alphabetic_char_fraction",
            parameters.plausibility_gain,
            True,
            "alphabetic_fraction",
            "plausibility",
        ),
        (
            "unusual_char_fraction",
            parameters.plausibility_gain,
            False,
            "unusual_character_fraction",
            "plausibility",
        ),
        (
            "short_token_fraction",
            parameters.plausibility_gain,
            False,
            "short_token_fraction",
            "plausibility",
        ),
        (
            "text_box_coverage_fraction",
            parameters.coverage_gain,
            True,
            "text_box_coverage",
            "layout",
        ),
    )
    for field, threshold, higher_is_better, label, family in directional:
        old, new = value(primary, field), value(candidate, field)
        if old is None or new is None:
            continue
        delta = (new - old) if higher_is_better else (old - new)
        if delta >= threshold:
            improvements.append(f"improved_{label}")
            improvement_families.add(family)
        elif delta <= -threshold:
            regressions.append(f"worse_{label}")
            regression_families.add(family)
    clear = (
        "recovered_missing_plausible_text" in improvements and not regression_families
    ) or (len(improvement_families) >= 2 and not regression_families)
    return {
        "improvements": improvements,
        "regressions": regressions,
        "improvement_families": sorted(improvement_families),
        "regression_families": sorted(regression_families),
        "clear_improvement": clear,
        "indicators_disagree": bool(improvement_families and regression_families),
    }


def select_rotation(
    primary: dict[str, Any],
    candidates: list[dict[str, Any]],
    parameters: PolicyParameters = DEFAULT_POLICY,
) -> tuple[dict[str, Any], str, bool, list[dict[str, Any]]]:
    comparisons = []
    for candidate in candidates:
        comparison = compare_candidate(primary, candidate, parameters)
        comparisons.append({"candidate_id": candidate["candidate_id"], **comparison})
    eligible = [
        (candidate, comparison)
        for candidate, comparison in zip(candidates, comparisons, strict=True)
        if comparison["clear_improvement"]
    ]
    disagreement = any(item["indicators_disagree"] for item in comparisons)
    if not eligible:
        return (
            primary,
            "Retained primary: no rotation improved multiple indicators without regression.",
            disagreement,
            comparisons,
        )
    best_count = max(
        len(comparison["improvement_families"]) for _, comparison in eligible
    )
    finalists = [
        (row, comparison)
        for row, comparison in eligible
        if len(comparison["improvement_families"]) == best_count
    ]
    if len(finalists) != 1:
        disagreement = True
        names = ", ".join(row["candidate_id"] for row, _ in finalists)
        return (
            primary,
            f"Retained primary: equally supported rotation candidates were ambiguous ({names}).",
            disagreement,
            comparisons,
        )
    selected, comparison = finalists[0]
    rationale = "Selected {name}: {reasons}.".format(
        name=selected["candidate_id"],
        reasons=", ".join(comparison["improvements"]),
    )
    return selected, rationale, disagreement, comparisons


def select_segmentation(
    current: dict[str, Any],
    candidate: dict[str, Any],
    parameters: PolicyParameters = DEFAULT_POLICY,
) -> tuple[dict[str, Any], str, bool, dict[str, Any]]:
    comparison = compare_candidate(current, candidate, parameters)
    if comparison["clear_improvement"]:
        rationale = "Selected {name}: {reasons}.".format(
            name=candidate["candidate_id"],
            reasons=", ".join(comparison["improvements"]),
        )
        return candidate, rationale, comparison["indicators_disagree"], comparison
    rationale = (
        "Retained rotation-stage result: PSM 11 lacked a clear "
        "multi-indicator improvement"
    )
    if comparison["regressions"]:
        rationale += " and regressed " + ", ".join(comparison["regressions"])
    return current, rationale + ".", comparison["indicators_disagree"], comparison


CandidateProvider = Callable[[int, int], dict[str, Any]]


def apply_j2_policy(
    primary: dict[str, Any],
    candidate_provider: CandidateProvider,
    parameters: PolicyParameters = DEFAULT_POLICY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the tested J2 recovery sequence without recomputing primary 0° OCR."""
    assessment = quality_assessment(primary, parameters)
    selected = primary
    attempted = [primary["candidate_id"]]
    rationale = "Primary result passed the recovery trigger."
    disagreement = False
    rotation_comparisons: list[dict[str, Any]] = []
    segmentation_comparison: dict[str, Any] | None = None
    if assessment["suspicious"]:
        rotations = [candidate_provider(angle, 3) for angle in (90, 180, 270)]
        attempted.extend(row["candidate_id"] for row in rotations)
        selected, rationale, disagreement, rotation_comparisons = select_rotation(
            primary, rotations, parameters
        )
        if quality_assessment(selected, parameters)["suspicious"]:
            psm11 = candidate_provider(int(selected["rotation"]), 11)
            attempted.append(psm11["candidate_id"])
            (
                selected,
                segmentation_rationale,
                segment_disagreement,
                segmentation_comparison,
            ) = select_segmentation(selected, psm11, parameters)
            rationale += " " + segmentation_rationale
            disagreement = disagreement or segment_disagreement
    return selected, {
        "triggered_recovery": assessment["suspicious"],
        "trigger_reasons": assessment["reasons"],
        "attempted_candidate_ids": attempted,
        "selected_candidate_id": selected["candidate_id"],
        "selection_rationale": rationale,
        "indicator_disagreement": disagreement,
        "remaining_suspicious": quality_assessment(selected, parameters)["suspicious"],
        "rotation_comparisons": rotation_comparisons,
        "segmentation_comparison": segmentation_comparison,
    }
