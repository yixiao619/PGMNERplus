"""Dependency-light binary metrics, threshold selection, and significance tests."""

from __future__ import annotations

import math
import random
from statistics import mean, pstdev
from typing import Any, Sequence


def _validate(labels: Sequence[int], probabilities: Sequence[float]) -> None:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must have the same non-zero length")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("labels must be binary")
    if any(not math.isfinite(float(value)) for value in probabilities):
        raise ValueError("probabilities must be finite")


def confusion_counts(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, int]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    tp = tn = fp = fn = 0
    for label, prediction in zip(labels, predictions):
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 0:
            tn += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 1 and prediction == 0:
            fn += 1
        else:
            raise ValueError("labels and predictions must be binary")
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _class_prf(*, tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    _validate(labels, probabilities)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    pairs = sorted(zip(probabilities, labels), key=lambda item: float(item[0]), reverse=True)
    tp = fp = 0
    previous_tpr = previous_fpr = area = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        group_tp = group_fp = 0
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1] == 1:
                group_tp += 1
            else:
                group_fp += 1
            index += 1
        tp += group_tp
        fp += group_fp
        tpr = tp / positives
        fpr = fp / negatives
        area += (fpr - previous_fpr) * (tpr + previous_tpr) / 2.0
        previous_tpr, previous_fpr = tpr, fpr
    return float(area)


def average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    _validate(labels, probabilities)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    pairs = sorted(zip(probabilities, labels), key=lambda item: float(item[0]), reverse=True)
    tp = fp = 0
    previous_recall = area = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1] == 1:
                tp += 1
            else:
                fp += 1
            index += 1
        recall = tp / positives
        precision = tp / max(1, tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return float(area)


def binary_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    _validate(labels, probabilities)
    predictions = [int(float(value) >= threshold) for value in probabilities]
    counts = confusion_counts(labels, predictions)
    positive = _class_prf(tp=counts["tp"], fp=counts["fp"], fn=counts["fn"])
    negative = _class_prf(tp=counts["tn"], fp=counts["fn"], fn=counts["fp"])
    accuracy = _safe_ratio(counts["tp"] + counts["tn"], len(labels))
    balanced = (positive["recall"] + negative["recall"]) / 2.0
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": (positive["f1"] + negative["f1"]) / 2.0,
        "positive": positive,
        "negative": negative,
        "roc_auc": roc_auc(labels, probabilities),
        "pr_auc": average_precision(labels, probabilities),
        "confusion_matrix": counts,
    }


def _candidate_thresholds(probabilities: Sequence[float]) -> list[float]:
    values = sorted({float(value) for value in probabilities})
    if not values:
        raise ValueError("probabilities must not be empty")
    return [math.nextafter(values[0], -math.inf), *values, math.nextafter(values[-1], math.inf)]


def _threshold_headline(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float
) -> tuple[float, float, float, float]:
    predictions = [int(float(value) >= threshold) for value in probabilities]
    counts = confusion_counts(labels, predictions)
    positive = _class_prf(tp=counts["tp"], fp=counts["fp"], fn=counts["fn"])
    negative = _class_prf(tp=counts["tn"], fp=counts["fn"], fn=counts["fp"])
    balanced = (positive["recall"] + negative["recall"]) / 2.0
    macro_f1 = (positive["f1"] + negative["f1"]) / 2.0
    return balanced, macro_f1, positive["recall"], negative["recall"]


def choose_balanced_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    _validate(labels, probabilities)
    best_threshold = 0.5
    best_key: tuple[float, float, float, float] | None = None
    for threshold in _candidate_thresholds(probabilities):
        balanced, macro_f1, positive_recall, _ = _threshold_headline(
            labels, probabilities, threshold
        )
        key = (
            balanced,
            macro_f1,
            positive_recall,
            -abs(threshold - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def choose_recall_constrained_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    minimum_recall: float = 0.95,
) -> float:
    _validate(labels, probabilities)
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be between 0 and 1")
    eligible: list[tuple[float, float, float]] = []
    for threshold in _candidate_thresholds(probabilities):
        _, _, recall, specificity = _threshold_headline(labels, probabilities, threshold)
        if recall + 1e-12 >= minimum_recall:
            eligible.append((specificity, threshold, recall))
    if not eligible:
        raise ValueError("no threshold satisfies the requested recall")
    _, threshold, _ = max(eligible)
    return float(threshold)


def aggregate_metric_dicts(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all requested scalar classification metrics across seeds."""

    if not metrics:
        raise ValueError("metrics must not be empty")

    def summarize(values: Sequence[float | int]) -> dict[str, float]:
        numeric = [float(value) for value in values]
        return {"mean": mean(numeric), "std": pstdev(numeric)}

    keys = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "pr_auc")
    result: dict[str, Any] = {"seeds": len(metrics)}
    for key in keys:
        result[key] = summarize([item[key] for item in metrics])
    for class_name in ("positive", "negative"):
        result[class_name] = {
            key: summarize([item[class_name][key] for item in metrics])
            for key in ("precision", "recall", "f1")
        }
    result["confusion_matrix"] = {
        key: summarize([item["confusion_matrix"][key] for item in metrics])
        for key in ("tn", "fp", "fn", "tp")
    }
    return result


def stratified_bootstrap_balanced_accuracy_difference(
    labels: Sequence[int],
    predictions_a: Sequence[int],
    predictions_b: Sequence[int],
    *,
    samples: int = 10_000,
    seed: int = 2026,
) -> dict[str, float | int]:
    if not (len(labels) == len(predictions_a) == len(predictions_b)):
        raise ValueError("bootstrap inputs must have the same length")
    positive_indices = [index for index, label in enumerate(labels) if label == 1]
    negative_indices = [index for index, label in enumerate(labels) if label == 0]
    if not positive_indices or not negative_indices:
        raise ValueError("stratified bootstrap requires both classes")
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(samples):
        indices = [rng.choice(positive_indices) for _ in positive_indices]
        indices += [rng.choice(negative_indices) for _ in negative_indices]
        sampled_labels = [labels[index] for index in indices]
        sampled_a = [int(predictions_a[index]) for index in indices]
        sampled_b = [int(predictions_b[index]) for index in indices]
        counts_a = confusion_counts(sampled_labels, sampled_a)
        counts_b = confusion_counts(sampled_labels, sampled_b)
        a = 0.5 * (
            _safe_ratio(counts_a["tp"], counts_a["tp"] + counts_a["fn"])
            + _safe_ratio(counts_a["tn"], counts_a["tn"] + counts_a["fp"])
        )
        b = 0.5 * (
            _safe_ratio(counts_b["tp"], counts_b["tp"] + counts_b["fn"])
            + _safe_ratio(counts_b["tn"], counts_b["tn"] + counts_b["fp"])
        )
        differences.append(a - b)
    differences.sort()
    lower = differences[int(0.025 * (samples - 1))]
    upper = differences[int(0.975 * (samples - 1))]
    return {
        "samples": samples,
        "seed": seed,
        "mean_difference": mean(differences),
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def mcnemar_exact(
    labels: Sequence[int], predictions_a: Sequence[int], predictions_b: Sequence[int]
) -> dict[str, float | int]:
    if not (len(labels) == len(predictions_a) == len(predictions_b)):
        raise ValueError("McNemar inputs must have the same length")
    a_only = b_only = 0
    for label, pred_a, pred_b in zip(labels, predictions_a, predictions_b):
        correct_a = pred_a == label
        correct_b = pred_b == label
        if correct_a and not correct_b:
            a_only += 1
        elif correct_b and not correct_a:
            b_only += 1
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(a_only, b_only)
        probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * probability)
    return {
        "a_correct_b_wrong": a_only,
        "b_correct_a_wrong": b_only,
        "discordant": discordant,
        "p_value_two_sided": p_value,
    }
