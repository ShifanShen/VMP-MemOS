"""Lightweight learned policy models for operation prediction.

The first learned policy intentionally stays small and dependency-free: it is a
multiclass logistic regression model trained with deterministic gradient
descent over the canonical ``PolicyFeatures`` vector.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import Field, FiniteFloat, JsonValue

from vmp_memos.schemas import OperationType, PolicyFeatures
from vmp_memos.schemas.base import NonEmptyStr, SchemaModel, Score, TimestampedSchema, new_id

DEFAULT_POLICY_LABELS: tuple[OperationType, ...] = (
    OperationType.ADD,
    OperationType.UPDATE,
    OperationType.MERGE,
    OperationType.ARCHIVE,
    OperationType.RETRIEVE,
    OperationType.IGNORE,
)


class PolicyTrainingExample(TimestampedSchema):
    """One supervised operation-label example for the learned policy."""

    example_id: NonEmptyStr = Field(default_factory=lambda: new_id("pex"), frozen=True)
    features: dict[NonEmptyStr, Score]
    label: OperationType
    sample_id: NonEmptyStr | None = None
    memory_id: NonEmptyStr | None = None
    source: NonEmptyStr = "benchmark"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    def as_vector(self, feature_names: Sequence[str] | None = None) -> list[float]:
        """Return features in a stable order."""

        names = tuple(feature_names or PolicyFeatures.FEATURE_NAMES)
        return [float(self.features.get(name, 0.0)) for name in names]


class LearnedPolicyPrediction(SchemaModel):
    """Probability distribution emitted by a learned policy model."""

    predicted_op: OperationType
    probabilities: dict[NonEmptyStr, Score]
    confidence: Score
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class LogisticPolicyModel(SchemaModel):
    """Multiclass logistic regression over ``PolicyFeatures``."""

    feature_names: list[NonEmptyStr] = Field(
        default_factory=lambda: list(PolicyFeatures.FEATURE_NAMES)
    )
    labels: list[OperationType] = Field(default_factory=lambda: list(DEFAULT_POLICY_LABELS))
    weights: dict[NonEmptyStr, list[FiniteFloat]]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def train(
        cls,
        examples: Sequence[PolicyTrainingExample],
        *,
        labels: Sequence[OperationType] | None = None,
        epochs: int = 500,
        learning_rate: float = 0.4,
        l2: float = 0.001,
    ) -> "LogisticPolicyModel":
        """Train a deterministic softmax classifier."""

        if not examples:
            raise ValueError("at least one training example is required")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if l2 < 0.0:
            raise ValueError("l2 must be non-negative")

        feature_names = list(PolicyFeatures.FEATURE_NAMES)
        label_values = _ordered_labels(examples, labels)
        weights = {
            label.value: [0.0 for _ in range(len(feature_names) + 1)]
            for label in label_values
        }

        for epoch in range(epochs):
            step_size = learning_rate / math.sqrt(epoch + 1.0)
            for example in examples:
                vector = [1.0, *example.as_vector(feature_names)]
                probabilities = _softmax_logits(weights, vector)
                for label in label_values:
                    label_name = label.value
                    target = 1.0 if example.label == label else 0.0
                    error = target - probabilities[label_name]
                    row = weights[label_name]
                    for index, value in enumerate(vector):
                        penalty = 0.0 if index == 0 else l2 * row[index]
                        row[index] += step_size * (error * value - penalty)

        return cls(
            feature_names=feature_names,
            labels=label_values,
            weights=weights,
            metadata={
                "model_type": "multiclass_logistic_regression",
                "num_examples": len(examples),
                "epochs": epochs,
                "learning_rate": learning_rate,
                "l2": l2,
                "label_counts": _label_counts(examples),
            },
        )

    def predict(
        self,
        features: PolicyFeatures | Mapping[str, float] | Sequence[float],
    ) -> LearnedPolicyPrediction:
        """Predict the most likely operation and full probability distribution."""

        probabilities = self.predict_proba(features)
        predicted_name, confidence = max(
            probabilities.items(),
            key=lambda item: (item[1], item[0]),
        )
        return LearnedPolicyPrediction(
            predicted_op=OperationType(predicted_name),
            probabilities=probabilities,
            confidence=confidence,
            metadata={"model_type": str(self.metadata.get("model_type", "unknown"))},
        )

    def predict_proba(
        self,
        features: PolicyFeatures | Mapping[str, float] | Sequence[float],
    ) -> dict[str, float]:
        """Return operation probabilities for a feature vector."""

        vector = [1.0, *self._feature_vector(features)]
        return _softmax_logits(self.weights, vector)

    def feature_weights(self, label: OperationType | str) -> dict[str, float]:
        """Return per-feature weights for one operation label."""

        label_name = label.value if isinstance(label, OperationType) else str(label)
        row = self.weights[label_name]
        return {
            feature_name: float(row[index + 1])
            for index, feature_name in enumerate(self.feature_names)
        }

    def save(self, path: str | Path) -> Path:
        """Persist the model as UTF-8 JSON."""

        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "LogisticPolicyModel":
        """Load a persisted model."""

        model_path = Path(path).expanduser().resolve()
        return cls.model_validate_json(model_path.read_text(encoding="utf-8"))

    def _feature_vector(
        self,
        features: PolicyFeatures | Mapping[str, float] | Sequence[float],
    ) -> list[float]:
        if isinstance(features, PolicyFeatures):
            return [float(getattr(features, name)) for name in self.feature_names]
        if isinstance(features, Mapping):
            return [
                _clamp01(float(features.get(name, 0.0)))
                for name in self.feature_names
            ]
        values = [float(value) for value in features]
        if len(values) != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, got {len(values)}"
            )
        return [_clamp01(value) for value in values]


def features_to_mapping(features: PolicyFeatures) -> dict[str, float]:
    """Serialize ``PolicyFeatures`` into the canonical training mapping."""

    return {
        name: float(getattr(features, name))
        for name in PolicyFeatures.FEATURE_NAMES
    }


def _ordered_labels(
    examples: Sequence[PolicyTrainingExample],
    labels: Sequence[OperationType] | None,
) -> list[OperationType]:
    ordered: list[OperationType] = []
    for label in labels or DEFAULT_POLICY_LABELS:
        if label not in ordered:
            ordered.append(label)
    for example in examples:
        if example.label not in ordered:
            ordered.append(example.label)
    return ordered


def _label_counts(examples: Sequence[PolicyTrainingExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.label.value] = counts.get(example.label.value, 0) + 1
    return counts


def _softmax_logits(
    weights: Mapping[str, Sequence[float]],
    vector: Sequence[float],
) -> dict[str, float]:
    logits: dict[str, float] = {}
    for label, row in weights.items():
        if len(row) != len(vector):
            raise ValueError(
                f"weight vector for {label} has length {len(row)}, "
                f"expected {len(vector)}"
            )
        logits[label] = sum(weight * value for weight, value in zip(row, vector, strict=True))

    max_logit = max(logits.values()) if logits else 0.0
    exp_values = {
        label: math.exp(_finite(logit - max_logit))
        for label, logit in logits.items()
    }
    denominator = sum(exp_values.values())
    if denominator <= 0.0:
        fallback = 1.0 / max(1, len(exp_values))
        return {label: fallback for label in exp_values}
    return {
        label: _clamp01(value / denominator)
        for label, value in exp_values.items()
    }


def _finite(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("learned policy value must be finite")
    return numeric


def _clamp01(value: float) -> float:
    numeric = _finite(value)
    return min(1.0, max(0.0, numeric))
