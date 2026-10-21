"""Validation-calibrated prediction sets and feature-space OOD detection."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


class ValidationCalibratedDetector:
    """Calibrate label sets and species feature distances from held-out data."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        self.alpha = alpha
        self.classes: list[str] = []
        self.centers: dict[str, list[float]] = {}
        self.scale: list[float] = []
        self.distance_thresholds: dict[str, float] = {}
        self.distance_calibration_counts: dict[str, int] = {}
        self.probability_scores: dict[str, dict[str, list[float]]] = {}
        self.probability_temperatures: dict[str, float] = {}
        self.minimum_sharpness: float | None = None

    @staticmethod
    def _fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if (
            probabilities.ndim != 2
            or len(probabilities) != len(labels)
            or not len(labels)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities.sum(axis=1) <= 0)
            or np.any(labels < 0)
            or np.any(labels >= probabilities.shape[1])
        ):
            raise ValueError("Cannot calibrate invalid validation probabilities.")

        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
        temperatures = np.logspace(-2, 2, 401)
        best_temperature = 1.0
        best_loss = float("inf")
        for temperature in temperatures:
            scaled_logits = logits / temperature
            scaled_logits -= np.max(scaled_logits, axis=1, keepdims=True)
            log_normalizer = np.log(np.exp(scaled_logits).sum(axis=1))
            loss = float(np.mean(
                log_normalizer - scaled_logits[np.arange(len(labels)), labels]
            ))
            if loss < best_loss:
                best_loss = loss
                best_temperature = float(temperature)
        return best_temperature

    def calibrate_probabilities(
        self,
        probabilities: np.ndarray,
        target: str,
    ) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if (
            probabilities.ndim != 2
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities.sum(axis=1) <= 0)
        ):
            raise ValueError("Prediction probabilities are invalid.")
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        temperature = self.probability_temperatures.get(target, 1.0)
        logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def has_probability_calibration(self, target: str) -> bool:
        return target in self.probability_temperatures

    @staticmethod
    def _upper_quantile(values: np.ndarray, alpha: float) -> float:
        ordered = np.sort(np.asarray(values, dtype=np.float64))
        rank = math.ceil((len(ordered) + 1) * (1 - alpha)) - 1
        return float(ordered[min(len(ordered) - 1, max(0, rank))])

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: dict[str, np.ndarray],
        validation_probabilities: dict[str, np.ndarray],
        probability_classes: dict[str, Sequence[str]],
        validation_sharpness: np.ndarray,
        classes: Sequence[str],
    ) -> None:
        train_features = np.asarray(train_features, dtype=np.float64)
        validation_features = np.asarray(validation_features, dtype=np.float64)
        train_labels = np.asarray(train_labels)
        validation_labels = {
            target: np.asarray(labels)
            for target, labels in validation_labels.items()
        }
        self.classes = list(classes)

        if train_features.ndim != 2 or validation_features.ndim != 2:
            raise ValueError("Detector features must be two-dimensional arrays.")
        if train_features.shape[1] != validation_features.shape[1]:
            raise ValueError("Training and validation feature widths do not match.")
        if not np.isfinite(train_features).all() or not np.isfinite(validation_features).all():
            raise ValueError("Detector features contain non-finite values.")
        if (
            len(train_labels) != len(train_features)
            or len(validation_labels.get("species", [])) != len(validation_features)
        ):
            raise ValueError("Species feature and label counts do not match.")
        if not self.classes:
            raise ValueError("At least one supported class is required.")
        validation_sharpness = np.asarray(validation_sharpness, dtype=np.float64)
        if (
            validation_sharpness.shape != (len(validation_features),)
            or not np.isfinite(validation_sharpness).all()
        ):
            raise ValueError("Validation sharpness values are missing or invalid.")
        self.minimum_sharpness = float(np.min(validation_sharpness))

        raw_scale = np.std(train_features, axis=0)
        nonzero_scale = raw_scale[raw_scale > 1e-12]
        scale_floor = float(np.median(nonzero_scale) * 1e-3) if len(nonzero_scale) else 1e-6
        scale = np.maximum(raw_scale, max(scale_floor, 1e-6))
        self.scale = scale.tolist()

        for class_index, class_name in enumerate(self.classes):
            class_train = train_features[train_labels == class_index]
            class_validation = validation_features[
                validation_labels["species"] == class_index
            ]
            if not len(class_train) or not len(class_validation):
                raise ValueError(
                    f"Cannot calibrate unknown detection for {class_name}: "
                    "training and validation examples are both required."
                )

            center = np.mean(class_train, axis=0)
            self.centers[class_name] = center.tolist()
            distances = np.sqrt(np.mean(np.square((class_validation - center) / scale), axis=1))
            self.distance_thresholds[class_name] = self._upper_quantile(distances, self.alpha)
            self.distance_calibration_counts[class_name] = int(len(class_validation))

        for target, probabilities in validation_probabilities.items():
            probabilities = np.asarray(probabilities, dtype=np.float64)
            target_labels = validation_labels.get(target)
            target_classes = list(probability_classes.get(target, ()))
            if (
                target_labels is None
                or probabilities.ndim != 2
                or probabilities.shape != (len(target_labels), len(target_classes))
            ):
                raise ValueError(f"Invalid validation probability shape for {target}.")
            if np.any(target_labels < 0) or np.any(target_labels >= len(target_classes)):
                raise ValueError(f"Validation labels are outside the supported classes for {target}.")
            if (
                not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or np.any(probabilities.sum(axis=1) <= 0)
            ):
                raise ValueError(f"Invalid validation probabilities for {target}.")

            self.probability_temperatures[target] = self._fit_temperature(
                probabilities,
                target_labels,
            )
            probabilities = self.calibrate_probabilities(probabilities, target)
            target_scores: dict[str, list[float]] = {}
            for class_index, class_name in enumerate(target_classes):
                matching = target_labels == class_index
                if not np.any(matching):
                    raise ValueError(
                        f"Cannot calibrate {target} label {class_name}: "
                        "no validation examples are available."
                    )
                target_scores[class_name] = (
                    1.0 - probabilities[matching, class_index]
                ).tolist()
            self.probability_scores[target] = target_scores

    def class_distance(self, features: np.ndarray, class_name: str) -> float:
        if class_name not in self.centers:
            raise ValueError(f"No distance calibration exists for class {class_name}.")
        vector = np.asarray(features, dtype=np.float64).reshape(-1)
        center = np.asarray(self.centers[class_name], dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if vector.shape != center.shape or not np.isfinite(vector).all():
            raise ValueError("Prediction features have an invalid shape or non-finite values.")
        return float(np.sqrt(np.mean(np.square((vector - center) / scale))))

    def is_out_of_distribution(self, features: np.ndarray, class_name: str) -> tuple[bool, float]:
        distance = self.class_distance(features, class_name)
        return distance > self.distance_thresholds[class_name], distance

    @staticmethod
    def image_sharpness(image: np.ndarray) -> float:
        rgb = np.asarray(image, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or min(rgb.shape[:2]) < 3:
            raise ValueError("Image is too small to measure sharpness.")
        gray = rgb[:, :, :3] @ np.array([0.299, 0.587, 0.114])
        laplacian = (
            gray[1:-1, :-2]
            + gray[1:-1, 2:]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            - 4 * gray[1:-1, 1:-1]
        )
        return float(np.var(laplacian))

    def is_too_blurry(self, sharpness: float) -> bool:
        if self.minimum_sharpness is None or not np.isfinite(sharpness):
            raise ValueError("Image-quality calibration is not available.")
        return sharpness < self.minimum_sharpness

    def prediction_set(self, probabilities: np.ndarray, target: str) -> list[str]:
        probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        calibration = self.probability_scores.get(target)
        if calibration is None:
            raise ValueError(f"No validation calibration is available for {target}.")
        class_names = list(calibration)
        pooled_scores = np.concatenate([
            np.asarray(scores, dtype=np.float64)
            for scores in calibration.values()
            if len(scores)
        ]) if any(calibration.values()) else np.asarray([], dtype=np.float64)
        if len(probabilities) != len(class_names) or not len(pooled_scores):
            raise ValueError(f"No compatible validation calibration is available for {target}.")
        if not np.isfinite(probabilities).all() or not np.isfinite(pooled_scores).all():
            raise ValueError("Prediction probabilities contain non-finite values.")

        included = []
        for class_index, class_name in enumerate(class_names):
            p_value = (
                1 + int(np.count_nonzero(pooled_scores >= 1.0 - probabilities[class_index]))
            ) / (len(pooled_scores) + 1)
            if p_value > self.alpha:
                included.append(class_name)
        return included

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "alpha": self.alpha,
            "classes": self.classes,
            "centers": self.centers,
            "scale": self.scale,
            "distance_thresholds": self.distance_thresholds,
            "distance_calibration_counts": self.distance_calibration_counts,
            "probability_scores": self.probability_scores,
            "probability_temperatures": self.probability_temperatures,
            "minimum_sharpness": self.minimum_sharpness,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValidationCalibratedDetector:
        if payload.get("version") != 1:
            raise ValueError("Unsupported unknown-detector artifact version.")
        detector = cls(alpha=float(payload["alpha"]))
        detector.classes = list(payload["classes"])
        detector.centers = {
            str(name): list(map(float, values))
            for name, values in payload["centers"].items()
        }
        detector.scale = list(map(float, payload["scale"]))
        detector.distance_thresholds = {
            str(name): float(value)
            for name, value in payload["distance_thresholds"].items()
        }
        detector.distance_calibration_counts = {
            str(name): int(value)
            for name, value in payload["distance_calibration_counts"].items()
        }
        detector.probability_scores = {
            str(target): {
                str(name): list(map(float, scores))
                for name, scores in target_scores.items()
            }
            for target, target_scores in payload["probability_scores"].items()
        }
        detector.probability_temperatures = {
            str(target): float(temperature)
            for target, temperature in payload.get("probability_temperatures", {}).items()
        }
        detector.minimum_sharpness = float(payload["minimum_sharpness"])
        if (
            not detector.classes
            or set(detector.classes) != set(detector.centers)
            or set(detector.classes) != set(detector.distance_thresholds)
            or len(detector.scale) != len(next(iter(detector.centers.values())))
            or not np.isfinite(detector.minimum_sharpness)
            or any(
                not np.isfinite(temperature) or temperature <= 0
                for temperature in detector.probability_temperatures.values()
            )
        ):
            raise ValueError("Unknown-detector artifact is incomplete or inconsistent.")
        return detector
