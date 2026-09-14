from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CheckpointDecision:
    eligible: bool
    improved: bool
    should_stop: bool
    metric: str
    current_value: float
    best_value: float | None
    best_step: int | None
    no_improve_evaluations: int
    reason: str


class CheckpointSelector:
    """State machine shared by macro-F1-driven and loss-driven experiments."""

    def __init__(self, selection_config: dict[str, Any], acceptance_config: dict[str, Any]):
        checkpoint = selection_config["checkpoint"]
        stopping = selection_config["stopping"]
        self.metric = str(checkpoint["metric"])
        self.direction = str(checkpoint["direction"])
        self.min_delta = float(checkpoint.get("min_delta", 0.0))
        self.stop_rule = str(stopping["rule"])
        self.patience = int(stopping.get("patience_evaluations", 0))
        self.required_predicted_classes = acceptance_config.get("required_predicted_classes")
        self.minimum_recall = {
            int(code): float(value)
            for code, value in acceptance_config.get("minimum_recall_by_code", {}).items()
        }
        self.best_value: float | None = None
        self.best_step: int | None = None
        self.no_improve = 0

    def _eligible(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        if self.required_predicted_classes is not None:
            predicted = int(metrics.get("predicted_classes", -1))
            if predicted < int(self.required_predicted_classes):
                return False, f"predicted_classes={predicted}"
        if self.minimum_recall:
            recalls = {int(key): float(value) for key, value in metrics.get("recall_by_code", {}).items()}
            failed = {
                code: recalls.get(code, 0.0)
                for code, minimum in self.minimum_recall.items()
                if recalls.get(code, 0.0) < minimum
            }
            if failed:
                return False, f"minimum recall failed: {failed}"
        return True, "acceptance pass"

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.direction == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def update(self, step: int, metrics: dict[str, Any]) -> CheckpointDecision:
        if self.metric not in metrics:
            raise KeyError(f"Missing checkpoint metric: {self.metric}")
        value = float(metrics[self.metric])
        if math.isfinite(value):
            eligible, reason = self._eligible(metrics)
        else:
            eligible, reason = False, f"non-finite {self.metric}={value}"
        improved = eligible and self._is_improvement(value)
        if improved:
            self.best_value = value
            self.best_step = int(step)
            self.no_improve = 0
        elif self.best_step is not None:
            self.no_improve += 1
        # Before the first accepted checkpoint there is nothing meaningful to
        # early-stop against. This matters for 11-class recovery experiments:
        # an initially collapsed model must be allowed to become eligible.
        should_stop = (
            self.stop_rule == "patience"
            and self.best_step is not None
            and self.no_improve >= self.patience
        )
        return CheckpointDecision(
            eligible=eligible,
            improved=improved,
            should_stop=should_stop,
            metric=self.metric,
            current_value=value,
            best_value=self.best_value,
            best_step=self.best_step,
            no_improve_evaluations=self.no_improve,
            reason=reason,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_value": self.best_value,
            "best_step": self.best_step,
            "no_improve": self.no_improve,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best_value = None if state["best_value"] is None else float(state["best_value"])
        self.best_step = None if state["best_step"] is None else int(state["best_step"])
        self.no_improve = int(state["no_improve"])
