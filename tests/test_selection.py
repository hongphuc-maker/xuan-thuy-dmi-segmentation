from xuanthuy_seg.training.selection import CheckpointSelector


def test_acceptance_gate_prevents_eight_class_model_from_becoming_best() -> None:
    selector = CheckpointSelector(
        {
            "checkpoint": {"metric": "validation_macro_f1_11", "direction": "max", "min_delta": 0.001},
            "stopping": {"rule": "patience", "patience_evaluations": 3},
        },
        {"required_predicted_classes": 11, "minimum_recall_by_code": {5: 0.05}},
    )
    rejected = selector.update(
        156,
        {"validation_macro_f1_11": 0.40, "predicted_classes": 8, "recall_by_code": {5: 0.0}},
    )
    assert not rejected.eligible
    assert not rejected.improved
    assert rejected.best_step is None
    assert rejected.no_improve_evaluations == 0
    assert not rejected.should_stop

    accepted = selector.update(
        312,
        {"validation_macro_f1_11": 0.35, "predicted_classes": 11, "recall_by_code": {5: 0.10}},
    )
    assert accepted.eligible
    assert accepted.improved
    assert accepted.best_step == 312


def test_patience_starts_only_after_first_accepted_checkpoint() -> None:
    selector = CheckpointSelector(
        {
            "checkpoint": {"metric": "validation_macro_f1_11", "direction": "max", "min_delta": 0.0},
            "stopping": {"rule": "patience", "patience_evaluations": 2},
        },
        {"required_predicted_classes": 11},
    )
    for step in (156, 312, 468):
        decision = selector.update(
            step,
            {"validation_macro_f1_11": 0.20, "predicted_classes": 8},
        )
        assert not decision.should_stop
        assert decision.no_improve_evaluations == 0
    selector.update(624, {"validation_macro_f1_11": 0.30, "predicted_classes": 11})
    selector.update(780, {"validation_macro_f1_11": 0.29, "predicted_classes": 11})
    assert selector.update(
        936,
        {"validation_macro_f1_11": 0.28, "predicted_classes": 10},
    ).should_stop


def test_loss_driven_maximum_steps_does_not_early_stop() -> None:
    selector = CheckpointSelector(
        {
            "checkpoint": {"metric": "fixed_panel_dmi_loss", "direction": "min", "min_delta": 0.0},
            "stopping": {"rule": "maximum_steps"},
        },
        {"required_predicted_classes": None, "minimum_recall_by_code": {}},
    )
    assert selector.update(100, {"fixed_panel_dmi_loss": 80.0}).improved
    decision = selector.update(200, {"fixed_panel_dmi_loss": 81.0})
    assert not decision.improved
    assert not decision.should_stop


def test_nonfinite_validation_dmi_loss_cannot_be_selected() -> None:
    selector = CheckpointSelector(
        {
            "checkpoint": {"metric": "validation_dmi_loss", "direction": "min", "min_delta": 0.0},
            "stopping": {"rule": "maximum_steps"},
        },
        {},
    )
    rejected = selector.update(0, {"validation_dmi_loss": float("inf")})
    assert not rejected.eligible
    assert not rejected.improved
    assert rejected.best_step is None
    assert "non-finite" in rejected.reason
    assert selector.update(156, {"validation_dmi_loss": 40.0}).improved
