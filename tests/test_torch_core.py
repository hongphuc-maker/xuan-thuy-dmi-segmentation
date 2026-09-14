from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")

from xuanthuy_seg.contracts import sha256_file
from xuanthuy_seg.evaluation import cross_entropy_from_flat_probabilities
from xuanthuy_seg.losses.cross_entropy import deterministic_masked_cross_entropy
from xuanthuy_seg.losses.dmi import (
    dmi_from_flat_probabilities,
    dmi_from_probabilities,
    exact_dmi_loss,
)
from xuanthuy_seg.models.logit_additive_closing import LogitAdditiveClosing2d
from xuanthuy_seg.models.registry import build_model
from xuanthuy_seg.models.unet_bn import UNetBN
from xuanthuy_seg.training.experiment import _first_initialization


def test_manual_ce_matches_native_cpu_ce() -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 11, 5, 7, dtype=torch.float64)
    target = torch.randint(0, 11, (2, 5, 7), dtype=torch.long)
    target[0, 0, 0] = 255
    manual = deterministic_masked_cross_entropy(logits, target)
    native = F.cross_entropy(logits, target, ignore_index=255)
    assert torch.allclose(manual, native, rtol=1e-12, atol=1e-12)


def test_manual_weighted_ce_matches_native_cpu_ce() -> None:
    torch.manual_seed(8)
    logits = torch.randn(2, 11, 5, 7, dtype=torch.float64)
    target = torch.randint(0, 11, (2, 5, 7), dtype=torch.long)
    target[0, 0, 0] = 255
    weights = torch.linspace(0.4, 2.2, 11, dtype=torch.float64)
    manual = deterministic_masked_cross_entropy(
        logits,
        target,
        class_weights=weights,
    )
    native = F.cross_entropy(logits, target, ignore_index=255, weight=weights)
    assert torch.allclose(manual, native, rtol=1e-12, atol=1e-12)


def test_flat_probability_ce_uses_training_weighted_mean_convention() -> None:
    probabilities = torch.tensor(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]],
        dtype=torch.float64,
    )
    target = torch.tensor([0, 1, 2])
    weights = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    observed = cross_entropy_from_flat_probabilities(
        probabilities,
        target,
        class_weights=weights,
    )
    expected = (-torch.log(torch.tensor([0.8, 0.7, 0.7])) * weights).sum() / weights.sum()
    assert torch.allclose(observed, expected, rtol=1e-7, atol=1e-7)


def test_dmi_returns_full_rank_diagnostics() -> None:
    logits = torch.tensor(
        [
            [
                [[8.0, -2.0, -2.0]],
                [[-2.0, 8.0, -2.0]],
                [[-2.0, -2.0, 8.0]],
            ]
        ],
        requires_grad=True,
    )
    target = torch.tensor([[[0, 1, 2]]])
    loss, diagnostics = exact_dmi_loss(logits, target)
    assert diagnostics.rank == 3
    assert diagnostics.classes_present == 3
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_flat_dmi_matches_dense_dmi_for_one_global_validation_matrix() -> None:
    probabilities = torch.tensor(
        [
            [
                [[0.90, 0.05, 0.05]],
                [[0.05, 0.90, 0.05]],
                [[0.05, 0.05, 0.90]],
            ]
        ],
        dtype=torch.float64,
    )
    target = torch.tensor([[[0, 1, 2]]])
    dense_loss, dense_diagnostics = dmi_from_probabilities(probabilities, target)
    flat_loss, flat_diagnostics = dmi_from_flat_probabilities(
        probabilities.permute(0, 2, 3, 1).reshape(-1, 3),
        target.reshape(-1),
    )
    assert torch.allclose(flat_loss, dense_loss, rtol=0.0, atol=0.0)
    assert flat_diagnostics == dense_diagnostics


def test_unet_output_contract() -> None:
    model = UNetBN(in_channels=10, num_classes=11, base_channels=4)
    output = model(torch.randn(2, 10, 32, 32))
    assert output.shape == (2, 11, 32, 32)


def test_logit_additive_closing_preserves_constant_fields_with_flat_se() -> None:
    layer = LogitAdditiveClosing2d(num_classes=3, beta=10.0)
    inputs = torch.randn(2, 3, 1, 1).expand(2, 3, 9, 7).clone()
    observed = layer(inputs)
    assert torch.allclose(observed, inputs, rtol=1e-6, atol=1e-6)


def test_logit_additive_closing_model_supports_exact_dmi_backward() -> None:
    model = build_model(
        {
            "architecture": "custom",
            "factory": (
                "xuanthuy_seg.models.logit_additive_closing:"
                "build_unet_with_logit_additive_closing"
            ),
            "parameters": {
                "in_channels": 10,
                "num_classes": 11,
                "base_channels": 4,
                "kernel_size": 3,
                "beta": 10.0,
                "padding_mode": "replicate",
                "se_initialization": "zeros",
            },
        }
    )
    logits = model(torch.randn(2, 10, 32, 32))
    target = torch.randint(0, 11, (2, 32, 32))
    loss, diagnostics = exact_dmi_loss(logits, target)
    loss.backward()

    assert logits.shape == (2, 11, 32, 32)
    assert diagnostics.rank == 11
    assert torch.isfinite(loss)
    assert model.logit_closing.structuring_element.grad is not None
    assert torch.isfinite(model.logit_closing.structuring_element.grad).all()


def test_weighted_ce_checkpoint_can_initialize_only_morphology_backbone(tmp_path) -> None:
    source = UNetBN(in_channels=10, num_classes=11, base_channels=4)
    checkpoint = tmp_path / "weighted_ce.pt"
    torch.save({"model_state": source.state_dict()}, checkpoint)
    expected_sha = sha256_file(checkpoint)
    model = build_model(
        {
            "architecture": "custom",
            "factory": (
                "xuanthuy_seg.models.logit_additive_closing:"
                "build_unet_with_logit_additive_closing"
            ),
            "parameters": {
                "in_channels": 10,
                "num_classes": 11,
                "base_channels": 4,
            },
        }
    )
    bundle = SimpleNamespace(
        experiment={"seed": 20260910},
        resolved={
            "phases": [
                {
                    "initialization": {
                        "type": "checkpoint",
                        "sha256": expected_sha,
                        "target_module": "unet",
                    }
                }
            ]
        },
    )

    metadata = _first_initialization(bundle, model, checkpoint)

    assert metadata["target_module"] == "unet"
    for name, value in source.state_dict().items():
        assert torch.equal(model.unet.state_dict()[name], value)
    assert torch.count_nonzero(model.logit_closing.structuring_element) == 0
