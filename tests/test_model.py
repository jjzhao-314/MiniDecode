import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from minidecode.model import RMSNorm


@pytest.mark.parametrize("shape", [(2, 3, 1024), (2, 8, 5, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_matches_hugging_face(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    eps = 1e-6
    hidden_size = shape[-1]
    actual_norm = RMSNorm(hidden_size, eps=eps).to(dtype=dtype)
    expected_norm = Qwen3RMSNorm(hidden_size, eps=eps).to(dtype=dtype)

    weight = torch.randn(hidden_size, dtype=dtype)
    with torch.no_grad():
        actual_norm.weight.copy_(weight)
        expected_norm.weight.copy_(weight)

    input_tensor = torch.randn(shape, dtype=dtype)

    actual = actual_norm(input_tensor)
    expected = expected_norm(input_tensor)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == input_tensor.shape
    assert actual.dtype == input_tensor.dtype


@pytest.mark.parametrize("hidden_size", [128, 1024])
def test_rms_norm_handles_zero_input(hidden_size: int) -> None:
    norm = RMSNorm(hidden_size)
    input_tensor = torch.zeros(2, 3, hidden_size)

    output = norm(input_tensor)

    torch.testing.assert_close(output, torch.zeros_like(input_tensor))
    assert torch.isfinite(output).all()


def test_rms_norm_initializes_weight_to_one() -> None:
    norm = RMSNorm(128)

    torch.testing.assert_close(norm.weight, torch.ones(128))
