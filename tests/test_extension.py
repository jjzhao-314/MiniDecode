import pytest
import torch

from minidecode import _C


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def test_add_one_matches_torch() -> None:
    input_tensor = torch.randn(1025, device="cuda", dtype=torch.float32)

    actual = _C.add_one(input_tensor)

    torch.testing.assert_close(actual, input_tensor + 1.0)


def test_add_one_handles_empty_tensor() -> None:
    input_tensor = torch.empty(0, device="cuda", dtype=torch.float32)

    actual = _C.add_one(input_tensor)

    assert actual.shape == input_tensor.shape
    assert actual.dtype == input_tensor.dtype
    assert actual.device == input_tensor.device
    assert actual.numel() == 0


def test_add_one_rejects_non_contiguous_tensor() -> None:
    input_tensor = torch.randn(3, 4, device="cuda", dtype=torch.float32).T
    assert not input_tensor.is_contiguous()

    with pytest.raises(RuntimeError, match="input must be contiguous"):
        _C.add_one(input_tensor)


def test_add_one_rejects_cpu_tensor() -> None:
    input_tensor = torch.randn(8, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="input must be a CUDA tensor"):
        _C.add_one(input_tensor)


def test_add_one_rejects_unsupported_dtype() -> None:
    input_tensor = torch.randn(8, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="input must have dtype float32"):
        _C.add_one(input_tensor)
