import torch
from torch import nn

from minidecode.generation import greedy_generate


class PredictSequenceModel(nn.Module):
    def __init__(self, prompt_length: int, predicted_ids: list[int], vocab_size: int):
        super().__init__()
        self.prompt_length = prompt_length
        self.predicted_ids = predicted_ids
        self.vocab_size = vocab_size
        self.grad_enabled_during_forward: list[bool] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        step = input_ids.shape[1] - self.prompt_length
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
        )
        logits[:, -1, self.predicted_ids[step]] = 1.0
        return logits


def test_greedy_generate_appends_argmax_tokens() -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    model = PredictSequenceModel(
        prompt_length=3,
        predicted_ids=[4, 5, 6],
        vocab_size=8,
    )

    actual = greedy_generate(model, input_ids, max_new_tokens=3)

    expected = torch.tensor([[1, 2, 3, 4, 5, 6]])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(input_ids, torch.tensor([[1, 2, 3]]))
    assert model.grad_enabled_during_forward == [False, False, False]


def test_greedy_generate_stops_after_eos() -> None:
    input_ids = torch.tensor([[1, 2]])
    model = PredictSequenceModel(
        prompt_length=2,
        predicted_ids=[3, 7, 4, 5],
        vocab_size=8,
    )

    actual = greedy_generate(
        model,
        input_ids,
        max_new_tokens=4,
        eos_token_id=7,
    )

    expected = torch.tensor([[1, 2, 3, 7]])
    torch.testing.assert_close(actual, expected)
    assert model.grad_enabled_during_forward == [False, False]


def test_greedy_generate_handles_zero_new_tokens() -> None:
    input_ids = torch.tensor([[1, 2, 3]])
    model = PredictSequenceModel(
        prompt_length=3,
        predicted_ids=[],
        vocab_size=8,
    )

    actual = greedy_generate(model, input_ids, max_new_tokens=0)

    torch.testing.assert_close(actual, input_ids)
    assert model.grad_enabled_during_forward == []
