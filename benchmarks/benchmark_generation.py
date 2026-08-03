from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig

from minidecode.config import MiniDecodeConfig
from minidecode.kv_cache import ContiguousKVCache
from minidecode.model import MiniDecodeForCausalLM


MIB = 1024**2


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str
    prompt_length: int
    new_tokens: int
    ttft_ms: float
    average_tpot_ms: float
    tokens_per_second: float
    cache_memory_mib: float
    peak_memory_mib: float


def resolve_model_path(model_path: str) -> Path:
    local_path = Path(model_path).expanduser()
    if local_path.exists():
        return local_path.resolve()

    return Path(
        snapshot_download(repo_id=model_path, local_files_only=True)
    ).resolve()


def load_model(
    model_path: Path,
    device: torch.device,
) -> MiniDecodeForCausalLM:
    hf_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config = MiniDecodeConfig.from_hf_config(hf_config)
    model = MiniDecodeForCausalLM(config).to(
        device=device,
        dtype=torch.bfloat16,
    )
    state_dict = load_file(model_path / "model.safetensors", device=str(device))
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()
    torch.cuda.empty_cache()
    return model.eval()


def make_cache(
    model: MiniDecodeForCausalLM,
    input_ids: torch.Tensor,
    capacity: int,
) -> ContiguousKVCache:
    return ContiguousKVCache(
        num_layers=model.config.num_hidden_layers,
        batch_size=input_ids.shape[0],
        num_kv_heads=model.config.num_key_value_heads,
        max_seq_len=capacity,
        head_dim=model.config.head_dim,
        dtype=model.lm_head.weight.dtype,
        device=input_ids.device,
    )


def cache_memory_mib(cache: ContiguousKVCache) -> float:
    num_bytes = (
        cache.k_cache.numel() * cache.k_cache.element_size()
        + cache.v_cache.numel() * cache.v_cache.element_size()
    )
    return num_bytes / MIB


def elapsed_times_ms(
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]],
) -> list[float]:
    return [start.elapsed_time(end) for start, end in event_pairs]


@torch.inference_mode()
def run_cached_once(
    model: MiniDecodeForCausalLM,
    input_ids: torch.Tensor,
    cache: ContiguousKVCache,
    new_tokens: int,
    record_timing: bool,
) -> tuple[
    torch.Tensor,
    tuple[torch.cuda.Event, torch.cuda.Event] | None,
    list[tuple[torch.cuda.Event, torch.cuda.Event]],
]:
    cache.reset()
    model_input = input_ids
    generated_tokens: list[torch.Tensor] = []
    ttft_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    decode_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    for step in range(new_tokens):
        if record_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        logits = model(model_input, cache, last_token_only=True)
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)

        if record_timing:
            end.record()
            if step == 0:
                ttft_events = (start, end)
            else:
                decode_events.append((start, end))

        generated_tokens.append(next_id)
        model_input = next_id

    generated = torch.cat([input_ids, *generated_tokens], dim=1)
    return generated, ttft_events, decode_events


@torch.inference_mode()
def run_recompute_once(
    model: MiniDecodeForCausalLM,
    input_ids: torch.Tensor,
    cache: ContiguousKVCache,
    new_tokens: int,
    record_timing: bool,
) -> tuple[
    torch.Tensor,
    tuple[torch.cuda.Event, torch.cuda.Event] | None,
    list[tuple[torch.cuda.Event, torch.cuda.Event]],
]:
    generated = input_ids
    ttft_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    decode_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    for step in range(new_tokens):
        cache.reset()
        if record_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        logits = model(generated, cache, last_token_only=True)
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)

        if record_timing:
            end.record()
            if step == 0:
                ttft_events = (start, end)
            else:
                decode_events.append((start, end))

        generated = torch.cat([generated, next_id], dim=1)

    return generated, ttft_events, decode_events


def benchmark_mode(
    mode: str,
    model: MiniDecodeForCausalLM,
    input_ids: torch.Tensor,
    new_tokens: int,
    warmup: int,
    repeats: int,
) -> tuple[BenchmarkResult, torch.Tensor]:
    run_once = run_cached_once if mode == "cached" else run_recompute_once
    capacity = input_ids.shape[1] + new_tokens
    cache = make_cache(model, input_ids, capacity)

    for _ in range(warmup):
        run_once(model, input_ids, cache, new_tokens, record_timing=False)
    torch.cuda.synchronize(input_ids.device)

    torch.cuda.reset_peak_memory_stats(input_ids.device)
    ttft_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    decode_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    generated = input_ids

    for _ in range(repeats):
        generated, ttft_events, decode_events = run_once(
            model,
            input_ids,
            cache,
            new_tokens,
            record_timing=True,
        )
        if ttft_events is None:
            raise RuntimeError("TTFT events were not recorded")
        ttft_event_pairs.append(ttft_events)
        decode_event_pairs.extend(decode_events)

    torch.cuda.synchronize(input_ids.device)
    ttft_values = elapsed_times_ms(ttft_event_pairs)
    tpot_values = elapsed_times_ms(decode_event_pairs)
    average_ttft_ms = sum(ttft_values) / len(ttft_values)
    average_tpot_ms = sum(tpot_values) / len(tpot_values)

    result = BenchmarkResult(
        mode=mode,
        prompt_length=input_ids.shape[1],
        new_tokens=new_tokens,
        ttft_ms=average_ttft_ms,
        average_tpot_ms=average_tpot_ms,
        tokens_per_second=input_ids.shape[0] * 1000.0 / average_tpot_ms,
        cache_memory_mib=cache_memory_mib(cache),
        peak_memory_mib=torch.cuda.max_memory_allocated(input_ids.device) / MIB,
    )
    return result, generated


def print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'mode':<12} {'prompt':>8} {'new':>6} {'TTFT ms':>12} "
        f"{'TPOT ms':>12} {'tokens/s':>12} {'cache MiB':>12} {'peak MiB':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.mode:<12} "
            f"{result.prompt_length:>8d} "
            f"{result.new_tokens:>6d} "
            f"{result.ttft_ms:>12.3f} "
            f"{result.average_tpot_ms:>12.3f} "
            f"{result.tokens_per_second:>12.2f} "
            f"{result.cache_memory_mib:>12.2f} "
            f"{result.peak_memory_mib:>12.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MiniDecode full recomputation and contiguous KV cache."
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if args.new_tokens < 2:
        raise ValueError("new_tokens must be at least 2 to measure TPOT")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model_path = resolve_model_path(args.model_path)
    model = load_model(model_path, device)
    input_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(1, args.prompt_length),
        dtype=torch.long,
        device=device,
    )

    cached_result, cached_output = benchmark_mode(
        "cached",
        model,
        input_ids,
        args.new_tokens,
        args.warmup,
        args.repeats,
    )
    recompute_result, recompute_output = benchmark_mode(
        "recompute",
        model,
        input_ids,
        args.new_tokens,
        args.warmup,
        args.repeats,
    )

    if not torch.equal(cached_output, recompute_output):
        raise RuntimeError("cached and recompute generation produced different tokens")

    print(f"model: {model_path}")
    print(f"device: {torch.cuda.get_device_name(device)}")
    print_results([recompute_result, cached_result])


if __name__ == "__main__":
    main()
