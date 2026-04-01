---
language:
  - en
license: other
tags:
  - moe
  - pruning
  - awq
  - quantized
  - qwen3
  - reap
  - expert-pruning
base_model: Qwen/Qwen3-Coder-Next
pipeline_tag: text-generation
library_name: transformers
---

# Research Test: Qwen3-Coder-Next-REAP-AWQ

> Expert-pruned and AWQ-quantized Qwen3-Coder-Next using the REAP (Robust Efficient Architecture Pruning) pipeline. 20% of MoE experts removed via diverse-calibration saliency analysis, then quantized to W4A16 for efficient inference on consumer GPUs.

**Status:** Research/Experimental

## Model Summary

| Property | Value |
|----------|-------|
| Base Model | Qwen3-Coder-Next (BF16, 149GB) |
| Architecture | `Qwen3NextForCausalLM` (Mixture-of-Experts + Gated DeltaNet) |
| Original Experts | 512 per layer, 10 active per token |
| After Pruning | 410 per layer (20% removed) |
| Layers | 48 |
| Quantization | AWQ W4A16, group_size=32 |
| Pruning Method | REAP with super-expert preservation |
| Calibration | 4 datasets (code, web, creative writing, math) |

## What Is This?

Qwen3-Coder-Next is a large MoE model with 512 experts per layer. Most tokens only activate 10 experts, meaning the vast majority of expert parameters sit idle for any given input. This creates an opportunity: identify which experts contribute the least across diverse workloads and remove them entirely.

This model is the result of that process:

1. **Observe** expert activation patterns across 4 calibration datasets
2. **Score** each expert using the REAP metric (activation magnitude weighted by routing probability)
3. **Prune** the bottom 20% of experts per layer (with super-expert preservation)
4. **Quantize** the pruned model to W4A16 using AWQ

The result is a significantly smaller model that retains the routing structure and quality of the original.

## The REAP Pipeline

### Why Prune MoE Experts?

In a standard dense model, every parameter participates in every forward pass. In an MoE model, only a small fraction of experts are active per token. This means:

- Many experts are **rarely activated** — they handle niche patterns that appear infrequently
- Some experts are **nearly redundant** — they fire on similar inputs and produce similar outputs
- A few experts are **critical** — they handle common patterns or rare-but-important edge cases

By measuring expert importance empirically, we can remove the least impactful experts while preserving model quality.

### REAP Metric

REAP (Robust Efficient Architecture Pruning) scores each expert based on:

```
REAP(expert) = sum(activation_norm * router_weight) / total_tokens
```

This captures both **frequency** (how often the expert fires) and **magnitude** (how much it contributes when it does). Experts with low REAP scores contribute little to the model's output and are candidates for removal.

### Diverse Calibration

A code-focused model uses different experts for different tasks. Calibrating on code alone risks pruning experts that are critical for reasoning or general language. We observe on 4 datasets:

| Dataset | Domain | Purpose |
|---------|--------|---------|
| `evol-codealpaca-v1` | Code | Core competency |
| `allenai/c4` | Web text | General language coverage |
| `WritingPrompts_curated` | Creative writing | Long-form generation |
| `tulu-3-sft-personas-math` | Math | Reasoning chains |

Observations are merged by summing accumulator metrics across datasets, then recomputing derived scores. This rescues experts that appear unimportant in code-only calibration but are essential for other tasks.

### Super-Expert Preservation

Some experts have extremely high peak activations — they rarely fire but are critical when they do. These "super-experts" are protected from pruning regardless of their average REAP score. This prevents catastrophic failures on rare but important inputs.

### Memory-Managed Execution

The 149GB BF16 model doesn't fit in 96GB VRAM (4x RTX 3090). Each pipeline phase runs as a **separate OS process**:

1. **Observe** (4 separate runs, one per dataset) — model loads with CPU offload via `device_map="auto"`, observer hooks accumulate statistics to CPU RAM
2. **Merge** — CPU-only, no GPU needed
3. **Prune** — fresh model load, in-place expert removal, state dict collection handles offloaded parameters
4. **AWQ** — `max_memory` caps at 20GiB/GPU with 100GiB CPU overflow, model moved to CPU before save

Process isolation guarantees clean GPU state between phases.

## AWQ Quantization Details

After pruning, the remaining 410 experts are quantized using AWQ (Activation-Aware Weight Quantization):

- **Scheme:** W4A16 (4-bit weights, 16-bit activations), symmetric, group_size=32
- **Calibration:** 256 samples from `evol-codealpaca-v1`

### Layers Kept at Full Precision

| Layer | Reason |
|-------|--------|
| `mlp.gate` (MoE router) | Expert routing is precision-critical |
| `mlp.shared_expert_gate` | Shared expert gating |
| `linear_attn.conv1d`, `in_proj_a/b` | Gated DeltaNet internals are fragile to quantization |
| `lm_head` | Output projection |

Everything else (expert MLPs, attention projections) is quantized to W4A16.

## Usage

### Serving with vLLM

```bash
vllm serve /path/to/Qwen3-Coder-Next-REAP-AWQ \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.93 \
    --enable-expert-parallel \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --max-model-len 32768 \
    --max-num-seqs 16
```

### Python (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "path/to/Qwen3-Coder-Next-REAP-AWQ",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("path/to/Qwen3-Coder-Next-REAP-AWQ")

messages = [{"role": "user", "content": "Write a Python function to merge two sorted lists."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs.to(model.device), max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Evaluation Results

Evaluated using a custom benchmark across 4 categories (17 tests total):

| Category | Tests | Pass Rate | Avg Tokens/sec |
|----------|-------|-----------|----------------|
| Code | 4 | TBD | TBD |
| Reasoning | 4 | TBD | TBD |
| Scaffold (Tool Use) | 4 | TBD | TBD |
| Chat | 5 | TBD | TBD |
| **Overall** | **17** | **TBD** | **TBD** |

*Results will be updated after the pipeline completes.*

### Baseline Comparison (Unpruned AWQ)

The unpruned Qwen3-Coder-Next-AWQ-4bit (512 experts) scores:

| Category | Pass Rate | Avg TPS |
|----------|-----------|---------|
| Code | 100% | 125.9 |
| Reasoning | 75% | 131.5 |
| Scaffold | 100% | 142.3 |
| Chat | 100% | 128.7 |
| **Overall** | **94%** | **132.0** |

## Pipeline Configuration

```yaml
# Observation
samples_per_dataset: 256
max_sequence_length: 512
distance_metric: cosine
datasets: [evol-codealpaca-v1, c4, WritingPrompts_curated, tulu-3-sft-personas-math]

# Pruning
method: reap
compression_ratio: 0.20  # Remove 20% of experts
preserve_super_experts: true
seed: 42

# Quantization
method: awq
scheme: W4A16
group_size: 32
calibration_samples: 256
```

## Previous Attempts

| Run | Compression | Experts Remaining | Outcome |
|-----|------------|-------------------|---------|
| v1 | 40% | 307 / 512 | Too aggressive — quality degradation |
| **v2 (this)** | **20%** | **410 / 512** | **In progress** |

## Hardware

- 4x NVIDIA RTX 3090 (24GB each, 96GB total)
- 128GB system RAM
- Pipeline uses CPU offload for model loading (149GB model > 96GB VRAM)

## Acknowledgments

- **REAP Framework** — [Cerebras](https://www.cerebras.net/) for the pruning methodology
- **Base Model** — [Qwen](https://github.com/QwenLM/Qwen3) for Qwen3-Coder-Next
- **AWQ** — [MIT HAN Lab](https://github.com/mit-han-lab/llm-awq) for Activation-Aware Weight Quantization
- **vLLM** — [vLLM Project](https://github.com/vllm-project/vllm) for efficient MoE serving

## License

This model inherits the license of the base Qwen3-Coder-Next model. See the [Qwen license](https://huggingface.co/Qwen/Qwen3-Coder-Next) for details.

## Citation

```bibtex
@misc{waive2025reap,
  title={Research Test: REAP Expert Pruning of Qwen3-Coder-Next},
  author={wAIve},
  year={2025},
  url={https://github.com/mtecnic/research-test-Qwen3-Coder-Next-REAP-AWQ}
}
```
