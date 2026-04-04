import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

parser = argparse.ArgumentParser(description="AWQ quantize a pruned REAP model")
parser.add_argument("--model-path", required=True, help="Path to pruned model")
parser.add_argument("--save-dir", required=True, help="Where to save quantized model")
args = parser.parse_args()

MODEL_PATH = args.model_path
SAVE_DIR = args.save_dir

print(f"Loading model from {MODEL_PATH}...")
# Reserve GPU headroom for AWQ grid search working memory
import torch
n_gpus = torch.cuda.device_count()
max_memory = {i: "20GiB" for i in range(n_gpus)}
max_memory["cpu"] = "100GiB"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="auto",
    trust_remote_code=True, max_memory=max_memory,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Calibration: same coding-focused dataset used for REAP pruning
print("Loading calibration dataset...")
ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train[:256]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            [{"role": "user", "content": example["instruction"]}],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)

# AWQ config: W4A16 symmetric (vLLM requirement for MoE)
# Ignore layers that are sensitive to quantization:
#   - lm_head: output head
#   - mlp.gate: MoE router (expert selection is precision-critical)
#   - mlp.shared_expert_gate: shared expert gating
#   - linear_attn conv1d/in_proj_a/in_proj_b: Gated DeltaNet is fragile
recipe = [
    AWQModifier(
        ignore=[
            "lm_head",
            "re:.*mlp.gate$",
            "re:.*mlp.shared_expert_gate$",
            "re:.*linear_attn[.]conv1d",
            "re:.*linear_attn[.]in_proj_a",
            "re:.*linear_attn[.]in_proj_b",
        ],
        scheme="W4A16",
        targets=["Linear"],
        offload_device=torch.device("cpu"),
        n_grid=12,
        duo_scaling=False,
    ),
]

print("Running AWQ quantization...")
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=256,
    num_calibration_samples=256,
    sequential_targets=["Qwen3NextDecoderLayer"],
)

# Offload model to CPU before saving to avoid GPU OOM during compression
print("Moving model to CPU for save...")
model = model.to("cpu")
torch.cuda.empty_cache()

# Save compressed
print(f"Saving quantized model to {SAVE_DIR}...")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
print("Done!")
