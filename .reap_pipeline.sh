#!/bin/bash
set -e

# REAP 20% Prune Pipeline — Memory-Managed
# Each phase is a SEPARATE process to fully free GPU between steps.
#
# Phase 1: Observe 4 datasets (BF16 model, 4 separate invocations)
# Phase 2: Merge observations (CPU-only)
# Phase 3: Prune at 20% with combined calibration (fresh model load)
# Phase 4: AWQ quantize the pruned model (separate process)
# Phase 5: Eval (vLLM serve + model_ab_eval.py)

cd /data/reap
source .venv/bin/activate

MODEL=/data/models/Qwen3-Coder-Next
NUM_SAMPLES=256
MAX_LEN=512
OBS_FILE=observations_${NUM_SAMPLES}_cosine.pt
PRUNED_DIR=""  # Set after Phase 3

echo "============================================================"
echo "REAP Pipeline: Observe → Merge → Prune → AWQ → Eval"
echo "Model: $MODEL (BF16, 149GB)"
echo "Compression: 20% (prune 102 of 512 experts)"
echo "Samples: $NUM_SAMPLES @ ${MAX_LEN} tokens per dataset"
echo "============================================================"

# ============================================================
# Phase 1: Observe each dataset (separate process per dataset)
# ============================================================
declare -A DS_MAP=(
    ["evol-codealpaca-v1"]="theblackcat102/evol-codealpaca-v1"
    ["c4"]="allenai/c4"
    ["WritingPrompts_curated"]="euclaise/WritingPrompts_curated"
    ["tulu-3-sft-personas-math"]="allenai/tulu-3-sft-personas-math"
)

for short in evol-codealpaca-v1 c4 WritingPrompts_curated tulu-3-sft-personas-math; do
    full="${DS_MAP[$short]}"
    obs_path="artifacts/Qwen3-Coder-Next/${short}/all/${OBS_FILE}"

    if [ -f "$obs_path" ]; then
        echo ""
        echo "[Phase 1] SKIP $short — already observed"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "[Phase 1] Observing: $short ($full)"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/reap/prune.py \
        --model-name "$MODEL" \
        --dataset-name "$full" \
        --compression-ratio 0.2 \
        --prune-method reap \
        --profile false \
        --do-eval false \
        --run_observer_only true \
        --distance_measure cosine \
        --seed 42 \
        --output_file_name "$OBS_FILE" \
        --samples_per_category $NUM_SAMPLES \
        --model_max_length $MAX_LEN \
        --record_pruning_metrics_only true \
        --smoke_test false

    echo "[Phase 1] Done: $short → $obs_path"
    # Process exits here, GPU memory fully freed
done

# Verify all observations exist
echo ""
echo "============================================================"
echo "[Phase 1] Verification"
echo "============================================================"
MISSING=0
for short in evol-codealpaca-v1 c4 WritingPrompts_curated tulu-3-sft-personas-math; do
    obs_path="artifacts/Qwen3-Coder-Next/${short}/all/${OBS_FILE}"
    if [ -f "$obs_path" ]; then
        size=$(du -sh "$obs_path" | cut -f1)
        echo "  OK: $short ($size)"
    else
        echo "  MISSING: $short"
        MISSING=1
    fi
done
if [ "$MISSING" -eq 1 ]; then
    echo "ERROR: Some observations missing. Cannot proceed."
    exit 1
fi

# ============================================================
# Phase 2: Merge observations (CPU-only, no GPU needed)
# ============================================================
echo ""
echo "============================================================"
echo "[Phase 2] Merging observations from all datasets"
echo "============================================================"

python3 - <<'PYEOF'
import torch
import pathlib
import sys

artifacts = pathlib.Path("artifacts/Qwen3-Coder-Next")
obs_file = "observations_256_cosine.pt"

datasets = ["evol-codealpaca-v1", "c4", "WritingPrompts_curated", "tulu-3-sft-personas-math"]
all_obs = []

for ds in datasets:
    f = artifacts / ds / "all" / obs_file
    if f.exists():
        print(f"  Loading {ds}...")
        data = torch.load(f, map_location="cpu", weights_only=False)
        all_obs.append((ds, data))
        layer0 = list(data.keys())[0]
        print(f"    layers={len(data)}, total_tokens={data[layer0]['total_tokens']}")
    else:
        print(f"  MISSING: {f}")
        sys.exit(1)

print(f"\nMerging {len(all_obs)} datasets: {[n for n, _ in all_obs]}")

merged = {}
layers = list(all_obs[0][1].keys())

for layer in layers:
    merged[layer] = {}
    merged[layer]["total_tokens"] = sum(obs[layer]["total_tokens"] for _, obs in all_obs)

    for key in ["expert_frequency", "pairwise_expert_frequency"]:
        if key in all_obs[0][1][layer]:
            merged[layer][key] = sum(obs[layer][key] for _, obs in all_obs)

    for key in ["ean_sum", "weighted_ean_sum", "weighted_expert_frequency_sum"]:
        if key in all_obs[0][1][layer]:
            merged[layer][key] = sum(obs[layer][key] for _, obs in all_obs)

    freq = merged[layer]["expert_frequency"].float()
    total = merged[layer]["total_tokens"]
    total = total.float() if isinstance(total, torch.Tensor) else float(total)
    safe_freq = freq.clamp(min=1)

    if "ean_sum" in merged[layer]:
        merged[layer]["ean_mean"] = (merged[layer]["ean_sum"] / safe_freq).float()
        merged[layer]["reap"] = (merged[layer]["ean_sum"] / total).float()

    if "max_activations" in all_obs[0][1][layer]:
        merged[layer]["max_activations"] = torch.stack(
            [obs[layer]["max_activations"] for _, obs in all_obs]
        ).max(dim=0).values

combined_dir = artifacts / "combined" / "all"
combined_dir.mkdir(parents=True, exist_ok=True)
out_path = combined_dir / obs_file
torch.save(merged, out_path)

layer0 = layers[0]
print(f"\nSaved merged data to {out_path}")
print(f"  Layers: {len(merged)}")
print(f"  Total tokens: {merged[layer0]['total_tokens']}")
print(f"  Datasets merged: {len(all_obs)}")

# Sanity check layer 24
if 24 in merged:
    code_reap = all_obs[0][1][24].get("reap")
    comb_reap = merged[24].get("reap")
    if code_reap is not None and comb_reap is not None:
        n_experts = code_reap.shape[0]
        n_prune_20 = int(n_experts * 0.2)
        _, code_prune = torch.topk(code_reap, n_prune_20, largest=False)
        _, comb_prune = torch.topk(comb_reap, n_prune_20, largest=False)
        rescued = set(code_prune.tolist()) - set(comb_prune.tolist())
        print(f"\n=== Layer 24: code-only vs combined (20% prune) ===")
        print(f"  Experts rescued by diverse calibration: {len(rescued)}")
PYEOF

echo "[Phase 2] Done."

# ============================================================
# Phase 3: Prune at 20% (separate process, fresh model load)
# ============================================================
echo ""
echo "============================================================"
echo "[Phase 3] Pruning at 20% with combined calibration"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/reap/prune.py \
    --model-name "$MODEL" \
    --dataset-name combined \
    --compression-ratio 0.2 \
    --prune-method reap \
    --profile false \
    --do-eval false \
    --distance_measure cosine \
    --seed 42 \
    --output_file_name "$OBS_FILE" \
    --perserve_super_experts true \
    --perserve_outliers false \
    --samples_per_category $NUM_SAMPLES \
    --model_max_length $MAX_LEN \
    --record_pruning_metrics_only true \
    --smoke_test true

# Find the pruned model directory
PRUNED_DIR=$(find artifacts/Qwen3-Coder-Next/combined/pruned_models/ -maxdepth 1 -type d -name "reap-*" | head -1)
echo "[Phase 3] Pruned model at: $PRUNED_DIR"

if [ -z "$PRUNED_DIR" ] || [ ! -f "$PRUNED_DIR/config.json" ]; then
    echo "ERROR: Pruned model not found or incomplete."
    exit 1
fi

# Verify safetensors exist
N_SHARDS=$(ls "$PRUNED_DIR"/*.safetensors 2>/dev/null | wc -l)
echo "[Phase 3] Found $N_SHARDS safetensor shards."
# Process exits here, GPU memory fully freed

# ============================================================
# Phase 4: AWQ quantize (separate process)
# ============================================================
echo ""
echo "============================================================"
echo "[Phase 4] AWQ quantization"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 quantize_awq.py \
    --model-path "$PRUNED_DIR" \
    --save-dir /data/models/Qwen3-Coder-Next-REAP-AWQ

echo "[Phase 4] Done. AWQ model at: /data/models/Qwen3-Coder-Next-REAP-AWQ"
# Process exits here, GPU memory fully freed

# ============================================================
# Phase 5: Quick eval
# ============================================================
echo ""
echo "============================================================"
echo "[Phase 5] Serving model and running eval"
echo "============================================================"

# Start vLLM server in background
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /data/models/Qwen3-Coder-Next-REAP-AWQ \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.93 \
    --tensor-parallel-size 4 \
    --port 8000 \
    --disable-log-stats \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 16 \
    --enable-expert-parallel \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    > /tmp/vllm_reap_eval.log 2>&1 &

VLLM_PID=$!
echo "vLLM server starting (PID: $VLLM_PID)..."

# Wait for server
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 10
done

# Run eval
python3 ~/model_ab_eval.py --categories code,reasoning,scaffold,chat -v

# Cleanup
kill $VLLM_PID 2>/dev/null

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo "Pruned BF16 model: $PRUNED_DIR"
echo "AWQ model: /data/models/Qwen3-Coder-Next-REAP-AWQ"
echo "Eval results: ~/eval_results_*.json"
