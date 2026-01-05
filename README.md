# OmniClusterFusion - MLP Part

This project is the MLP part of [ClusterFusionPlus](https://github.com/superk668/ClusterFusionPlus). Refer to the upstream repo for the full system.
OmniClusterFusion implements a CUDA-accelerated decoder layer for EleutherAI Pythia-2.8B model, and this repo focuses on the **MLP Down Projection** computation.

## Environment

- Python 3.13, NVIDIA GPU with `sm_120` compute capability, 5090 suggested to achieve designed performance
- CUDA 12.8+ user-space wheels via PyTorch cu130 index

## Quick Start

```bash
conda create -n OmniClusterFusion python=3.13 -y
conda activate OmniClusterFusion
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Kernel + HF stack
pip install flashinfer-python
pip install transformers accelerate datasets

# ClusterFusion build
pip install -e .

# Test OmniClusterFusion MLP part
python tests/test_mlp.py
```

## API Usage

```python
import clusterfusion
import torch.nn.functional as F

# Compute attention and MLP Up in PyTorch first
attn_output = pytorch_attention(hidden_states, ...)
mlp_up = F.linear(post_ln, mlp_up_weight, mlp_up_bias)
mlp_intermediate = F.gelu(mlp_up)

# CUDA kernel: MLP Down + Residual
output = clusterfusion.pythia_2b8_mlp_only(
    hidden_states,      # [1, hidden_dim] - original input for residual
    attn_output,        # [1, hidden_dim] - attention output for residual
    mlp_intermediate,   # [ffn_dim] - MLP Up + GELU output
    mlp_down_weight,    # [hidden_dim, ffn_dim] transposed
    mlp_down_bias       # [hidden_dim]
)
# output = hidden_states + attn_output + MLP_Down(mlp_intermediate)
```

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Hidden Size | 2560 |
| Attention Heads | 32 |
| Head Dimension | 80 |
| FFN Dimension | 10240 |
| Layers | 32 |

## Key Optimizations

1. **TMA Weight Loading**: Hardware-accelerated tensor memory access
2. **Cluster-level Reduction**: Efficient cross-block accumulation
3. **Fused Residual**: Combined MLP output with residual connection

## Performance Results

Benchmarked on NVIDIA RTX 5090 (sm_120), batch=1:

### End-to-End Benchmark (vs PyTorch Baseline)

| Tokens | CF(s) | PyTorch(s) | Speedup | TPOT CF(ms) | TPOT PT(ms) |
|--------|-------|------------|---------|-------------|-------------|
| 16 | 0.134 | 0.098 | 0.73x | 8.96 | 6.54 |
| 32 | 0.279 | 0.202 | 0.72x | 9.01 | 6.52 |
| 64 | 0.575 | 0.413 | 0.72x | 9.12 | 6.56 |
| 128 | 1.145 | 0.835 | 0.73x | 9.02 | 6.57 |
| 256 | 2.303 | 1.700 | 0.74x | 9.03 | 6.67 |
| 512 | 4.621 | 3.449 | 0.75x | 9.04 | 6.75 |
| 1024 | 9.272 | 7.128 | 0.77x | 9.06 | 6.97 |
| 2048 | 18.507 | 14.980 | 0.81x | 9.04 | 7.32 |

## Analysis
This repo focuses on the **MLP Down + residual** piece. In the full decoder, MLP Down runs immediately after Attention + MLP-Up, so it’s important to evaluate both **standalone** and **fused** behavior.

### Ablation (sequence length 2048)

| Configuration | Avg TPOT (ms) | vs PyTorch |
|---|---:|---:|
| PyTorch baseline | 6.80 | 1.00x |
| CUDA Attention + PyTorch MLP Down | 5.32 | 1.28x |
| PyTorch Attention + CUDA MLP Down | 9.04 | 0.75x |
| Full fused kernel | 4.90 | 1.39x |

**Important Observation**: Although MLP down alone decreases TPOT, when combined with Attention part, the overall performance outputs attention part alone, proving its importance in the complet kernel.

### What to take away

- **MLP Down alone can be slower than PyTorch**: PyTorch's `F.linear` uses highly optimized cuBLAS paths, and a standalone custom kernel pays fixed costs (e.g., TMA/cluster launch) without much work to amortize them.
- **Fusion is where MLP Down helps**: when fused after Attention + MLP-Up, the end-to-end kernel improves beyond Attention-only acceleration (1.39x vs 1.28x in the table).
- **Why fusion helps** (high-level): fewer kernel launches, less intermediate memory traffic, and better reuse of on-chip data/metadata across the combined computation.

One concrete example is avoiding extra reads/writes of the MLP intermediate. For batch=1 and fp16, the eliminated traffic is about 1.31 MB per decode step (read + write of a 10240-element vector across 32 layers). With ~1.8 TB/s device memory bandwidth (RTX 5090), that’s on the order of ~0.7 ms saved, consistent with the drop from 5.32 ms to 4.90 ms.
