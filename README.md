# OmniClusterFusion - MLP Part

This project is the mlp part of the project OmniClusterFusion. Refer to the link for further information. OmniClusterFusion implements a **CUDA-accelerated decoder** layer for EleutherAI Pythia-2.8B model, and this repo focuses on the **MLP Down Projection** computation.

## Files

| File | Description |
|------|-------------|
| `include/5090/pythia_2b8/kernel_mlp.cuh` | CUDA kernel implementation |
| `include/5090/pythia_2b8/pythia_mlp_dispatch.cu` | Kernel dispatch |
| `tests/test_mlp_only.py` | Test and benchmark |

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

Benchmarked on GPU with sm_120:

| Metric | PyTorch | ClusterFusion | Speedup |
|--------|---------|---------------|---------|
| MLP Down per layer | 0.052 ms | 0.048 ms | **1.08x** |

Note: The MLP Down projection is a memory-bound operation (loading 26.2M weight parameters), limiting the achievable speedup.
