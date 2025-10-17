# Visual Guide & Quick Reference: Fused MoE in TensorRT-LLM

This document provides visual diagrams and quick reference tables for understanding the fused MoE implementation.

## Table of Contents

1. [Visual Flow Diagrams](#visual-flow-diagrams)
2. [Data Structure Reference](#data-structure-reference)
3. [API Quick Reference](#api-quick-reference)
4. [Performance Characteristics](#performance-characteristics)
5. [Debugging Guide](#debugging-guide)

---

## Visual Flow Diagrams

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT PREPARATION                            │
├─────────────────────────────────────────────────────────────────────┤
│  Input Tokens: [4, 1024] bf16                                       │
│  Router Logits: [4, 8] float32  (4 tokens, 8 experts)              │
│                                                                      │
│  Routing (TopK=2):                                                  │
│    Token 0 → Experts [2, 5] with weights [0.6, 0.4]                │
│    Token 1 → Experts [1, 3] with weights [0.7, 0.3]                │
│    Token 2 → Experts [2, 7] with weights [0.5, 0.5]                │
│    Token 3 → Experts [1, 5] with weights [0.8, 0.2]                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      TOKEN DISPATCH (Permutation)                    │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 1: Build Expert Maps                                          │
│  ┌────────────────────────────────────────────────────────┐        │
│  │ CUDA Kernel: fusedBuildExpertMapsSortFirstToken        │        │
│  │ - Sort tokens by expert ID                              │        │
│  │ - Build permutation indices                             │        │
│  │ - Compute expert boundaries                             │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
│  Original Order:        Permuted Order (by expert):                 │
│  ┌──────┬──────┐       ┌──────┬──────┐                             │
│  │ T0→E2│ T0→E5│       │ T1→E1│ T3→E1│  } Expert 1                  │
│  │ T1→E1│ T1→E3│  →    │ T0→E2│ T2→E2│  } Expert 2                  │
│  │ T2→E2│ T2→E7│       │ T1→E3│      │  } Expert 3                  │
│  │ T3→E1│ T3→E5│       │ T0→E5│ T3→E5│  } Expert 5                  │
│  └──────┴──────┘       │ T2→E7│      │  } Expert 7                  │
│                         └──────┴──────┘                             │
│                                                                      │
│  expert_first_token_offset = [0, 2, 4, 5, 5, 7, 7, 8]              │
│                               E0 E1 E2 E3 E4 E5 E6 E7               │
│                                                                      │
│  STEP 2: Expand & Permute Input                                     │
│  ┌────────────────────────────────────────────────────────┐        │
│  │ CUDA Kernel: expandInputRowsKernel                     │        │
│  │ - Copy tokens to permuted positions                     │        │
│  │ - Replicate for top_k > 1                               │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
│  Output: permuted_data [8, 1024] bf16                               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      FP8 QUANTIZATION                                │
├─────────────────────────────────────────────────────────────────────┤
│  Input: [8, 1024] bf16                                              │
│                                                                      │
│  ┌────────────────────────────────────────────────────────┐        │
│  │ CUDA Kernel: fp8_1x128_cs                              │        │
│  │                                                         │        │
│  │ For each row:                                           │        │
│  │   For each 128-element block:                           │        │
│  │     1. Find max_abs in block                            │        │
│  │     2. scale = max_abs / 448.0                          │        │
│  │     3. quantize: fp8 = bf16 / scale                     │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
│  Output: act_fp8 [8, 1024] fp8                                      │
│          act_sf [8, 8] float32  (8 rows × 8 blocks of 128)          │
│                                                                      │
│  Memory Layout:                                                      │
│  ┌────────┬────────┬────────┬─────┬────────┐                       │
│  │ 0-127  │128-255 │256-383 │ ... │896-1023│ ← Row 0               │
│  │ scale₀ │ scale₁ │ scale₂ │ ... │ scale₇ │                       │
│  └────────┴────────┴────────┴─────┴────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    FC1 GROUP GEMM (w3_w1)                           │
├─────────────────────────────────────────────────────────────────────┤
│  Activation: [8, 1024] fp8                                          │
│  Weights: [8, 4096, 1024] fp8  (8 experts, each 4096×1024)         │
│                                                                      │
│  ┌────────────────────────────────────────────────────────┐        │
│  │ Group GEMM (via CuTe DSL or CUTLASS)                   │        │
│  │                                                         │        │
│  │ For expert_id in 0..7:                                  │        │
│  │   start = expert_first_token_offset[expert_id]         │        │
│  │   end = expert_first_token_offset[expert_id + 1]       │        │
│  │   num_tokens = end - start                              │        │
│  │                                                         │        │
│  │   if num_tokens > 0:                                    │        │
│  │     acts = activation[start:end]     # [num_tokens, k] │        │
│  │     wts = weights[expert_id]         # [n, k]          │        │
│  │     out[start:end] = matmul(acts, wts.T)               │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
│  Visual:                                                             │
│  Expert 1: ┌─────┐   ┌───────────┐   ┌─────┐                       │
│            │ T1  │ × │  W₁ (n×k) │ = │ O₁  │                       │
│            │ T3  │   └───────────┘   │ O₃  │                       │
│            └─────┘                    └─────┘                       │
│                                                                      │
│  Expert 2: ┌─────┐   ┌───────────┐   ┌─────┐                       │
│            │ T0  │ × │  W₂ (n×k) │ = │ O₀  │                       │
│            │ T2  │   └───────────┘   │ O₂  │                       │
│            └─────┘                    └─────┘                       │
│  ... (for all experts)                                              │
│                                                                      │
│  Output: h1 [8, 4096] bf16                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         ACTIVATION (SwiGLU)                          │
├─────────────────────────────────────────────────────────────────────┤
│  Input: h1 [8, 4096] bf16                                           │
│                                                                      │
│  ┌────────────────────────────────────────────────────────┐        │
│  │ def swiglu(x):                                          │        │
│  │     x, gate = x.chunk(2, dim=-1)                        │        │
│  │     return F.silu(gate) * x                             │        │
│  │                                                         │        │
│  │ Split 4096 → [2048, 2048]                               │        │
│  │                                                         │        │
│  │     ┌──────┐      ┌──────┐                             │        │
│  │     │  x   │      │ gate │                             │        │
│  │     │ 2048 │      │ 2048 │                             │        │
│  │     └──────┘      └──────┘                             │        │
│  │         │             │                                 │        │
│  │         │             ↓                                 │        │
│  │         │         SiLU(gate)                            │        │
│  │         │             │                                 │        │
│  │         └─────×───────┘                                 │        │
│  │                 │                                       │        │
│  │                 ↓                                       │        │
│  │             output                                      │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
│  Output: h2 [8, 2048] bf16                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│              FP8 QUANTIZATION (again) + FC2 GROUP GEMM              │
├─────────────────────────────────────────────────────────────────────┤
│  h2 [8, 2048] bf16 → Quantize → [8, 2048] fp8                       │
│                                                                      │
│  FC2 Weights: [8, 1024, 2048] fp8                                   │
│                                                                      │
│  Group GEMM (same as FC1):                                          │
│  For each expert: matmul(tokens, expert_weight)                     │
│                                                                      │
│  Output: h3 [8, 1024] bf16                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  TOKEN COMBINE (Unpermutation)                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐        │
│  │ CUDA Kernel: finalizeMoeRoutingKernel                  │        │
│  │                                                         │        │
│  │ For each original token:                                │        │
│  │   For each expert it used (top_k):                      │        │
│  │     1. Find permuted position                           │        │
│  │     2. Read expert output                               │        │
│  │     3. Multiply by routing weight                       │        │
│  │     4. Accumulate                                       │        │
│  │   5. Write to original position                         │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
│  Permuted Order:        Original Order:                             │
│  ┌──────┬────────┐     ┌───────────────────┐                       │
│  │ T1→E1│ 0.7×O₁ │     │ T0 = 0.6×O₂ + 0.4×O₅ │                     │
│  │ T3→E1│ 0.8×O₃ │     │ T1 = 0.7×O₁ + 0.3×O₃ │                     │
│  │ T0→E2│ 0.6×O₀ │  →  │ T2 = 0.5×O₂ + 0.5×O₇ │                     │
│  │ T2→E2│ 0.5×O₂ │     │ T3 = 0.8×O₁ + 0.2×O₅ │                     │
│  │ T1→E3│ 0.3×O₁ │     └───────────────────┘                       │
│  │ T0→E5│ 0.4×O₀ │                                                  │
│  │ T3→E5│ 0.2×O₃ │     Unpermuted + Weighted Sum                    │
│  │ T2→E7│ 0.5×O₂ │                                                  │
│  └──────┴────────┘                                                  │
│                                                                      │
│  Output: [4, 1024] bf16  (back to original token count)            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                         FINAL OUTPUT
                      [4, 1024] bf16
```

### Memory Layout Visualization

#### Token Permutation Example

```
Original Token Layout (4 tokens, top_k=2):
┌─────────────────────────────────────────────────────┐
│ Token 0 │ Token 1 │ Token 2 │ Token 3 │             │
│  [1024] │  [1024] │  [1024] │  [1024] │             │
│  E2, E5 │  E1, E3 │  E2, E7 │  E1, E5 │             │
└─────────────────────────────────────────────────────┘

After Expansion (8 token-expert pairs):
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ T0→E2  │ T0→E5  │ T1→E1  │ T1→E3  │ T2→E2  │ T2→E7  │ T3→E1  │ T3→E5  │
│ [1024] │ [1024] │ [1024] │ [1024] │ [1024] │ [1024] │ [1024] │ [1024] │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

After Permutation (grouped by expert):
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│ T1→E1  │ T3→E1  │ T0→E2  │ T2→E2  │ T1→E3  │ T0→E5  │ T3→E5  │ T2→E7  │
│ [1024] │ [1024] │ [1024] │ [1024] │ [1024] │ [1024] │ [1024] │ [1024] │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
   Expert 1         Expert 2         Expert 3  Expert 5         Expert 7

Mapping Arrays:
permuted_row_to_unpermuted_row = [2, 6, 0, 4, 3, 1, 7, 5]
unpermuted_row_to_permuted_row = [2, 5, 0, 4, 3, 7, 1, 6]

expert_first_token_offset = [0, 0, 2, 4, 5, 5, 7, 7, 8]
                             E0 E1 E2 E3 E4 E5 E6 E7 END
```

#### FP8 Block Scaling Layout

```
Input: [M=3, K=512] bf16
       Row 0: [0.5, 1.2, -0.8, ..., 2.1]  ← 512 elements
       Row 1: [0.3, -1.5, 0.9, ..., 1.7]
       Row 2: [2.0, 0.4, -1.1, ..., 0.6]

Block size: 128 elements
Blocks per row: 512 / 128 = 4

Scale Computation (1x128 blocking):
Row 0:
  Block 0 [0:127]:   max_abs = 1.2  → scale₀ = 1.2/448
  Block 1 [128:255]: max_abs = 1.8  → scale₁ = 1.8/448
  Block 2 [256:383]: max_abs = 2.3  → scale₂ = 2.3/448
  Block 3 [384:511]: max_abs = 2.1  → scale₃ = 2.1/448

Output FP8:
┌──────────────────────────────────────────────────────┐
│         Block 0   │   Block 1   │   Block 2   │  Block 3   │
│ Row 0: [fp8 vals] │ [fp8 vals]  │ [fp8 vals]  │ [fp8 vals] │
│ Row 1: [fp8 vals] │ [fp8 vals]  │ [fp8 vals]  │ [fp8 vals] │
│ Row 2: [fp8 vals] │ [fp8 vals]  │ [fp8 vals]  │ [fp8 vals] │
└──────────────────────────────────────────────────────┘

Output Scales: [M, num_blocks] = [3, 4] float32
┌─────────┬─────────┬─────────┬─────────┐
│ scale₀₀ │ scale₀₁ │ scale₀₂ │ scale₀₃ │  Row 0
│ scale₁₀ │ scale₁₁ │ scale₁₂ │ scale₁₃ │  Row 1
│ scale₂₀ │ scale₂₁ │ scale₂₂ │ scale₂₃ │  Row 2
└─────────┴─────────┴─────────┴─────────┘

During GEMM:
  fp8_value * scale_a * scale_b → fp32 accumulator → bf16 output
```

### Autotuning Flow

```
┌─────────────────────────────────────────────────────┐
│          First Kernel Invocation                     │
└─────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  AutoTuner.choose_one()                              │
│                                                      │
│  1. Check cache for this operation + input shape     │
│     Cache Key: (op_name, input_shape_bucket)        │
│     └─→ If found: return cached tactic              │
│                                                      │
│  2. If not cached: Benchmark all tactics             │
├─────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────┐ │
│  │  runner.get_valid_tactics()                   │ │
│  │                                               │ │
│  │  Generate candidates:                         │ │
│  │  - MMA tile shapes: (128,128), (256,128), ... │ │
│  │  - Cluster configs: (1,1), (2,2), (4,4), ...  │ │
│  │  - Other params: swap_ab, pipeline_stages     │ │
│  │                                               │ │
│  │  Filter by hardware capabilities:             │ │
│  │  - Check if config is valid for SM version    │ │
│  │  - Check shared memory requirements           │ │
│  │  - Check register usage                       │ │
│  └───────────────────────────────────────────────┘ │
│                     ↓                                │
│  ┌───────────────────────────────────────────────┐ │
│  │  For each valid tactic:                       │ │
│  │                                               │ │
│  │  1. Compile kernel (if not compiled)          │ │
│  │  2. Warm-up run (2 iterations)                │ │
│  │  3. Benchmark run (10 iterations)             │ │
│  │  4. Record average time                       │ │
│  └───────────────────────────────────────────────┘ │
│                     ↓                                │
│  ┌───────────────────────────────────────────────┐ │
│  │  Select fastest tactic                        │ │
│  │  Cache result                                 │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  Execute with optimal tactic                         │
└─────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  Subsequent Invocations                              │
│  - Lookup in cache: O(1)                            │
│  - Execute with cached tactic                       │
│  - No benchmarking overhead                         │
└─────────────────────────────────────────────────────┘
```

---

## Data Structure Reference

### Key Tensors

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `input` | `[num_tokens, hidden_size]` | bf16 | Input tokens |
| `router_logits` | `[num_tokens, num_experts]` | float32 | Router output |
| `token_selected_experts` | `[num_tokens, top_k]` | int32 | Expert IDs for each token |
| `token_final_scales` | `[num_tokens, top_k]` | float32 | Routing weights |
| `permuted_data` | `[num_tokens*top_k, hidden_size]` | bf16 | Tokens grouped by expert |
| `permuted_row_to_unpermuted_row` | `[num_tokens*top_k]` | int32 | Permuted → original mapping |
| `unpermuted_row_to_permuted_row` | `[num_tokens*top_k]` | int32 | Original → permuted mapping |
| `expert_first_token_offset` | `[num_experts+1]` | int64 | Start index for each expert |
| `act_input_fp8` | `[num_tokens*top_k, hidden_size]` | fp8 | FP8 quantized activations |
| `act_input_sf` | `[num_tokens*top_k, K/128]` | float32 | FP8 scales (K=hidden_size) |
| `w3_w1_weight` | `[num_experts, inter_size, hidden_size]` | fp8 | FC1 expert weights |
| `w2_weight` | `[num_experts, hidden_size, inter_size/2]` | fp8 | FC2 expert weights |
| `h1` | `[num_tokens*top_k, inter_size]` | bf16 | After FC1 |
| `h2` | `[num_tokens*top_k, inter_size/2]` | bf16 | After SwiGLU |
| `h3` | `[num_tokens*top_k, hidden_size]` | bf16 | After FC2 |
| `final_hidden_states` | `[num_tokens, hidden_size]` | bf16 | Final output |

### Mapping Arrays Explained

#### `permuted_row_to_unpermuted_row`
```
Index: permuted position (after grouping by expert)
Value: original position (before permutation)

Example:
  permuted_row_to_unpermuted_row[0] = 2
  → The token at permuted position 0 came from original position 2
```

#### `unpermuted_row_to_permuted_row`
```
Index: original position (before permutation)
Value: permuted position (after grouping)

Example:
  unpermuted_row_to_permuted_row[2] = 0
  → The token at original position 2 is now at permuted position 0
```

#### `expert_first_token_offset`
```
Index: expert_id
Value: starting index in permuted array

Example:
  expert_first_token_offset = [0, 0, 2, 4, 5, 5, 7, 7, 8]
                               E0 E1 E2 E3 E4 E5 E6 E7 END

  Expert 1: tokens at indices [0:2)   → 2 tokens
  Expert 2: tokens at indices [2:4)   → 2 tokens
  Expert 3: tokens at indices [4:5)   → 1 token
  Expert 4: tokens at indices [5:5)   → 0 tokens (not used)
  Expert 5: tokens at indices [5:7)   → 2 tokens
  Expert 7: tokens at indices [7:8)   → 1 token
```

---

## API Quick Reference

### Python Entry Points

#### 1. CuteDslFusedMoE

```python
from tensorrt_llm._torch.modules.fused_moe import CuteDslFusedMoE

moe = CuteDslFusedMoE(
    routing_method=routing_method,     # BaseMoeRoutingMethod
    num_experts=64,
    hidden_size=4096,
    intermediate_size=14336,
    dtype=torch.bfloat16,
    reduce_results=False,
    model_config=model_config,
)

output = moe(
    hidden_states=x,                   # [num_tokens, hidden_size]
    router_logits=router_logits,       # [num_tokens, num_experts]
)
```

#### 2. Custom Ops

```python
# Token permutation
(permuted_row_to_unpermuted, permuted_experts, permuted_data,
 expert_offsets, permuted_scales, unpermuted_to_permuted) = \
    torch.ops.trtllm.moe_permute_op(
        input, token_selected_experts, token_final_scales,
        fc1_weights, fc2_weights, quant_scales, input_sf,
        num_experts_on_rank, tp_size, tp_rank, ep_size, ep_rank,
        cluster_size, cluster_rank, min_latency_mode, use_fp8_block_scaling
    )

# FP8 quantization
act_fp8, act_sf = torch.ops.trtllm.fp8_quantize_1x128(input_bf16)

# Token combine
output = torch.ops.trtllm.moe_finalize_scale_op(
    gemm2_output, biases, unpermuted_final_scales,
    unpermuted_to_permuted, permuted_to_unpermuted,
    token_selected_experts, expert_offsets,
    enable_alltoall, num_rows, hidden_size, unpadded_hidden_size,
    experts_per_token, num_experts_per_node,
    tp_size, tp_rank, ep_size, ep_rank
)
```

### C++ API

#### MoE Utilities

```cpp
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h"

namespace tk = tensorrt_llm::kernels;

// Build expert maps (fused)
bool success = tk::cutlass_kernels::fusedBuildExpertMapsSortFirstToken(
    token_selected_experts,
    permuted_row_to_unpermuted_row,
    unpermuted_row_to_permuted_row,
    expert_first_token_offset,
    num_tokens,
    num_experts_per_node,
    experts_per_token,
    start_expert,
    end_expert,
    stream
);

// Expand input rows
tk::cutlass_kernels::expandInputRowsKernelLauncher<InputType, OutputType>(
    unpermuted_input,
    permuted_output,
    unpermuted_scales,
    permuted_scales,
    permuted_row_to_unpermuted_row,
    num_rows,
    hidden_size,
    k,
    num_experts_per_node,
    quant_params,
    use_per_expert_act_scale,
    expert_first_token_offset,
    fc1_act_sf_flat,
    input_sf,
    swizzled_input_sf,
    prequant_scales,
    stream
);

// Finalize routing
tk::cutlass_kernels::finalizeMoeRoutingKernelLauncher<OutputType, GemmType, BiasType>(
    expanded_permuted_rows,
    reduced_unpermuted_output,
    bias,
    final_scales,
    unpermuted_row_to_permuted_row,
    permuted_row_to_unpermuted_row,
    token_selected_experts,
    expert_first_token_offset,
    num_rows,
    padded_cols,
    unpadded_cols,
    experts_per_token,
    num_experts_per_node,
    parallelism_config,
    enable_alltoall,
    stream
);
```

#### FP8 Quantization

```cpp
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"

using namespace tensorrt_llm::kernels::fp8_blockscale_gemm;

CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16> runner;

// 1x128 block quantization
runner.fp8CS1x128(
    mat_quant,      // __nv_fp8_e4m3* output
    scales,         // float* scales
    mat,            // __nv_bfloat16 const* input
    shape_x,        // int (columns)
    shape_y,        // int (rows)
    stream
);

// Grouped GEMM
runner.moeGemm(
    mat_d,                  // void* output
    mat_a,                  // void const* activations
    mat_b,                  // void const* weights
    problem_m_offsets,      // int64_t const* expert offsets
    num_problems,           // size_t (num experts)
    shape_n,                // size_t (output cols)
    shape_k,                // size_t (reduction dim)
    stream,
    scales_a,               // float const* activation scales
    scales_b                // float const* weight scales
);
```

---

## Performance Characteristics

### Kernel Launch Counts

For a single MoE layer with `num_tokens=1024, num_experts=64, top_k=2`:

| Operation | Kernel Launches | Notes |
|-----------|----------------|-------|
| **Routing** | 1 | TopK kernel |
| **Token Dispatch** | 1-2 | Fused (1) or 3-step (2-3) |
| **FP8 Quantize (FC1)** | 1 | 1x128 block scaling |
| **FC1 Group GEMM** | 1 | Grouped/persistent kernel |
| **SwiGLU** | 1 | Element-wise activation |
| **FP8 Quantize (FC2)** | 1 | 1x128 block scaling |
| **FC2 Group GEMM** | 1 | Grouped/persistent kernel |
| **Token Combine** | 1 | Finalize kernel |
| **Total** | **8-9** | vs ~300+ for naive loop |

### Memory Bandwidth Analysis

For `num_tokens=1024, hidden_size=4096, intermediate_size=14336, num_experts=64, top_k=2`:

**Token Dispatch:**
- Read: `1024 * 4096 * 2` bytes (bf16) = 8 MB
- Write: `2048 * 4096 * 2` bytes (bf16) = 16 MB
- Metadata: ~200 KB
- **Total: ~24 MB**

**FP8 Quantization (FC1):**
- Read: `2048 * 4096 * 2` bytes (bf16) = 16 MB
- Write: `2048 * 4096 * 1` bytes (fp8) = 8 MB
- Scales: `2048 * 32 * 4` bytes = 256 KB
- **Total: ~24 MB**

**FC1 Group GEMM:**
- Activations: `2048 * 4096 * 1` bytes (fp8) = 8 MB
- Weights: `64 * 14336 * 4096 * 1` bytes (fp8) = 3.7 GB
- Output: `2048 * 14336 * 2` bytes (bf16) = 56 MB
- **Total: ~3.76 GB**
- **Note:** Weights reused across batches, often in L2 cache

**SwiGLU:**
- Read+Write: `2048 * 14336 * 2 * 2` bytes = 112 MB

**FP8 Quantization (FC2):**
- Similar to FC1 quantization: ~28 MB

**FC2 Group GEMM:**
- Activations: `2048 * 7168 * 1` bytes (fp8) = 14 MB
- Weights: `64 * 4096 * 7168 * 1` bytes (fp8) = 1.8 GB
- Output: `2048 * 4096 * 2` bytes (bf16) = 16 MB
- **Total: ~1.83 GB**

**Token Combine:**
- Read: `2048 * 4096 * 2` bytes = 16 MB
- Write: `1024 * 4096 * 2` bytes = 8 MB
- **Total: ~24 MB**

**Grand Total: ~5.9 GB** (dominated by weight reads)

### Compute Analysis

**FC1 GEMM:** `2 * M * N * K = 2 * 2048 * 14336 * 4096 ≈ 244 TFLOPS`
**FC2 GEMM:** `2 * M * N * K = 2 * 2048 * 4096 * 7168 ≈ 122 TFLOPS`
**Total: ~366 TFLOPS**

On H100 (1979 TFLOPS FP8):
- Ideal time: 366 / 1979 = 185 ms
- Memory time: 5900 MB / 3350 GB/s = 1.76 ms
- **Compute-bound** (good!)

### Roofline Analysis

```
                   Compute Bound
                         │
    TFLOPS               │
       │                 │
  2000 ├─────────────────┼──────────── H100 Peak (FP8)
       │             ╱   │
  1500 │         ╱       │
       │     ╱           │
  1000 │ ╱   MoE GEMM    │
       │                 │
   500 ├─────────────────┼─────────────
       │                 │
     0 └─────────────────┴──────────────→
       0    1    2    3    4    5    Arithmetic Intensity
                                        (FLOPS/Byte)

MoE GEMM Arithmetic Intensity:
  = Total FLOPs / Total Bytes
  = 366 TFLOPS / 5.9 GB
  = 62 FLOPS/Byte

H100 Balance Point = 1979 TFLOPS / 3350 GB/s = 0.59 FLOPS/Byte

MoE is heavily compute-bound (62 >> 0.59)
```

---

## Debugging Guide

### Common Issues & Solutions

#### 1. Shape Mismatches

**Symptom:** Runtime errors about incompatible shapes

**Debug Steps:**
```python
# Add shape logging
print(f"Input: {x.shape}")
print(f"Router logits: {router_logits.shape}")
print(f"Selected experts: {token_selected_experts.shape}")
print(f"Permuted data: {permuted_data.shape}")
print(f"Expert offsets: {expert_first_token_offset}")
```

**Common Causes:**
- `hidden_size` not multiple of 16 (FP8 requirement)
- `num_tokens` exceeds expected range
- Mismatch between `top_k` and routing method

#### 2. CUDA Errors

**Symptom:** `CUDA error: invalid configuration argument`

**Debug Steps:**
```python
# Check CUDA synchronization
torch.cuda.synchronize()

# Enable CUDA error checking
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

**Common Causes:**
- Grid/block dimensions exceed hardware limits
- Shared memory exceeds available per-block
- Invalid pointer access

#### 3. NaN/Inf in Output

**Symptom:** Output contains NaN or Inf values

**Debug Steps:**
```python
# Check intermediate outputs
print(f"h1 - min: {h1.min()}, max: {h1.max()}, has_nan: {h1.isnan().any()}")
print(f"h2 - min: {h2.min()}, max: {h2.max()}, has_nan: {h2.isnan().any()}")
print(f"h3 - min: {h3.min()}, max: {h3.max()}, has_nan: {h3.isnan().any()}")

# Check FP8 scales
print(f"Act scales - min: {act_input_sf.min()}, max: {act_input_sf.max()}")
```

**Common Causes:**
- FP8 overflow (max_abs > 448.0)
- Division by zero in scaling
- Invalid expert weights

#### 4. Permutation Bugs

**Symptom:** Incorrect token ordering in output

**Debug Steps:**
```python
# Verify round-trip permutation
original_indices = torch.arange(num_tokens * top_k)
permuted_indices = unpermuted_row_to_permuted_row
reconstructed = permuted_row_to_unpermuted_row[permuted_indices]

assert (original_indices == reconstructed).all(), "Permutation maps inconsistent!"

# Check expert offsets
for i in range(num_experts):
    start = expert_first_token_offset[i]
    end = expert_first_token_offset[i + 1]
    print(f"Expert {i}: tokens [{start}:{end}), count={end-start}")
```

**Common Causes:**
- Incorrect mapping array computation
- Off-by-one errors in expert boundaries
- Corruption of permutation indices

### Profiling & Optimization

#### CUDA Events Profiling

```python
import torch

# Create events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Profile token dispatch
start.record()
permuted_data, offsets, mappings = moe_permute_op(...)
end.record()
torch.cuda.synchronize()
print(f"Token dispatch: {start.elapsed_time(end):.2f} ms")

# Profile FC1 GEMM
start.record()
h1 = group_gemm(...)
end.record()
torch.cuda.synchronize()
print(f"FC1 GEMM: {start.elapsed_time(end):.2f} ms")
```

#### Nsight Compute Profiling

```bash
# Profile specific kernels
ncu --target-processes all \
    --kernel-name fusedBuildExpertMapsSortFirstToken \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    python your_script.py

# Generate detailed report
ncu --set full \
    --export profile.ncu-rep \
    python your_script.py
```

#### Interpreting Results

**Metrics to watch:**
- **SM Throughput:** Should be >80% for compute-bound kernels
- **Memory Throughput:** Check for bottlenecks
- **Occupancy:** Aim for >50%
- **Bank Conflicts:** Should be minimal
- **Cache Hit Rate:** Higher is better for weight reads

### Optimization Checklist

- [ ] **Input shapes** are optimal (multiple of 128 for FP8)
- [ ] **Autotuning** has run (check cache hit rate)
- [ ] **Expert load balancing** is good (check token distribution)
- [ ] **Memory coalescing** verified (use ncu)
- [ ] **Compute occupancy** >50% (use ncu)
- [ ] **L2 cache utilization** optimized (persistent kernels)
- [ ] **Top-k value** is appropriate (lower = faster)
- [ ] **FP8 quantization** error is acceptable

---

## Summary

This guide provides visual diagrams, data structure reference, and debugging tools for understanding and optimizing the fused MoE implementation in TensorRT-LLM.

**Key Takeaways:**
1. **Permutation** is the key to efficient grouped computation
2. **FP8 quantization** provides 2x speedup with minimal accuracy loss
3. **Group GEMM** processes all experts in a single kernel
4. **Autotuning** is essential for optimal performance
5. **Profiling tools** help identify bottlenecks

For more details, refer to:
- [Main Call Stack Trace](fused_moe_callstack_trace.md)
- [Detailed Kernel Trace](detailed_kernel_trace.md)
