# TensorRT-LLM Fused MoE Call Stack Trace

This document provides a comprehensive, step-by-step trace of the entire call stack for the fused Mixture of Experts (MoE) implementation in TensorRT-LLM, specifically focusing on the CuteDSL backend.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component 1: Token Dispatch (All-to-All Routing)](#component-1-token-dispatch-all-to-all-routing)
3. [Component 2: MoE Computation (Group GEMM)](#component-2-moe-computation-group-gemm)
4. [Component 3: Token Combine (Distribution Back)](#component-3-token-combine-distribution-back)
5. [Component 4: Autotuning Mechanisms](#component-4-autotuning-mechanisms)
6. [Data Flow Diagrams](#data-flow-diagrams)

---

## Architecture Overview

The fused MoE implementation follows this high-level pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Input                                   │
│  tokens: [num_tokens, hidden_size]                              │
│  router_logits: [num_tokens, num_experts]                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: ROUTING                                                │
│  - Apply routing method (TopK, etc.)                            │
│  - Get expert selections & routing weights                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: TOKEN DISPATCH (Permutation)                          │
│  - Build expert maps                                             │
│  - Permute tokens by expert assignment                           │
│  - FP8 quantization (if needed)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: MoE COMPUTATION                                        │
│  - Expert FC1 (w3_w1): Group GEMM                               │
│  - Activation (SwiGLU)                                          │
│  - Expert FC2 (w2): Group GEMM                                  │
└─────────────────────────────────────────────────────────────────┐
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: TOKEN COMBINE (Unpermutation)                         │
│  - Apply routing weights                                         │
│  - Unpermute tokens back to original order                       │
│  - Reduce across experts (if top-k > 1)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Final Output                                  │
│  [num_tokens, hidden_size]                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Key Files

**Python Layer:**
- [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py`](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py) - Main CuteDSL implementation
- [`tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`](../tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py) - Custom ops for CuteDSL

**C++ Bindings:**
- [`cpp/tensorrt_llm/thop/moeUtilOp.cpp`](../cpp/tensorrt_llm/thop/moeUtilOp.cpp) - Torch bindings for MoE utilities
- [`cpp/tensorrt_llm/thop/fp8Quantize.cpp`](../cpp/tensorrt_llm/thop/fp8Quantize.cpp) - FP8 quantization bindings

**CUDA Kernels:**
- [`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu) - MoE CUDA kernels
- [`cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h`](../cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h) - MoE kernel headers
- [`cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu`](../cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu) - FP8 GEMM kernels

---

## Component 1: Token Dispatch (All-to-All Routing)

Token dispatch reorganizes tokens so that all tokens assigned to the same expert are grouped together for efficient batched computation.

### 1.1 Python Entry Point

**File:** [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py:193-210`](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L193-L210)

```python
# Line 193-210
(
    permuted_row_to_unpermuted_row_tensor,
    permuted_token_selected_experts_tensor,
    permuted_data_tensor,
    expert_first_token_offset_tensor,
    permuted_token_final_scales_tensor,
    unpermuted_row_to_permuted_row_tensor,
) = torch.ops.trtllm.moe_permute_op(
    x,
    token_selected_experts,
    token_final_scales,
    None,  # w3_w1_weight
    None,  # w2_weight
    None,  # quant_scales
    input_sf=x_sf,
    num_experts_on_rank=self.expert_size_per_partition,
    tp_size=self.tp_size,
    # ... other parameters
)
```

**Inputs:**
- `x`: Input tokens `[num_tokens, hidden_size]` (bf16)
- `token_selected_experts`: Expert IDs selected for each token `[num_tokens, top_k]` (int32)
- `token_final_scales`: Routing weights `[num_tokens, top_k]` (float32)

**Outputs:**
- `permuted_row_to_unpermuted_row_tensor`: Mapping from permuted → original indices
- `permuted_data_tensor`: Tokens reorganized by expert assignment
- `expert_first_token_offset_tensor`: Start offset for each expert's tokens
- `unpermuted_row_to_permuted_row_tensor`: Mapping from original → permuted indices

### 1.2 Torch Library Registration

**File:** [`cpp/tensorrt_llm/thop/moeUtilOp.cpp:332-348`](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L332-L348)

```cpp
// Line 332-348
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_permute_op(Tensor input, Tensor token_selected_experts, "
        "Tensor? token_final_scales, Tensor fc1_expert_weights, "
        "Tensor fc2_expert_weights, Tensor[]? quant_scales, Tensor? input_sf, "
        "int num_experts_on_rank, int tp_size, int tp_rank, int ep_size, "
        "int ep_rank, int cluster_size, int cluster_rank, "
        "bool min_latency_mode, bool use_fp8_block_scaling)"
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_permute_op", &torch_ext::moe_permute_op);
}
```

### 1.3 C++ Implementation Entry

**File:** [`cpp/tensorrt_llm/thop/moeUtilOp.cpp:90-230`](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L90-L230)

```cpp
// Line 90-230
std::tuple<torch::Tensor, ...> moe_permute_op(
    torch::Tensor const& input,
    torch::Tensor const& token_selected_experts,
    torch::optional<torch::Tensor> token_final_scales,
    // ... other parameters
)
{
    // Allocate output tensors
    auto permuted_row_to_unpermuted_row_tensor =
        torch::empty({num_moe_inputs}, torch::dtype(torch::kInt32)...);

    // Dispatch based on data type
    switch (data_type) {
        case torch::kBFloat16:
            runPermute<__nv_bfloat16>(/*...*/);
            break;
        // ... other cases
    }

    return std::make_tuple(/*...*/);
}
```

### 1.4 Template Dispatcher

**File:** [`cpp/tensorrt_llm/thop/moeUtilOp.cpp:38-88`](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L38-L88)

```cpp
// Line 38-88
template <typename T>
void runPermute(
    void const* input_activations_void,
    void const* input_sf_void,
    int const* token_selected_experts,
    float const* token_final_scales,
    // ... parameters
)
{
    // Step 1: Build expert maps (fused or 3-step)
    bool fused_prologue_result =
        cutlass_kernels::fusedBuildExpertMapsSortFirstToken(/*...*/);

    if (!fused_prologue_result) {
        cutlass_kernels::threeStepBuildExpertMapsSortFirstToken(/*...*/);
    }

    // Step 2: Expand and permute input rows
    cutlass_kernels::expandInputRowsKernelLauncher(/*...*/);
}
```

### 1.5 CUDA Kernel: Fused Expert Map Building

**File:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:526-548`](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L526-L548)

```cpp
// Line 526-548
bool fusedBuildExpertMapsSortFirstToken(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    int64_t const num_tokens,
    int const num_experts_per_node,
    int const experts_per_token,
    int const start_expert,
    int const end_expert,
    cudaStream_t stream)
{
    // Determine number of bits needed for expert representation
    int expert_log = static_cast<int>(log2(num_experts_per_node + 1)) + 1;

    // Dispatch to templated kernel based on log2(num_experts)
    if (expert_log <= 9) {
        return funcs[expert_log - 1](/*...*/);
    }
    return false;
}
```

**Key Algorithm:** Uses bitonic sort-like approach to build mappings in a single kernel pass.

### 1.6 Alternative: Three-Step Expert Map Building

For cases where fused approach isn't available:

**File:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:563-657`](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L563-L657)

```cpp
// Line 563-613: Step 1 - Block-level prefix sum
__global__ void blockExpertPrefixSumKernel(
    int const* token_selected_experts,
    int* blocked_expert_counts,
    int* blocked_row_to_unpermuted_row,
    // ... parameters
)
{
    using BlockScan = cub::BlockScan<int, kNumTokensPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    // Each block processes tokens for one expert
    int const target_expert_id = blockIdx.x;
    int const block_id = blockIdx.y;

    // Find tokens assigned to this expert
    int expanded_token_id = -1;
    if (token_id < num_tokens) {
        for (int i = 0; i < num_experts_per_token; i++) {
            int const expert_id = token_selected_experts[...];
            if (expert_id == target_expert_id) {
                expanded_token_id = i * num_tokens + token_id;
                break;
            }
        }
    }

    // Compute prefix sum to get write positions
    int const has_matched = expanded_token_id >= 0 ? 1 : 0;
    int index;
    BlockScan(temp_storage).ExclusiveSum(has_matched, index);

    // Write to output
    if (has_matched) {
        blocked_row_to_unpermuted_row[...] = expanded_token_id;
    }
}
```

### 1.7 CUDA Kernel: Expand Input Rows

**File:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h:55-62`](../cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h#L55-L62)

```cpp
// Line 55-62
template <class InputActivationsType, class ExpandedActivationsType>
void expandInputRowsKernelLauncher(
    InputActivationsType const* unpermuted_input,
    ExpandedActivationsType* permuted_output,
    float const* unpermuted_scales,
    float* permuted_scales,
    int const* permuted_row_to_unpermuted_row,
    int64_t const num_rows,
    int64_t const hidden_size,
    // ... parameters
);
```

**What it does:**
1. Reads input tokens in original order
2. Writes them to permuted locations based on expert assignment
3. Handles top-k > 1 by replicating tokens
4. Applies any pre-quantization scaling

---

## Component 2: MoE Computation (Group GEMM)

After tokens are permuted, expert computation happens in three stages: FC1, Activation, FC2.

### 2.1 Python: FP8 Quantization

**File:** [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py:211-212`](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L211-L212)

```python
# Line 211-212
act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(
    permuted_data_tensor)
```

**Input:** `permuted_data_tensor` - bf16 tokens `[num_permuted_tokens, hidden_size]`
**Output:**
- `act_input_fp8` - FP8 quantized tokens `[num_permuted_tokens, hidden_size]`
- `act_input_sf` - Scale factors for 1x128 blocks

### 2.2 FP8 Quantization: Torch Binding

**File:** [`cpp/tensorrt_llm/thop/fp8Quantize.cpp:29-84`](../cpp/tensorrt_llm/thop/fp8Quantize.cpp#L29-L84)

```cpp
// Line 29-84
std::tuple<at::Tensor, at::Tensor> fp8_quantize_1x128(at::Tensor const& self)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);
    TORCH_CHECK(self.scalar_type() == at::ScalarType::BFloat16, ...);

    auto const m = self.sizes()[0];
    auto const n = self.sizes()[1];

    auto mGemmRunner =
        tensorrt_llm::kernels::fp8_blockscale_gemm::
            CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>();

    // Allocate output tensors
    at::Tensor valueE4M3 = at::detail::empty_cuda(...);
    at::Tensor scaleFP8SF = at::detail::empty_cuda(...);

    // Launch quantization kernel
    mGemmRunner.fp8CS1x128(
        act_buffer, act_scale_buffer,
        reinterpret_cast<__nv_bfloat16 const*>(self.data_ptr()),
        n, m, stream);

    return {valueE4M3.slice(0, 0, m), scaleFP8SF};
}
```

### 2.3 FP8 Quantization: CUDA Kernel Dispatch

**File:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu:179-183`](../cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu#L179-L183)

```cpp
// Line 179-183
void CutlassFp8BlockScaleGemmRunner<...>::fp8CS1x128(
    __nv_fp8_e4m3* mat_quant,
    float* scales,
    __nv_bfloat16 const* mat,
    int shape_x,
    int shape_y,
    cudaStream_t stream)
{
    fp8_1x128_cs(mat_quant, scales, mat, shape_x, shape_y, stream);
}
```

**Algorithm:** Quantizes input in 1x128 blocks:
- Each 1x128 block (1 row, 128 columns) shares a single FP8 scale factor
- Finds max absolute value in block
- Computes scale: `scale = max_abs / FP8_MAX`
- Quantizes: `fp8_value = bf16_value / scale`

### 2.4 Python: FC1 Group GEMM (w3_w1)

**File:** [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py:213-219`](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L213-L219)

```python
# Line 213-219
h1 = cute_dsl_fp8_group_blockwise_gemm_ref(
    a=act_input_fp8,
    b=self.w3_w1_weight.view(weight_dtype),
    a_sf=act_input_sf,
    b_sf=self.quant_scales[0],
    offset_array=expert_first_token_offset_tensor,
)
```

**What it does:**
- Performs grouped GEMM where each group = tokens for one expert
- Uses `expert_first_token_offset_tensor` to know boundaries
- Each expert has different weight matrix
- FP8 block-scaled matrix multiplication

### 2.5 Reference Implementation: Group Blockwise GEMM

**File:** [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py:21-88`](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L21-L88)

```python
# Line 21-88
def cute_dsl_fp8_group_blockwise_gemm_ref(
    a: torch.Tensor,              # [m, k] - activations
    b: torch.Tensor,              # [l, n, k] - weights for l experts
    a_sf: torch.Tensor,           # activation scales
    b_sf: torch.Tensor,           # weight scales
    offset_array: torch.Tensor,   # [num_experts+1] - expert boundaries
) -> torch.Tensor:

    # Prepare scale factors
    def pad_and_multiply(scale, tensor):
        # Broadcasts scale to match tensor dimensions
        # Handles 1x128 groupwise and 128x128 blockwise scaling
        # ...
        return expanded_scale * tensor

    # Apply scales
    updated_a = pad_and_multiply(input_scale_tmp, a_tmp.to(torch.float32))
    updated_b = pad_and_multiply(weight_scale_tmp, b_tmp.to(torch.float32))

    # Group GEMM: process each expert separately
    ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    for i in range(len(offset_array) - 1):
        start = offset_array[i]
        end = offset_array[i + 1]
        # Compute GEMM for expert i
        ref[start:end, :] = torch.einsum(
            "mk,nk->mn",
            updated_a[start:end, :, 0],
            updated_b[:, :, i]
        )

    return ref.to(torch.bfloat16)
```

**Note:** This is a reference implementation. The actual kernel uses optimized CUTLASS/CuTe DSL kernels.

### 2.6 Python: Activation Function (SwiGLU)

**File:** [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py:16-18`](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L16-L18)

```python
# Line 16-18
def swiglu_fused_moe(x):
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x
```

Applied at line 220:
```python
h2 = swiglu_fused_moe(h1)
```

**SwiGLU:** `SwiGLU(x) = Swish(gate) * x = SiLU(gate) * x`

### 2.7 Python: FC2 Group GEMM (w2)

**File:** [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py:221-228`](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L221-L228)

```python
# Line 221-228
act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(h2)
h3 = cute_dsl_fp8_group_blockwise_gemm_ref(
    a=act_input_fp8,
    b=self.w2_weight.view(weight_dtype),
    a_sf=act_input_sf,
    b_sf=self.quant_scales[1],
    offset_array=expert_first_token_offset_tensor,
)
```

Same pattern as FC1:
1. Quantize activation to FP8
2. Perform group GEMM with expert weights
3. Output is in bf16

---

## Component 3: Token Combine (Distribution Back)

After expert computation, tokens need to be unpermuted and routing weights applied.

### 3.1 Python Entry Point

**File:** [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py:229-248`](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L229-L248)

```python
# Line 229-248
final_hidden_states = torch.ops.trtllm.moe_finalize_scale_op(
    h3,                                      # gemm2 output
    None,                                    # biases
    token_final_scales,                      # routing weights
    unpermuted_row_to_permuted_row_tensor,
    permuted_row_to_unpermuted_row_tensor,
    token_selected_experts,
    expert_first_token_offset_tensor,
    False,                                   # enable_alltoall
    x.shape[0],                              # num_rows
    x.shape[1],                              # hidden_size
    self.unpadded_hidden_size,
    self.routing_method.top_k,
    self.expert_size_per_partition,
    self.tp_size,
    self.tp_rank,
    self.ep_size,
    self.ep_rank,
)
```

### 3.2 Torch Library Registration

**File:** [`cpp/tensorrt_llm/thop/moeUtilOp.cpp:340-354`](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L340-L354)

```cpp
// Line 340-354
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_finalize_scale_op(Tensor gemm2_output, Tensor? biases, "
        "Tensor unpermuted_final_scales, Tensor unpermuted_row_to_permuted_row, "
        "Tensor permuted_row_to_unpermuted_row, Tensor token_selected_experts, "
        "Tensor expert_first_token_offset_tensor, bool enable_alltoall, "
        "SymInt num_rows, SymInt hidden_size, SymInt unpadded_hidden_size, "
        "int experts_per_token, int num_experts_per_node, "
        "int tp_size, int tp_rank, int ep_size, int ep_rank)"
        "-> (Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m) {
    m.impl("moe_finalize_scale_op", &torch_ext::run_moe_finalize_scale_op);
}
```

### 3.3 C++ Implementation

**File:** [`cpp/tensorrt_llm/thop/moeUtilOp.cpp:248-328`](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L248-L328)

```cpp
// Line 248-328
torch::Tensor run_moe_finalize_scale_op(
    torch::Tensor const& gemm2_output,
    torch::optional<torch::Tensor> biases,
    torch::Tensor const& unpermuted_final_scales,
    torch::Tensor const& unpermuted_row_to_permuted_row,
    torch::Tensor const& permuted_row_to_unpermuted_row,
    torch::Tensor const& token_selected_experts,
    torch::Tensor const& expert_first_token_offset_tensor,
    bool enable_alltoall,
    c10::SymInt num_rows_param,
    // ... parameters
)
{
    auto parallelism_config = cutlass_kernels::MOEParallelismConfig(
        tp_size, tp_rank, ep_size, ep_rank);

    auto final_output = torch::empty({num_rows, unpadded_hidden_size}, ...);

    // Dispatch based on data type
    switch (data_type) {
        case torch::kBFloat16:
            runMoEFinalizeScaleOp<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
                gemm2_output.const_data_ptr(),
                biases,
                unpermuted_final_scales,
                // ... mappings ...
                final_output.data_ptr()
            );
            break;
        // ... other cases
    }

    return final_output;
}
```

### 3.4 Template Implementation

**File:** [`cpp/tensorrt_llm/thop/moeUtilOp.cpp:232-246`](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L232-L246)

```cpp
// Line 232-246
template <class UnfusedGemmOutputType, class ScaleBiasType, class OutputType>
void runMoEFinalizeScaleOp(
    UnfusedGemmOutputType const* const gemm2_output,
    ScaleBiasType const* const biases,
    float const* const unpermuted_final_scales,
    int const* const unpermuted_row_to_permuted_row,
    int const* const permuted_row_to_unpermuted_row,
    int const* const token_selected_experts,
    int64_t const* const expert_first_token_offset,
    // ... parameters
    OutputType* const final_output)
{
    cutlass_kernels::finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
        gemm2_output, final_output, biases,
        unpermuted_final_scales,
        unpermuted_row_to_permuted_row,
        permuted_row_to_unpermuted_row,
        token_selected_experts,
        expert_first_token_offset,
        num_rows, hidden_size, unpadded_hidden_size,
        experts_per_token, num_experts_per_node,
        parallelism_config, enable_alltoall, stream
    );
}
```

### 3.5 CUDA Kernel Launcher

**File:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h:64-71`](../cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h#L64-L71)

```cpp
// Line 64-71
template <class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(
    GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output,
    ScaleBiasType const* bias,
    float const* final_scales,
    int const* unpermuted_row_to_permuted_row,
    int const* permuted_row_to_unpermuted_row,
    int const* token_selected_experts,
    int64_t const* expert_first_token_offset,
    int64_t const num_rows,
    int64_t const padded_cols,
    int64_t const unpadded_cols,
    int64_t const experts_per_token,
    int64_t const num_experts_per_node,
    MOEParallelismConfig parallelism_config,
    bool const enable_alltoall,
    cudaStream_t stream
);
```

**What the kernel does:**
1. **Unpermute:** Use `permuted_row_to_unpermuted_row` to read tokens in permuted order, write to original positions
2. **Scale:** Multiply by routing weights from `final_scales`
3. **Reduce:** If `top_k > 1`, accumulate contributions from multiple experts for same token
4. **Bias:** Add bias if provided
5. **All-to-all:** Handle distributed expert parallelism if enabled

---

## Component 4: Autotuning Mechanisms

TensorRT-LLM uses sophisticated autotuning to select optimal kernels and configurations.

### 4.1 Autotuning Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AutoTuner (Singleton)                 │
│                                                         │
│  - Manages tuning cache                                 │
│  - Coordinates runner selection                         │
│  - Stores optimal tactics per operation                 │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  TunableRunner                          │
│                                                         │
│  - get_valid_tactics(): Generate candidate tactics      │
│  - forward(inputs, tactic): Execute with tactic         │
│  - TuningConfig: Define dynamic specs & constraints     │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Kernel Execution & Timing                  │
│                                                         │
│  - Benchmark each tactic                                │
│  - Select fastest                                       │
│  - Cache result                                         │
└─────────────────────────────────────────────────────────┘
```

### 4.2 CuteDSL FP4 Autotuning Example

**File:** [`tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py:31-126`](../tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py#L31-L126)

```python
# Line 31-126
class CuteDSLNVFP4BlackwellLinear(TunableRunner):
    kernel_dict = dict()

    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                0, 0,  # tensor_idx=0, dim=0
                get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2
            ),
        ),
        constraint_specs=(
            ConstraintSpec(2, 0, fp4_scale_infer_shape),
        ),
    )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[Tuple[int, int]]:

        # Generate candidate tactics
        mma_tiler_mn_candidates = [
            (256, 128), (128, 128), (128, 256),
            (256, 256), (256, 64), (128, 64),
        ]
        cluster_shape_mn_candidates = [
            (1, 1), (1, 2), (1, 4),
            (2, 1), (2, 2), (2, 4),
            (4, 1), (4, 2), (4, 4),
        ]
        swap_ab_candidates = [True, False]

        valid_tactics = []
        for swap_ab in swap_ab_candidates:
            for mma_tiler_mn in mma_tiler_mn_candidates:
                for cluster_shape_mn in cluster_shape_mn_candidates:
                    # Check if this configuration is valid
                    if Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
                        # ... params ...
                    ):
                        valid_tactics.append(
                            (mma_tiler_mn, cluster_shape_mn, swap_ab)
                        )

        return valid_tactics
```

### 4.3 Tactic Selection Flow

**File:** [`tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py:288-314`](../tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py#L288-L314)

```python
# Line 288-314
@torch.library.custom_op("trtllm::cute_dsl_nvfp4_gemm_blackwell",
                         mutates_args=(),
                         device_types="cuda")
def cute_dsl_nvfp4_gemm_blackwell(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:

    tuner = AutoTuner.get()

    cute_dsl_nvfp4_gemm_blackwell_runner = CuteDSLNVFP4BlackwellLinear(
        alpha, output_dtype)

    # AutoTuner selects best tactic
    _, best_tactic = tuner.choose_one(
        "trtllm::cute_dsl_nvfp4_gemm_blackwell",
        [cute_dsl_nvfp4_gemm_blackwell_runner],
        CuteDSLNVFP4BlackwellLinear.tuning_config,
        [input, weight, input_scale, weight_scale],
    )

    # Execute with best tactic
    return cute_dsl_nvfp4_gemm_blackwell_runner(
        inputs=[input, weight, input_scale, weight_scale],
        tactic=best_tactic,
    )
```

### 4.4 Kernel Compilation & Caching

**File:** [`tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py:229-265`](../tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py#L229-L265)

```python
# Line 229-265
def forward(self, inputs: List[torch.Tensor], tactic) -> torch.Tensor:
    if isinstance(tactic, tuple):
        mma_tiler_mn, cluster_shape_mn, swap_ab = tactic
    else:
        # fallback to default
        mma_tiler_mn, cluster_shape_mn, swap_ab = [(128, 128), (1, 1), False]

    CACHE_KEY = (sf_vec_size, mma_tiler_mn, cluster_shape_mn, swap_ab)

    # Check kernel cache
    if CACHE_KEY not in CuteDSLNVFP4BlackwellLinear.kernel_dict:
        # Create kernel wrapper
        gemm = gemm_wrapper_func(sf_vec_size, mma_tiler_mn, cluster_shape_mn)

        # Compile with CuTe DSL
        compiled_gemm = cute.compile(
            gemm,
            kernel_m, kernel_n, real_k,
            # ... shape parameters ...
            max_active_clusters,
            stream,
            swap_ab,
        )

        # Cache compiled kernel
        CuteDSLNVFP4BlackwellLinear.kernel_dict[CACHE_KEY] = compiled_gemm
    else:
        compiled_gemm = CuteDSLNVFP4BlackwellLinear.kernel_dict[CACHE_KEY]

    # Launch kernel
    compiled_gemm(/* ... */)

    return c_tensor
```

### 4.5 Autotuning: Token Dispatch Kernels

The token dispatch kernels are NOT autotuned but use heuristics:

**File:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:526-548`](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L526-L548)

```cpp
// Kernel selection based on number of experts
int expert_log = static_cast<int>(log2(num_experts_per_node + 1)) + 1;

if (expert_log <= 9) {
    // Use fused kernel for <=512 experts
    auto funcs = std::array{
        &fusedBuildExpertMapsSortFirstTokenBlockSize<1>,
        &fusedBuildExpertMapsSortFirstTokenBlockSize<2>,
        // ... up to <9>
    };
    return funcs[expert_log - 1](/*...*/);
}

// Fall back to 3-step approach for more experts
return false;
```

**File:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:550-561`](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L550-L561)

```cpp
// Determine block size based on num_tokens and num_experts
int64_t computeNumTokensPerBlock(
    int64_t const num_tokens,
    int64_t const num_experts_per_node)
{
    for (int64_t num_tokens_per_block = 32;
         num_tokens_per_block <= 1024;
         num_tokens_per_block *= 2)
    {
        int64_t const num_blocks_per_seq =
            ceilDiv(num_tokens, num_tokens_per_block);

        if (num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block) {
            return num_tokens_per_block;
        }
    }
    return 1024;
}
```

### 4.6 Autotuning: MoE GEMM Kernels

MoE GEMM kernels use CUTLASS's built-in autotuning:

1. **Kernel variants** are generated at compile time based on templates
2. **Runtime selection** happens via dispatch tables
3. **Configuration space** includes:
   - Thread block shapes (e.g., 128x128, 256x128)
   - Cluster configurations (for Hopper+)
   - Pipeline stages
   - Swizzle patterns

The autotuning happens during the **first invocation** and results are cached.

---

## Data Flow Diagrams

### Overall Data Flow

```
Input Tokens [num_tokens, hidden_size] (bf16)
              │
              ▼
    ┌─────────────────┐
    │  Routing        │
    │  (TopK/Expert)  │
    └─────────────────┘
              │
              ▼
    token_selected_experts [num_tokens, top_k] (int32)
    token_final_scales [num_tokens, top_k] (float32)
              │
              ▼
    ┌──────────────────────────────────────┐
    │  moe_permute_op                      │
    │  1. Build expert maps (CUDA)         │
    │  2. Permute tokens (CUDA)            │
    │  3. Expand for top_k > 1             │
    └──────────────────────────────────────┘
              │
              ▼
    permuted_data [num_permuted_tokens, hidden_size] (bf16)
    expert_first_token_offset [num_experts+1] (int64)
              │
              ▼
    ┌──────────────────────────────────────┐
    │  fp8_quantize_1x128 (CUDA)           │
    │  Quantize to FP8 with 1x128 blocks   │
    └──────────────────────────────────────┘
              │
              ▼
    act_input_fp8 [num_permuted_tokens, hidden_size] (fp8)
    act_input_sf (scales)
              │
              ▼
    ┌──────────────────────────────────────┐
    │  Group GEMM FC1 (CUDA)               │
    │  Multiply with w3_w1 weights         │
    │  FP8 blockscale GEMM                 │
    └──────────────────────────────────────┘
              │
              ▼
    h1 [num_permuted_tokens, intermediate_size] (bf16)
              │
              ▼
    ┌──────────────────────────────────────┐
    │  SwiGLU Activation                   │
    │  swish(gate) * x                     │
    └──────────────────────────────────────┘
              │
              ▼
    h2 [num_permuted_tokens, intermediate_size/2] (bf16)
              │
              ▼
    ┌──────────────────────────────────────┐
    │  fp8_quantize_1x128 (CUDA)           │
    └──────────────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────────────┐
    │  Group GEMM FC2 (CUDA)               │
    │  Multiply with w2 weights            │
    └──────────────────────────────────────┘
              │
              ▼
    h3 [num_permuted_tokens, hidden_size] (bf16)
              │
              ▼
    ┌──────────────────────────────────────┐
    │  moe_finalize_scale_op (CUDA)        │
    │  1. Apply routing weights            │
    │  2. Unpermute to original order      │
    │  3. Reduce across experts            │
    └──────────────────────────────────────┘
              │
              ▼
    Output [num_tokens, hidden_size] (bf16)
```

### Token Permutation Visualization

```
Original Token Order:          Permuted by Expert Assignment:
┌───┬───┬───┬───┬───┐         ┌───┬───┬───┬───┬───┐
│T0 │T1 │T2 │T3 │T4 │         │T0 │T3 │T1 │T4 │T2 │
│E2 │E0 │E1 │E0 │E1 │         │E0 │E0 │E1 │E1 │E2 │
└───┴───┴───┴───┴───┘         └───┴───┴───┴───┴───┘
                                 ▲       ▲       ▲
                                 │       │       │
                           Expert 0  Expert 1  Expert 2

expert_first_token_offset = [0, 2, 4, 5]
                             │  │  │  │
                             │  │  │  └─ End of Expert 2
                             │  │  └──── Start of Expert 2
                             │  └─────── Start of Expert 1
                             └────────── Start of Expert 0
```

### FP8 Block Scaling Visualization

```
Input Matrix (bf16):
┌──────────────────────────────────────┐
│  [ 128 elements ]  │  [ 128 elements ] │
│  Block 0           │  Block 1          │
│  max_abs = 5.2     │  max_abs = 3.1    │
│  scale_0 = 5.2/448 │  scale_1 = 3.1/448│
└──────────────────────────────────────┘
                  │
                  ▼
FP8 Quantized + Scales:
┌──────────────────────────────────────┐
│  FP8 values       │  FP8 values       │
│  [128 elements]   │  [128 elements]   │
└──────────────────────────────────────┘
        │                   │
        ▼                   ▼
    scale_0             scale_1
```

---

## Summary of Call Paths

### Token Dispatch Path
```
Python: CuteDslFusedMoE.forward_chunk()
  └─> torch.ops.trtllm.moe_permute_op()
      └─> C++: moeUtilOp.cpp::moe_permute_op()
          └─> C++: moeUtilOp.cpp::runPermute<T>()
              ├─> CUDA: moe_kernels.cu::fusedBuildExpertMapsSortFirstToken()
              │   └─> CUDA: fusedBuildExpertMapsSortFirstTokenBlockSize<K,LOG2>()
              │       └─> CUDA: fusedMoEPrologueKernel<...>() [actual kernel]
              │
              └─> CUDA: moe_kernels.cu::expandInputRowsKernelLauncher()
                  └─> CUDA: expandInputRowsKernel<...>() [actual kernel]
```

### FP8 Quantization Path
```
Python: torch.ops.trtllm.fp8_quantize_1x128()
  └─> C++: fp8Quantize.cpp::fp8_quantize_1x128()
      └─> C++: fp8_blockscale_gemm.cu::fp8CS1x128()
          └─> CUDA: fp8_blockscale_gemm_kernel.cuh::fp8_1x128_cs()
              └─> CUDA: fp8_1x128_cs_kernel<...>() [actual kernel]
```

### Group GEMM Path
```
Python: cute_dsl_fp8_group_blockwise_gemm_ref() [reference impl]
  OR
  CuTe DSL: Sm100BlockScaledPersistentDenseGemmKernelWrapper()
    └─> cute.compile() → generates optimized CUDA kernel
        └─> CUTLASS/CuTe generated kernel [actual kernel]
```

### Token Combine Path
```
Python: torch.ops.trtllm.moe_finalize_scale_op()
  └─> C++: moeUtilOp.cpp::run_moe_finalize_scale_op()
      └─> C++: moeUtilOp.cpp::runMoEFinalizeScaleOp<...>()
          └─> CUDA: moe_util_kernels.h::finalizeMoeRoutingKernelLauncher()
              └─> CUDA: finalizeMoeRoutingKernel<...>() [actual kernel]
```

### Autotuning Path
```
First invocation:
  AutoTuner.choose_one()
    ├─> runner.get_valid_tactics() → generate candidates
    ├─> For each tactic:
    │   ├─> runner.forward(inputs, tactic) → benchmark
    │   └─> measure execution time
    ├─> Select fastest tactic
    └─> Cache result

Subsequent invocations:
  AutoTuner.choose_one()
    └─> Return cached tactic (no benchmarking)
```

---

## File Reference Summary

| Component | Python | C++ Binding | CUDA Kernel |
|-----------|--------|-------------|-------------|
| **Token Dispatch** | [fused_moe_cute_dsl.py:193](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L193) | [moeUtilOp.cpp:90](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L90) | [moe_kernels.cu:526](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L526) |
| **FP8 Quantization** | [fused_moe_cute_dsl.py:211](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L211) | [fp8Quantize.cpp:29](../cpp/tensorrt_llm/thop/fp8Quantize.cpp#L29) | [fp8_blockscale_gemm.cu:179](../cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu#L179) |
| **Group GEMM** | [fused_moe_cute_dsl.py:213](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L213) | CuTe DSL | [cute_dsl_custom_ops.py:229](../tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py#L229) |
| **Token Combine** | [fused_moe_cute_dsl.py:229](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L229) | [moeUtilOp.cpp:248](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L248) | [moe_util_kernels.h:64](../cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h#L64) |
| **Autotuning** | [cute_dsl_custom_ops.py:31](../tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py#L31) | N/A | N/A |

---

## Key Insights

1. **Permutation Strategy**: The token dispatch uses a sophisticated two-level approach:
   - **Fused kernel** for small expert counts (≤512 experts)
   - **Three-step approach** for larger configurations

2. **FP8 Quantization**: Uses 1x128 block scaling:
   - Each block of 128 elements shares one FP8 scale factor
   - Enables high precision while reducing memory bandwidth

3. **Group GEMM**: Processes multiple expert GEMMs efficiently:
   - Uses `expert_first_token_offset` to define problem boundaries
   - Leverages CUTLASS grouped GEMM or CuTe DSL kernels

4. **Autotuning**: Multi-level optimization:
   - **Compile-time**: Template instantiation for different configurations
   - **First-run**: Benchmark tactics and cache results
   - **Runtime**: Use cached optimal tactic

5. **Memory Efficiency**:
   - In-place permutation where possible
   - FP8 reduces memory footprint by ~2x vs bf16
   - Grouped operations minimize kernel launches

This architecture achieves high performance through careful kernel fusion, memory layout optimization, and adaptive tactic selection.
