# Detailed Kernel-Level Trace: Fused MoE in TensorRT-LLM

This document provides line-by-line traces with actual code snippets for each major kernel in the fused MoE pipeline.

## Table of Contents

1. [Token Dispatch Kernels](#token-dispatch-kernels)
2. [FP8 Quantization Kernels](#fp8-quantization-kernels)
3. [Group GEMM Kernels](#group-gemm-kernels)
4. [Token Combine Kernels](#token-combine-kernels)

---

## Token Dispatch Kernels

### Overview

Token dispatch reorganizes tokens based on expert assignments to enable efficient grouped computation.

### Kernel 1: Fused Expert Map Building

**Purpose:** Build permutation maps and expert offsets in a single kernel pass

**Launch Site:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:526-548`](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L526-L548)

```cpp
bool fusedBuildExpertMapsSortFirstToken(
    int const* token_selected_experts,           // [num_tokens * experts_per_token]
    int* permuted_row_to_unpermuted_row,         // [num_tokens * experts_per_token]
    int* unpermuted_row_to_permuted_row,         // [num_tokens * experts_per_token]
    int64_t* expert_first_token_offset,          // [num_experts_per_node + 1]
    int64_t const num_tokens,
    int const num_experts_per_node,
    int const experts_per_token,
    int const start_expert,
    int const end_expert,
    cudaStream_t stream)
{
    // Determine bits needed for expert representation
    int expert_log = static_cast<int>(log2(num_experts_per_node + 1)) + 1;

    if (expert_log <= 9) {  // Up to 512 experts
        auto funcs = std::array{
            &fusedBuildExpertMapsSortFirstTokenBlockSize<1>,
            &fusedBuildExpertMapsSortFirstTokenBlockSize<2>,
            &fusedBuildExpertMapsSortFirstTokenBlockSize<3>,
            &fusedBuildExpertMapsSortFirstTokenBlockSize<4>,
            &fusedBuildExpertMapsSortFirstTokenBlockSize<5>,
            &fusedBuildExpertMapsSortFirstTokenBlockSize<6>,
            &fusedBuildExpertMapsSortFirstTokenBlockSize<7>,
            &fusedBuildExpertMapsSortFirstTokenBlockSize<8>,
            &fusedBuildExpertMapsSortFirstTokenBlockSize<9>
        };

        return funcs[expert_log - 1](
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
    }

    return false;  // Fall back to 3-step approach
}
```

**Kernel Template Dispatcher:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:468-524`](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L468-L524)

The actual kernel is complex and uses bit-packing to efficiently sort tokens by expert. Key algorithm:

1. **Bit-pack** expert IDs with token IDs into a single value
2. **Radix sort** by expert ID (using CUB)
3. **Extract** sorted indices to build permutation maps
4. **Compute** prefix sum for expert offsets

### Kernel 2: Block Expert Prefix Sum (3-Step Approach)

**Purpose:** Count tokens per expert per block (Step 1 of 3-step approach)

**Launch Site:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:615-657`](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L615-L657)

```cpp
void blockExpertPrefixSum(
    int const* token_selected_experts,
    int* blocked_expert_counts,
    int* blocked_row_to_unpermuted_row,
    int64_t const num_tokens,
    int64_t const num_experts_per_node,
    int64_t const num_experts_per_token,
    int64_t const num_tokens_per_block,
    int64_t const num_blocks_per_seq,
    int const start_expert_id,
    cudaStream_t stream)
{
    dim3 const blocks(num_experts_per_node, num_blocks_per_seq);
    dim3 const threads(num_tokens_per_block);

    cudaLaunchConfig_t config;
    config.gridDim = blocks;
    config.blockDim = threads;
    config.stream = stream;

    // Select kernel based on block size
    auto func = blockExpertPrefixSumKernel<1024>;
    if (num_tokens_per_block <= 32) {
        func = blockExpertPrefixSumKernel<32>;
    } else if (num_tokens_per_block <= 64) {
        func = blockExpertPrefixSumKernel<64>;
    } else if (num_tokens_per_block <= 128) {
        func = blockExpertPrefixSumKernel<128>;
    } else if (num_tokens_per_block <= 256) {
        func = blockExpertPrefixSumKernel<256>;
    } else if (num_tokens_per_block <= 512) {
        func = blockExpertPrefixSumKernel<512>;
    }

    cudaLaunchKernelEx(&config, func, /*...args...*/);
}
```

**Actual Kernel:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:563-613`](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L563-L613)

```cpp
template <int kNumTokensPerBlock>
__global__ void blockExpertPrefixSumKernel(
    int const* token_selected_experts,
    int* blocked_expert_counts,
    int* blocked_row_to_unpermuted_row,
    int64_t const num_tokens,
    int64_t const num_experts_per_token,
    int const start_expert_id)
{
    using BlockScan = cub::BlockScan<int, kNumTokensPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    // Grid layout: blockIdx.x = expert_id, blockIdx.y = block_id
    int const target_expert_id = blockIdx.x;
    int const block_id = blockIdx.y;
    int const num_blocks_per_seq = gridDim.y;
    int const token_id = block_id * kNumTokensPerBlock + threadIdx.x;

    // Find if this token is assigned to this expert
    int expanded_token_id = -1;
    if (token_id < num_tokens) {
        for (int i = 0; i < num_experts_per_token; i++) {
            int const expert_id =
                token_selected_experts[token_id * num_experts_per_token + i]
                - start_expert_id;

            if (expert_id == target_expert_id) {
                expanded_token_id = i * num_tokens + token_id;
                break;
            }
        }
    }

    // Compute prefix sum to get write position
    int const has_matched = expanded_token_id >= 0 ? 1 : 0;
    int index;
    BlockScan(temp_storage).ExclusiveSum(has_matched, index);

    // Write to output
    if (has_matched) {
        blocked_row_to_unpermuted_row[
            target_expert_id * num_tokens +
            block_id * kNumTokensPerBlock +
            index
        ] = expanded_token_id;
    }

    // Last thread writes count
    if (threadIdx.x == kNumTokensPerBlock - 1) {
        blocked_expert_counts[
            target_expert_id * num_blocks_per_seq + block_id
        ] = index + has_matched;
    }
}
```

**Algorithm:**
- **Grid**: `(num_experts_per_node, num_blocks_per_seq)`
- **Block**: `(num_tokens_per_block,)`
- Each block processes tokens for one expert in one block
- Uses CUB BlockScan for efficient prefix sum
- Outputs per-block counts and token indices

### Kernel 3: Expand Input Rows

**Purpose:** Permute and expand input tokens based on expert assignment

**Declaration:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h:55-62`](../cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h#L55-L62)

```cpp
template <class InputActivationsType, class ExpandedActivationsType>
void expandInputRowsKernelLauncher(
    InputActivationsType const* unpermuted_input,      // Original token order
    ExpandedActivationsType* permuted_output,          // Permuted token order
    float const* unpermuted_scales,                    // Routing weights (original order)
    float* permuted_scales,                            // Routing weights (permuted order)
    int const* permuted_row_to_unpermuted_row,         // Permutation mapping
    int64_t const num_rows,
    int64_t const hidden_size,
    int const k,                                       // experts_per_token
    int const num_experts_per_node,
    QuantParams const& quant_params,
    bool use_per_expert_act_scale,
    int64_t* expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf,
    bool const swizzled_input_sf,
    void const* prequant_scales,
    cudaStream_t stream
);
```

**What it does (pseudo-code):**

```cpp
__global__ void expandInputRowsKernel(/*...*/) {
    int permuted_row_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (permuted_row_id < num_permuted_rows) {
        // Get original row index
        int unpermuted_row_id = permuted_row_to_unpermuted_row[permuted_row_id];
        int original_token_id = unpermuted_row_id % num_rows;
        int expert_k_idx = unpermuted_row_id / num_rows;

        // Copy data from original to permuted position
        for (int col = 0; col < hidden_size; col++) {
            permuted_output[permuted_row_id * hidden_size + col] =
                unpermuted_input[original_token_id * hidden_size + col];
        }

        // Copy routing weight
        if (unpermuted_scales && permuted_scales) {
            permuted_scales[permuted_row_id] =
                unpermuted_scales[original_token_id * k + expert_k_idx];
        }
    }
}
```

**Key Features:**
- Coalesced reads from original order
- Coalesced writes to permuted order
- Handles top-k > 1 by replicating tokens
- Optionally applies pre-quantization

---

## FP8 Quantization Kernels

### Overview

FP8 quantization converts bf16 activations to FP8 format with block-wise scaling for efficient computation.

### Python Entry

**File:** [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py:211-212`](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L211-L212)

```python
act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(
    permuted_data_tensor  # [num_permuted_tokens, hidden_size] bf16
)
# act_input_fp8: [num_permuted_tokens, hidden_size] fp8
# act_input_sf: scale factors for 1x128 blocks
```

### Torch Binding

**File:** [`cpp/tensorrt_llm/thop/fp8Quantize.cpp:29-84`](../cpp/tensorrt_llm/thop/fp8Quantize.cpp#L29-L84)

```cpp
std::tuple<at::Tensor, at::Tensor> fp8_quantize_1x128(at::Tensor const& self)
{
    // Validate input
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);
    TORCH_CHECK(self.scalar_type() == at::ScalarType::BFloat16,
                "Input matrix dtype must be BF16.");
    TORCH_CHECK(self.dim() == 2, "input must be a matrix");

    auto const m = self.sizes()[0];  // num_tokens
    auto const n = self.sizes()[1];  // hidden_size

    // Create GEMM runner
    auto mGemmRunner =
        tensorrt_llm::kernels::fp8_blockscale_gemm::
            CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>();

    // Pad m dimension to multiple of 4 (SM90 requirement)
    auto const m_padded = (m + 4 - 1) / 4 * 4;

    // Allocate output tensors
    at::Tensor valueE4M3 = at::detail::empty_cuda(
        {m_padded, n},
        at::ScalarType::Float8_e4m3fn,
        self.device(),
        std::nullopt
    );

    // Allocate scale tensor (128-byte aligned)
    int64_t scaleSizeInBytes = mGemmRunner.getActScaleSize(m, n);
    int64_t elementSize = scaleSizeInBytes / torch::elementSize(FP8_BLOCK_SCALING_SF_DTYPE);

    at::Tensor scaleFP8SF = at::detail::empty_cuda(
        {elementSize},
        FP8_BLOCK_SCALING_SF_DTYPE,  // float32
        self.device(),
        std::nullopt
    );

    // Get pointers
    __nv_fp8_e4m3* act_buffer = reinterpret_cast<__nv_fp8_e4m3*>(valueE4M3.data_ptr());
    float* act_scale_buffer = reinterpret_cast<float*>(scaleFP8SF.data_ptr());

    auto stream = at::cuda::getCurrentCUDAStream(self.get_device());

    // Launch quantization kernel
    mGemmRunner.fp8CS1x128(
        act_buffer,
        act_scale_buffer,
        reinterpret_cast<__nv_bfloat16 const*>(self.data_ptr()),
        n,  // shape_x (columns)
        m,  // shape_y (rows)
        stream
    );

    // Handle SM100 post-processing
    if (tensorrt_llm::common::isSM100Family()) {
        auto const num_n_blocks = (n + 127) / 128;
        auto const act_scal_elesize = num_n_blocks * m_padded;

        scaleFP8SF = scaleFP8SF
            .slice(0, 0, act_scal_elesize)
            .view({num_n_blocks, m_padded})
            .slice(1, 0, m)
            .contiguous();
    }

    return {valueE4M3.slice(0, 0, m), scaleFP8SF};
}
```

### C++ GEMM Runner

**File:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu:179-183`](../cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu#L179-L183)

```cpp
template <typename ElementA, typename ElementB, typename ElementD>
void CutlassFp8BlockScaleGemmRunner<ElementA, ElementB, ElementD>::fp8CS1x128(
    __nv_fp8_e4m3* mat_quant,
    float* scales,
    __nv_bfloat16 const* mat,
    int shape_x,  // columns
    int shape_y,  // rows
    cudaStream_t stream)
{
    fp8_1x128_cs(mat_quant, scales, mat, shape_x, shape_y, stream);
}
```

### CUDA Kernel (Conceptual)

The actual kernel `fp8_1x128_cs` is in a `.cuh` file. Here's the algorithm:

```cpp
__global__ void fp8_1x128_cs_kernel(
    __nv_fp8_e4m3* output,
    float* scales,
    __nv_bfloat16 const* input,
    int num_cols,
    int num_rows)
{
    // Each block processes multiple 1x128 blocks
    int row = blockIdx.x;
    int col_block = blockIdx.y * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        int block_idx = col_block / 128;
        int col_in_block = col_block % 128;

        // Step 1: Find max absolute value in this 1x128 block
        __shared__ float shared_max[BLOCK_SIZE / 128];

        if (col_in_block == 0) {
            float max_val = 0.0f;
            for (int c = 0; c < 128 && block_idx * 128 + c < num_cols; c++) {
                float val = fabs(float(input[row * num_cols + block_idx * 128 + c]));
                max_val = fmax(max_val, val);
            }
            shared_max[threadIdx.x / 128] = max_val;
        }
        __syncthreads();

        // Step 2: Compute scale
        float scale = shared_max[threadIdx.x / 128] / FP8_E4M3_MAX;  // 448.0
        if (col_in_block == 0 && block_idx * 128 < num_cols) {
            scales[row * ((num_cols + 127) / 128) + block_idx] = scale;
        }

        // Step 3: Quantize
        if (block_idx * 128 + col_in_block < num_cols) {
            float val = float(input[row * num_cols + block_idx * 128 + col_in_block]);
            __nv_fp8_e4m3 quantized = __nv_fp8_e4m3(val / scale);
            output[row * num_cols + block_idx * 128 + col_in_block] = quantized;
        }
    }
}
```

**Algorithm:**
1. **Partition** each row into blocks of 128 elements
2. **Find max** absolute value in each block
3. **Compute scale** = max / 448.0 (FP8 E4M3 max value)
4. **Quantize** each element: `fp8_val = bf16_val / scale`

**Memory Layout:**
- Input: `[num_rows, num_cols]` bf16
- Output: `[num_rows, num_cols]` fp8
- Scales: `[num_rows, (num_cols + 127) / 128]` float32

---

## Group GEMM Kernels

### Overview

Group GEMM performs batched matrix multiplication where each "group" corresponds to tokens assigned to one expert.

### Python Reference Implementation

**File:** [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py:21-88`](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L21-L88)

```python
def cute_dsl_fp8_group_blockwise_gemm_ref(
    a: torch.Tensor,              # [m, k] - activations (fp8)
    b: torch.Tensor,              # [num_experts, n, k] - weights (fp8)
    a_sf: torch.Tensor,           # activation scales
    b_sf: torch.Tensor,           # weight scales
    offset_array: torch.Tensor,   # [num_experts + 1] - boundaries
) -> torch.Tensor:
    """
    Reference implementation of group blockwise GEMM.

    For each expert i:
        tokens_start = offset_array[i]
        tokens_end = offset_array[i+1]
        output[tokens_start:tokens_end] = matmul(
            dequantize(a[tokens_start:tokens_end], a_sf),
            dequantize(b[i], b_sf[i])
        )
    """
    m, k = a.shape[0], a.shape[1]
    num_experts, n, k = b.shape[0], b.shape[1], b.shape[2]

    # Prepare scales for broadcasting
    # a_sf handles 1x128 groupwise scaling
    # b_sf handles blockwise (128x128) scaling

    def pad_and_multiply(scale, tensor):
        """Apply blockwise/groupwise scaling"""
        cm, ck, _ = scale.shape
        m, k, _ = tensor.shape

        # Determine scaling granularity
        IsGroupWise = (ck == math.ceil(k / 128))
        IsBlockWise = (cm == math.ceil(m / 128))

        # Create indices for broadcasting
        k_idx = torch.arange(k, device=scale.device)
        if IsGroupWise:
            k_idx = k_idx // 128  # Map columns to blocks

        m_idx = torch.arange(m, device=scale.device)
        if IsBlockWise:
            m_idx = m_idx // 128  # Map rows to blocks

        # Broadcast scales
        expanded_scale = scale[m_idx[:, None], k_idx, :]

        # Apply scaling (dequantization)
        result = expanded_scale * tensor.to(torch.float32)
        return result

    # Dequantize activations and weights
    updated_a = pad_and_multiply(input_scale_tmp, a_tmp.to(torch.float32))
    updated_b = pad_and_multiply(weight_scale_tmp, b_tmp.to(torch.float32))

    # Initialize output
    ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)

    # Process each expert group
    for i in range(len(offset_array) - 1):
        start = offset_array[i]
        end = offset_array[i + 1]

        # GEMM for expert i
        ref[start:end, :] = torch.einsum(
            "mk,nk->mn",
            updated_a[start:end, :, 0],  # [num_tokens_for_expert_i, k]
            updated_b[:, :, i]            # [n, k]
        )

    return ref.to(torch.bfloat16)
```

### CuTe DSL Implementation

**File:** [`tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py:137-285`](../tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py#L137-L285)

```python
def forward(self, inputs: List[torch.Tensor], tactic) -> torch.Tensor:
    """
    Execute FP4 GEMM using CuTe DSL with specified tactic.
    """
    if isinstance(tactic, tuple):
        mma_tiler_mn, cluster_shape_mn, swap_ab = tactic
    else:
        mma_tiler_mn, cluster_shape_mn, swap_ab = [(128, 128), (1, 1), False]

    a_tensor, b_tensor, a_sf_tensor, b_sf_tensor = inputs
    m, k, n = a_tensor.shape[0], a_tensor.shape[1], b_tensor.shape[0]

    # Allocate output
    c_tensor = torch.empty(
        *(m, n),
        dtype=self.output_dtype,
        device="cuda"
    )

    # Handle AB swap optimization
    if swap_ab:
        c_tensor = c_tensor.permute(1, 0)

    # Create CuTe DSL pointers
    a_ptr = self.make_cute_dsl_global_pointer(a_tensor, cutlass.Float4E2M1FN, 32)
    b_ptr = self.make_cute_dsl_global_pointer(b_tensor, cutlass.Float4E2M1FN, 32)
    a_sf_ptr = self.make_cute_dsl_global_pointer(a_sf_tensor, cutlass.Float8E4M3FN, 16)
    b_sf_ptr = self.make_cute_dsl_global_pointer(b_sf_tensor, cutlass.Float8E4M3FN, 16)
    c_ptr = self.make_cute_dsl_global_pointer(c_tensor, cutlass.BFloat16, 16)

    # Get CUDA stream
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # Create cache key
    CACHE_KEY = (sf_vec_size, mma_tiler_mn, cluster_shape_mn, swap_ab)

    # Check kernel cache
    if CACHE_KEY not in CuteDSLNVFP4BlackwellLinear.kernel_dict:
        # Create kernel wrapper
        gemm = Sm100BlockScaledPersistentDenseGemmKernelWrapper(
            sf_vec_size,
            mma_tiler_mn,
            cluster_shape_mn,
        )

        # Compute max active clusters
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )

        # Compile kernel with CuTe
        compiled_gemm = cute.compile(
            gemm,
            kernel_m, kernel_n, real_k,
            kernel_sf_m // 128,
            kernel_sf_n // 128,
            sf_k // 4,
            1,  # batch_size
            kernel_a_ptr, kernel_b_ptr,
            kernel_a_sf_ptr, kernel_b_sf_ptr,
            c_ptr,
            self.alpha,
            max_active_clusters,
            stream,
            swap_ab,
        )

        # Cache compiled kernel
        CuteDSLNVFP4BlackwellLinear.kernel_dict[CACHE_KEY] = compiled_gemm
    else:
        compiled_gemm = CuteDSLNVFP4BlackwellLinear.kernel_dict[CACHE_KEY]

    # Launch kernel
    compiled_gemm(
        kernel_m, kernel_n, real_k,
        kernel_sf_m // 128,
        kernel_sf_n // 128,
        sf_k // 4,
        kernel_a_ptr, kernel_b_ptr,
        kernel_a_sf_ptr, kernel_b_sf_ptr,
        c_ptr,
        self.alpha,
        stream,
    )

    if swap_ab:
        c_tensor = c_tensor.permute(1, 0)

    return c_tensor
```

### CUTLASS/CuTe Kernel (High-level)

The actual kernel is generated by CUTLASS/CuTe compilation. Here's the conceptual structure:

```cpp
// Conceptual CuTe DSL Kernel Structure
template <typename MMA_Tiler, typename ClusterShape>
__global__ void cute_dsl_fp8_grouped_gemm_kernel(
    __nv_fp8_e4m3 const* A,           // [total_tokens, k]
    __nv_fp8_e4m3 const* B,           // [num_experts, n, k]
    float const* A_scales,            // Activation scales
    float const* B_scales,            // Weight scales
    __nv_bfloat16* C,                 // [total_tokens, n]
    int64_t const* problem_offsets,   // [num_experts + 1]
    // ... shape parameters ...
)
{
    // Thread block cooperative group
    using TiledMMA = decltype(make_tiled_mma(MMA_Tiler{}));

    // Shared memory
    __shared__ typename TiledMMA::SmemLayoutA smem_a;
    __shared__ typename TiledMMA::SmemLayoutB smem_b;

    // Determine which expert this threadblock processes
    int expert_id = /* computed from blockIdx */;
    int token_start = problem_offsets[expert_id];
    int token_end = problem_offsets[expert_id + 1];

    // Tile iterators
    auto tCrA = make_tensor(/* ... */);  // Register tile for A
    auto tCrB = make_tensor(/* ... */);  // Register tile for B
    auto tCrC = make_tensor(/* ... */);  // Register tile for C (accumulator)

    // Main loop: tile over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A tile from GMEM -> SMEM -> Registers
        // Load B tile from GMEM -> SMEM -> Registers

        // MMA operation
        gemm(tCrC, tCrA, tCrB, tCrC);

        __syncthreads();
    }

    // Apply scaling and store to GMEM
    // output = scale_a * scale_b * accumulator
}
```

**Key CuTe DSL Features:**
- **Compile-time shapes**: All tile sizes known at compile time
- **Layout algebra**: Automatic coalescing and bank conflict avoidance
- **TMA (Tensor Memory Accelerator)**: Hardware-accelerated memory copies on Hopper+
- **Persistent kernels**: Kernel stays resident and processes multiple problems
- **Cluster support**: Thread block clusters for better L2 cache utilization

---

## Token Combine Kernels

### Overview

Token combine unpermutes tokens back to original order and applies routing weights.

### Kernel: Finalize MoE Routing

**Declaration:** [`cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h:64-71`](../cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_util_kernels.h#L64-L71)

```cpp
template <class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(
    GemmOutputType const* expanded_permuted_rows,    // [num_permuted_tokens, hidden_size]
    OutputType* reduced_unpermuted_output,           // [num_tokens, hidden_size]
    ScaleBiasType const* bias,                       // Optional bias
    float const* final_scales,                       // [num_tokens, top_k] routing weights
    int const* unpermuted_row_to_permuted_row,       // [num_tokens * top_k]
    int const* permuted_row_to_unpermuted_row,       // [num_tokens * top_k]
    int const* token_selected_experts,               // [num_tokens, top_k]
    int64_t const* expert_first_token_offset,        // [num_experts + 1]
    int64_t const num_rows,                          // num_tokens
    int64_t const padded_cols,                       // hidden_size (padded)
    int64_t const unpadded_cols,                     // hidden_size (actual)
    int64_t const experts_per_token,                 // top_k
    int64_t const num_experts_per_node,
    MOEParallelismConfig parallelism_config,
    bool const enable_alltoall,
    cudaStream_t stream
);
```

**Kernel Algorithm (Pseudo-code):**

```cpp
__global__ void finalizeMoeRoutingKernel(
    GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output,
    ScaleBiasType const* bias,
    float const* final_scales,
    int const* unpermuted_row_to_permuted_row,
    int const* permuted_row_to_unpermuted_row,
    // ... parameters ...
)
{
    // Each thread processes one element of one token
    int token_id = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (token_id < num_rows && col < unpadded_cols) {
        OutputType result = 0;

        // Accumulate contributions from all experts this token uses
        for (int k = 0; k < experts_per_token; k++) {
            // Find permuted position for this token-expert pair
            int unpermuted_row_id = token_id * experts_per_token + k;
            int permuted_row_id = unpermuted_row_to_permuted_row[unpermuted_row_id];

            // Read from permuted output
            GemmOutputType value = expanded_permuted_rows[
                permuted_row_id * padded_cols + col
            ];

            // Get routing weight
            float scale = final_scales[unpermuted_row_id];

            // Accumulate weighted contribution
            result += OutputType(float(value) * scale);
        }

        // Add bias if present
        if (bias) {
            result += bias[col];
        }

        // Write to final output
        reduced_unpermuted_output[token_id * unpadded_cols + col] = result;
    }
}
```

**Launch Configuration:**

```cpp
// Typical launch config
dim3 grid(num_tokens, (hidden_size + 127) / 128);
dim3 block(128);

finalizeMoeRoutingKernel<<<grid, block, 0, stream>>>(/*...*/);
```

**Algorithm Steps:**
1. **For each token** (outer loop over tokens)
2. **For each expert** assigned to token (inner loop, top_k iterations)
   - Look up permuted position using `unpermuted_row_to_permuted_row`
   - Read expert output from permuted tensor
   - Multiply by routing weight
   - Accumulate
3. **Add bias** (if present)
4. **Write result** to unpermuted output position

**Memory Access Pattern:**
- **Reads**: Scattered (following permutation map)
- **Writes**: Coalesced (sequential token order)
- **Routing weights**: Sequential access per token

### Distributed Expert Parallelism (All-to-All)

When `enable_alltoall=true`, additional communication happens:

```cpp
if (enable_alltoall && parallelism_config.ep_size > 1) {
    // Gather results from all EP ranks
    // Each rank has computed partial results for tokens
    // assigned to its experts

    ncclGroupStart();
    for (int i = 0; i < parallelism_config.ep_size; i++) {
        ncclSend(/* local results */);
        ncclRecv(/* remote results */);
    }
    ncclGroupEnd();

    // Combine results from all ranks
}
```

This enables expert parallelism across multiple GPUs.

---

## Performance Optimizations

### 1. Kernel Fusion

**Fused Operations:**
- Expert map building + sorting (fused kernel)
- Quantization + scaling (single kernel)
- Dequantization + GEMM (via CUTLASS epilogue)

**Benefits:**
- Reduced kernel launches
- Less GMEM traffic
- Better instruction-level parallelism

### 2. Memory Coalescing

**Token Dispatch:**
- Original: Scattered reads, coalesced writes
- Permuted: Coalesced reads for subsequent GEMMs

**Token Combine:**
- Permuted: Coalesced reads
- Original: Coalesced writes

### 3. FP8 Quantization

**Benefits:**
- 2x memory bandwidth reduction
- 2x faster computation (Hopper Tensor Cores)
- Block-wise scaling maintains accuracy

**Overhead:**
- Quantization: ~5-10% of total time
- Dequantization: Fused into GEMM epilogue (free)

### 4. Grouped GEMM

**vs. Loop over Experts:**
- Single kernel launch
- Better instruction cache utilization
- Reduced dispatch overhead

**Persistent Kernels (Hopper+):**
- Kernel stays resident
- Process multiple groups without re-launch
- Better L2 cache utilization with thread block clusters

### 5. Autotuning

**Tuned Parameters:**
- Thread block shape (e.g., 128x128, 256x128)
- Cluster shape (Hopper+)
- Pipeline stages
- Swizzle patterns

**Tuning Strategy:**
- Profile on first run
- Cache optimal configuration
- Adapt to input shape buckets

---

## Summary

This trace shows how TensorRT-LLM's fused MoE achieves high performance through:

1. **Efficient permutation**: Fused kernels for small expert counts, optimized 3-step for large
2. **FP8 quantization**: Block-wise scaling balances speed and accuracy
3. **Grouped GEMM**: Process all experts in single kernel with CuTe DSL
4. **Smart unpermutation**: Coalesced writes, weighted accumulation
5. **Autotuning**: Adapt kernel configuration to hardware and problem size

The implementation leverages modern GPU features (Tensor Cores, TMA, Thread Block Clusters) while maintaining portability through CUTLASS/CuTe abstractions.
