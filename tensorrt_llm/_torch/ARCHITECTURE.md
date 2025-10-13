# TensorRT-LLM Python Runtime Architecture (`tensorrt_llm/_torch`)

> "A comprehensive guide to the inner workings of TensorRT-LLM's PyTorch-based runtime"
>
> Inspired by matklad's [ARCHITECTURE.md](https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html)

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Compilation vs Auto Deploy](#compilation-vs-auto-deploy)
3. [Backend Architecture and torch.compile Integration](#backend-architecture-and-torchcompile-integration)
4. [Custom Ops Dispatch: trtllm_gen_custom_ops](#custom-ops-dispatch-trtllm_gen_custom_ops)
5. [Custom Ops Dispatch: userbuffers_custom_ops](#custom-ops-dispatch-userbuffers_custom_ops)
6. [AutoTuner Deep Dive](#autotuner-deep-dive)
7. [Code Snippets and Usage Examples](#code-snippets-and-usage-examples)

---

## High-Level Overview

The `tensorrt_llm/_torch` directory contains TensorRT-LLM's **PyTorch-based runtime**, which enables:

1. **JIT compilation** via `torch.compile` for graph-level optimization
2. **AutoDeploy** mode for direct PyTorch model execution with selective optimizations
3. **Custom CUDA kernels** exposed as PyTorch operators
4. **Autotuning** for kernel selection and performance optimization
5. **Multi-stream execution** and CUDA graph capture

### Key Components

```
tensorrt_llm/_torch/
├── compilation/          # torch.compile backend and graph optimizations
├── auto_deploy/          # Direct model execution with selective optimization
├── custom_ops/           # PyTorch custom operator definitions
├── autotuner.py          # Kernel autotuning framework
├── attention_backend/    # Attention implementations (FlashInfer, vanilla)
└── models/               # Model implementations and checkpoint loaders
```

**Philosophy**: Bridge PyTorch's flexibility with TensorRT-LLM's high-performance CUDA kernels through a unified compilation and execution framework.

---

## Compilation vs Auto Deploy

### Compilation Mode (`tensorrt_llm/_torch/compilation/`)

**Purpose**: Full graph optimization via `torch.compile` with custom fusion patterns.

**Key File**: [compilation/backend.py](compilation/backend.py)

```python
# compilation/backend.py:23-167
class Backend:
    """torch.compile backend for graph-level optimization."""

    def __init__(self, enable_inductor=True, enable_userbuffers=False,
                 enable_piecewise_cuda_graph=False, ...):
        self.enable_inductor = enable_inductor
        self.piecewise_cuda_graph = enable_piecewise_cuda_graph
        self.custom_passes = Backend.get_custom_pass(enable_userbuffers)
        # Pattern matchers for fusion optimization

    def optimize(self, gm: GraphModule, example_inputs: List[torch.Tensor]):
        """Apply custom fusion patterns + optionally Inductor."""
        for custom_pass in self.custom_passes:
            custom_pass.apply(gm.graph)

        if self.piecewise_cuda_graph:
            return piecewise_optimizer(gm, ...)
        elif self.enable_inductor:
            return compile_fx(gm, example_inputs)  # Inductor backend
        return gm
```

**Workflow**:
1. **Capture**: `torch.compile` captures FX graph from PyTorch model
2. **Pattern Matching**: Apply custom fusion patterns (AllReduce+RMSNorm, etc.)
3. **Inductor (Optional)**: Run PyTorch's Inductor for triton kernel generation
4. **CUDA Graph**: Optionally capture execution in CUDA graphs
5. **Multi-Stream**: Schedule ops across multiple CUDA streams

**Example**:
```python
from tensorrt_llm._torch.compilation import Backend

model = MyTransformer()
backend = Backend(enable_inductor=True, enable_userbuffers=True)
compiled_model = torch.compile(model, backend=backend)

output = compiled_model(input_ids)  # Optimized execution
```

### Auto Deploy Mode (`tensorrt_llm/_torch/auto_deploy/`)

**Purpose**: Direct model execution with **selective** kernel replacement (no full graph compilation).

**Key Files**:
- [auto_deploy/llm.py](auto_deploy/llm.py)
- [auto_deploy/custom_ops/](auto_deploy/custom_ops/)

```python
# auto_deploy/llm.py:103-147
class LLM(_TorchLLM):
    """AutoDeploy: Direct PyTorch execution with custom kernel swaps."""

    def __init__(self, *args, **kwargs):
        kwargs["backend"] = "_autodeploy"  # Bypass compilation
        super().__init__(*args, **kwargs)

    def _build_model(self):
        # Load HuggingFace model directly
        self._prefetch_model()
        # Apply monkey patches to replace ops
        # e.g., replace nn.Linear with custom FP8 GEMM
```

**Workflow**:
1. **Load Model**: Import HuggingFace Transformers model directly
2. **Monkey Patch**: Replace specific ops (Linear, RMSNorm, RoPE) with custom kernels
3. **Export (Optional)**: Use `torch.export` for graph capture without Inductor
4. **Execute**: Run model with custom kernels inline

**Example**:
```python
from tensorrt_llm._torch.auto_deploy import LLM, LlmArgs

args = LlmArgs(model="meta-llama/Llama-2-7b-hf", tp_size=2)
llm = LLM(args)  # No compilation, direct execution

outputs = llm.generate(["Hello world"], max_tokens=100)
```

### Key Differences

| Aspect | Compilation Mode | Auto Deploy Mode |
|--------|------------------|------------------|
| **Trigger** | `torch.compile(model)` | Direct model instantiation |
| **Optimization** | Full graph optimization | Selective op replacement |
| **Inductor** | Optional | Not used |
| **CUDA Graphs** | Supported | Limited |
| **Flexibility** | Lower (static graph) | Higher (dynamic) |
| **Use Case** | Maximum performance | Rapid prototyping, flexibility |

**Mental Model**:
- **Compilation** = "Give me the fastest code, I'll wait for optimization"
- **Auto Deploy** = "Give me working code now, optimize hotspots only"

---

## Backend Architecture and torch.compile Integration

### How torch.compile Works with TensorRT-LLM

```
┌──────────────┐
│ PyTorch Model│
└──────┬───────┘
       │ torch.compile(model, backend=Backend())
       ↓
┌──────────────────────┐
│  FX Graph Capture    │  ← PyTorch's AOTAutograd
└──────┬───────────────┘
       │
       ↓
┌──────────────────────┐
│  Backend.optimize()  │  ← TensorRT-LLM custom patterns
│  - Pattern Matching  │
│  - Fusion (AR+Norm)  │
│  - Multi-stream      │
└──────┬───────────────┘
       │
       ↓ (Optional)
┌──────────────────────┐
│  Inductor Backend    │  ← torch._inductor.compile_fx
│  - Triton codegen    │
└──────┬───────────────┘
       │
       ↓
┌──────────────────────┐
│  Executable Graph    │
└──────────────────────┘
```

### Backend Registration

```python
# Example usage showing the dispatch flow
import torch
from tensorrt_llm._torch.compilation import Backend

# 1. Create backend instance
backend = Backend(
    enable_inductor=True,
    enable_userbuffers=True,  # Enable NCCL user buffers
    max_num_streams=4         # Multi-stream execution
)

# 2. Compile model
model = torch.compile(model, backend=backend, mode="reduce-overhead")

# 3. First call: triggers optimization
output = model(input_ids)  # Backend.__call__() → optimize() → compile_fx()

# 4. Subsequent calls: use cached compiled graph
output2 = model(input_ids2)  # Fast path
```

### Pattern Matching and Fusion

**Key File**: [compilation/patterns/ar_residual_norm.py](compilation/patterns/ar_residual_norm.py)

```python
# Simplified pattern matching for AllReduce + RMSNorm fusion
from torch._inductor.pattern_matcher import PatternMatcherPass

def register_ar_fusions(passes: List[PatternMatcherPass], ub_enabled: bool):
    """Register AllReduce + Norm fusion patterns."""

    # Pattern: AllReduce → Add (residual) → RMSNorm
    def ar_residual_norm_pattern(allreduce, residual, rms_norm):
        return fused_ar_residual_norm(allreduce, residual, rms_norm,
                                      use_userbuffers=ub_enabled)

    passes[0].register_pattern(ar_residual_norm_pattern)
```

**Why this matters**: Transformer decoders repeatedly do `output = norm(input + allreduce(x))`. Fusing these 3 ops:
- Eliminates intermediate memory traffic
- Uses NCCL user buffers for zero-copy AllReduce
- 1.5-2x speedup on multi-GPU

### Multi-Stream Execution

**Key File**: [compilation/multi_stream/auto_multi_stream.py](compilation/multi_stream/auto_multi_stream.py)

```python
def multi_stream_schedule(gm: GraphModule, num_streams: int):
    """Automatically schedule ops across multiple CUDA streams."""

    # Build dependency graph
    for node in gm.graph.nodes:
        deps = get_dependencies(node)

        # Schedule independent ops on different streams
        if can_parallelize(node, deps):
            assign_to_stream(node, next_available_stream())

        # Insert stream synchronization events
        if needs_sync(node):
            insert_event_wait(node)
```

**Example**: In a transformer layer, attention and FFN can run in parallel:

```
Stream 0: [Attention: QKV proj → SDPA → Output proj]
                                                      ↓ sync
Stream 1:    [FFN: Gate → Up → Down]                ↓
                                                     ↓
Stream 0:                            [Residual + Norm]
```

---

## Custom Ops Dispatch: trtllm_gen_custom_ops

### Overview

The `trtllm_gen_custom_ops.py` file contains **autotuned MOE (Mixture-of-Experts) kernels** that bridge Python → C++ → CUDA.

**Key File**: [custom_ops/trtllm_gen_custom_ops.py](custom_ops/trtllm_gen_custom_ops.py)

### Complete Dispatch Path (MoE Runner + Autotuner)

1) Python op definition and autotune

- Python custom ops: [custom_ops/trtllm_gen_custom_ops.py](custom_ops/trtllm_gen_custom_ops.py)
  - FP4: `trtllm::fp4_block_scale_moe_runner`
  - FP8: `trtllm::fp8_block_scale_moe_runner`
  - Mixed: `trtllm::e4m3_mxe2m1_block_scale_moe_runner`, `trtllm::fp8_fp4_block_scale_moe_runner`
- Each op constructs a Python `TunableRunner` (e.g., `FP8BlockScaleMoERunner`) and invokes the autotuner:
  - `AutoTuner.choose_one(...)` picks a `(runner, tactic)` for the current shapes using profiles
    - Autotuner implementation: [autotuner.py: AutoTuner.choose_one](autotuner.py)
    - Timing and warmup: [autotuner.py: _profile_single_kernel](autotuner.py)
    - Delay kernel to reduce host bias: [autotuner.py: delay_kernel](autotuner.py)
    - Cache encoding and persistence: [autotuner.py: AutoTunerProfilingCache](autotuner.py)
- The Python runner then calls the bound Torch class:
  - `torch.classes.trtllm.FP8BlockScaleMoERunner(...).run_moe(..., moeConfigIndex)`

2) PyTorch C++ custom class (THOP) and registration

- C++ THOP runners under `cpp/tensorrt_llm/thop/` expose Torch custom classes:
  - FP8: [thop/fp8BlockScaleMoe.cpp](../../cpp/tensorrt_llm/thop/fp8BlockScaleMoe.cpp)
    - Registers `class_<FP8BlockScaleMoeRunner>("FP8BlockScaleMoERunner")` and `.def("run_moe", ...)`
  - FP4 / FP8FP4: [thop/fp4BlockScaleMoe.cpp](../../cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp)
    - Registers `FP4BlockScaleMoERunner`, `FP8FP4BlockScaleMoERunner` and `.run_moe`
  - Mixed MXE: [thop/mxFp4BlockScaleMoe.cpp](../../cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp)
    - Registers `MxE4m3MxE2m1BlockScaleMoERunner` and `.run_moe`
- Each `.run_moe(...)` assembles `MoERunnerArgs` and forwards to the CUDA MoE runner:
  - `tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner::run(...)`

3) CUDA MoE runner orchestration (routing → GEMM1 → activation → GEMM2 → finalize)

- CUDA-side MoE orchestration: [kernels/trtllmGenKernels/blockScaleMoe/runner.h](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h) and [runner.cu](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu)
- Sequence invoked by `MoE::Runner::run(...)`:
  - Routing/top-k: `Routing::Runner::run(...)` chooses experts per token
    - Implementation: [blockScaleMoe/runner.cu (Routing::Runner::run)](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu)
    - DeepSeekV3 routing kernel: [RoutingDeepSeek.cu](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/RoutingDeepSeek.cu) → `routingDeepSeek::routingMainKernel`
    - Llama4 routing kernel: [RoutingLlama4.cu](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/RoutingLlama4.cu)
    - Renormalize routing kernel: [RoutingRenormalize.cu](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/RoutingRenormalize.cu)
    - Routing data/types: [RoutingKernel.h](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/RoutingKernel.h)
  - Permute + GEMM1 (FC1 with routing): `PermuteGemm1::Runner::run(...)`
    - Wrapper: [blockScaleMoe/runner.h (PermuteGemm1::Runner)](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h)
    - Calls batched GEMM runner: [batchedGemm/KernelRunner.h](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.h) / [KernelRunner.cpp](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.cpp)
    - Final kernel dispatch goes through `BatchedGemmInterface::run(...)`:
      - Interface + metadata: [trtllmGen_bmm_export/BatchedGemmInterface.h](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/BatchedGemmInterface.h)
      - Kernel registry: [trtllmGen_bmm_export/KernelMetaInfo.h](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/KernelMetaInfo.h) (`config.mFunctionName` and CUBINs)
      - Final launch: [trtllm/gen/CudaKernelLauncher.h](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/trtllm/gen/CudaKernelLauncher.h) via `cuLaunchKernelEx`
  - Activation (DeepSeek FP8 only): `moe::dev::activation::run(...)`
    - Kernels: `activationKernel`, `activationDeepSeekKernel` in [blockScaleMoe/DevKernel.cu](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu)
- GEMM2 (FC2): `Gemm2::Runner::run(...)` (same batched GEMM flow as above)
- Finalize (unpermute + weighted combine + dequantize): `moe::dev::finalize::run(...)`
  - Kernels: `finalizeKernel`, `finalizeKernelVecLoad`, `finalizeDeepSeekKernel` in [blockScaleMoe/DevKernel.cu](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu)

Note on tactics/configs: the `moeConfigIndex` chosen by the autotuner indexes `mPassingConfigs` in [blockScaleMoe/runner.cu](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu), which pairs a specific GEMM1 config with a GEMM2 config; both must be valid for the current shape.

4) Config/tactic selection end-to-end

- Python runner `get_valid_tactics(...)` calls into the C++ class to retrieve valid config indices:
  - e.g. [thop/fp8BlockScaleMoe.cpp (getValidConfigs)](../../cpp/tensorrt_llm/thop/fp8BlockScaleMoe.cpp)
  - Bridges to [blockScaleMoe/runner.h (MoE::Runner::getValidConfigIndices)](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h)
  - Each sub-runner queries [batchedGemm/KernelRunner.cpp getValidConfigIndices](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.cpp) for GEMM compatibility
- `AutoTuner.choose_one(...)` picks the fastest `(runner, tactic)` using timing with CUDA events:
  - Profiles via [autotuner.py: _profile_single_kernel](autotuner.py)
  - Inserts a stream delay via [autotuner.py: delay_kernel](autotuner.py)
  - Persists results in JSON via [autotuner.py: AutoTunerProfilingCache](autotuner.py)

5) Final CUDA kernel(s)

- GEMM kernels are looked up by `config.mFunctionName` inside [KernelMetaInfo.h](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/KernelMetaInfo.h), loaded with `cuModuleLoadData`, and launched via [CudaKernelLauncher.h](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/trtllm/gen/CudaKernelLauncher.h).
- Routing and finalize/activation kernels are compiled C++ CUDA kernels in:
  - Routing: [RoutingDeepSeek.cu](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/RoutingDeepSeek.cu), [RoutingRenormalize.cu](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/RoutingRenormalize.cu), [RoutingLlama4.cu](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/RoutingLlama4.cu)
  - Activation/Finalize/Permute: [DevKernel.cu](../../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu)

### Step 1: Python Custom Op Definition

```python
# custom_ops/trtllm_gen_custom_ops.py:235-309
@torch.library.custom_op("trtllm::fp4_block_scale_moe_runner", mutates_args=())
def fp4_block_scale_moe_runner(
    routing_logits: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    ...
) -> List[torch.Tensor]:
    """FP4 quantized MOE with autotuning."""

    # 1. Create runner instance
    kernel_runner = FP4BlockScaleMoERunner(
        num_experts, top_k, intermediate_size, ...
    )

    # 2. Autotune to select best kernel tactic
    tuner = AutoTuner.get()
    _, best_tactic = tuner.choose_one(
        "trtllm::fp4_block_scale_moe_runner",
        [kernel_runner],              # List of runner candidates
        kernel_runner.tuning_config,  # Dynamic shapes to profile
        input_tensors_for_tuner,
    )

    # 3. Execute with best tactic
    return kernel_runner(input_tensors, tactic=best_tactic)
```

**Key Concepts**:
- `@torch.library.custom_op`: Registers op with PyTorch dispatcher
- `TunableRunner`: Python interface for C++ kernel runners
- `AutoTuner.choose_one()`: Profiles multiple implementations, caches best

### Step 2: Fake Tensor Implementation (for `torch.compile`)

```python
# custom_ops/trtllm_gen_custom_ops.py:340-377
@fp4_block_scale_moe_runner.register_fake
def _(routing_logits, hidden_states, ...) -> List[torch.Tensor]:
    """Fake impl for shape inference during torch.compile."""
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1] * 2  # FP4 packing

    # Return dummy tensors with correct shapes
    return [
        hidden_states.new_empty((num_tokens, hidden_size), dtype=torch.bfloat16)
    ]
```

**Why needed**: `torch.compile` uses `FakeTensor` mode to trace graphs without executing kernels. This fake impl provides shape/dtype info.

### Step 3: C++ Torch Library Registration (THOP)

**File**: [cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp](../../cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp)

```cpp
// Line 1-53 (omitted headers, see file)

namespace torch_ext {

class FP4BlockScaleMoeRunner {
public:
    FP4BlockScaleMoeRunner(int64_t tile_tokens_dim)
        : moe_runner_(tile_tokens_dim) {}

    // Returns list of valid kernel configs for given input shape
    std::vector<int64_t> getValidConfigs(
        int64_t top_k, int64_t hidden_size,
        int64_t intermediate_size, int64_t local_num_experts,
        int64_t num_tokens
    ) {
        // Query C++ kernel for supported tactics
        return moe_runner_.getValidConfigs(...);
    }

    // Execute MOE kernel
    std::vector<torch::Tensor> run(
        torch::optional<torch::Tensor> routing_logits,
        torch::Tensor hidden_states,
        ...
        int64_t tactic  // Selected by autotuner
    ) {
        // Setup args from PyTorch tensors
        MoERunnerArgs args;
        args.hidden_states = hidden_states.data_ptr();
        args.num_tokens = hidden_states.sizes()[0];
        ...

        // Launch CUDA kernel
        moe_runner_.run(args, stream, tactic);

        return {output_tensor};
    }

private:
    // C++ kernel runner (contains CUDA kernel logic)
    tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner moe_runner_;
};

} // namespace torch_ext

// Register with PyTorch
TORCH_LIBRARY_FRAGMENT(trtllm, m) {
    m.class_<torch_ext::FP4BlockScaleMoeRunner>("FP4BlockScaleMoERunner")
        .def(torch::init<int64_t>())
        .def("get_valid_configs", &FP4BlockScaleMoeRunner::getValidConfigs)
        .def("run_moe", &FP4BlockScaleMoeRunner::run);
}
```

**Key Points**:
- `TORCH_LIBRARY_FRAGMENT`: Extends the `trtllm` namespace
- `m.class_<>()`: Exposes C++ class to Python as `torch.classes.trtllm.FP4BlockScaleMoERunner`
- `.def(torch::init<>())`: Constructor
- `.def("method", &Class::method)`: Bind C++ methods

### Step 4: Python Runner Integration with AutoTuner

```python
# custom_ops/trtllm_gen_custom_ops.py:56-233
class FP4BlockScaleMoERunner(TunableRunner):
    """Python wrapper around C++ runner for autotuning."""

    runner_dict = dict()  # Cache C++ runner instances

    def get_runner(self, num_tokens: int):
        """Lazily instantiate C++ runner."""
        tile_tokens_dim = calculate_tile_tokens_dim(num_tokens, ...)

        if tile_tokens_dim not in self.runner_dict:
            # Create C++ runner (binds to TORCH_LIBRARY class)
            self.runner_dict[tile_tokens_dim] = \
                torch.classes.trtllm.FP4BlockScaleMoERunner(tile_tokens_dim)

        return self.runner_dict[tile_tokens_dim]

    def get_valid_tactics(self, inputs, profile, **kwargs):
        """Query C++ runner for available kernel configs."""
        kernel_runner = self.get_runner(num_tokens)
        return kernel_runner.get_valid_configs(
            top_k, hidden_size, intermediate_size, ...
        )

    def forward(self, inputs: List[torch.Tensor], tactic: int = -1):
        """Execute kernel with selected tactic."""
        kernel_runner = self.get_runner(num_tokens)
        return kernel_runner.run_moe(..., tactic)
```

### Step 5: CUDA Kernel Launch (C++ → CUDA)

**Simplified C++ kernel launcher** (actual code in `cpp/tensorrt_llm/kernels/trtllmGenKernels/`):

```cpp
namespace tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE {

class Runner {
public:
    void run(MoERunnerArgs const& args, cudaStream_t stream, int tactic) {
        // 1. Select kernel based on tactic
        auto kernel_config = getKernelConfig(tactic);

        // 2. Launch routing kernel (top-k selection)
        launchRoutingKernel(
            args.routing_logits, args.topk_ids, args.topk_weights,
            args.num_tokens, args.num_experts, args.top_k,
            stream
        );

        // 3. Launch MOE GEMM kernel
        launchMoeGemmKernel(
            args.hidden_states, args.gemm1_weights, args.gemm2_weights,
            args.topk_ids, kernel_config,
            stream
        );
    }

private:
    // Actual CUDA kernels (in .cu files)
    void launchMoeGemmKernel(...);
};

} // namespace
```

### Complete Example: End-to-End Trace

```python
# User code
import torch
from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops

hidden_states = torch.randn(1024, 4096, dtype=torch.float8_e4m3fn, device='cuda')
routing_logits = torch.randn(1024, 64, dtype=torch.bfloat16, device='cuda')
gemm1_weights = torch.randn(64, 4096, 14336, dtype=torch.float8_e4m3fn, device='cuda')
gemm2_weights = torch.randn(64, 14336, 4096, dtype=torch.float8_e4m3fn, device='cuda')

# This single call triggers:
# 1. Python: fp4_block_scale_moe_runner()
# 2. AutoTuner: Profile all valid tactics
# 3. C++: FP4BlockScaleMoeRunner::run()
# 4. CUDA: Launch kernel with best tactic
output = torch.ops.trtllm.fp4_block_scale_moe_runner(
    routing_logits, None, hidden_states, None,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    output1_scale, output1_scale_gate, output2_scale,
    num_experts=64, top_k=8, intermediate_size=14336,
    local_expert_offset=0, local_num_experts=64,
    routed_scaling_factor=1.0, routing_method_type=0,
    do_finalize=True
)
```

**Trace**:
1. `torch.ops.trtllm.fp4_block_scale_moe_runner(...)` → PyTorch dispatcher
2. Dispatcher → `TORCH_LIBRARY_FRAGMENT(trtllm)` lookup
3. No CUDA impl found → fallback to Python `@custom_op` definition
4. Python → `FP4BlockScaleMoERunner.forward()` → `torch.classes.trtllm.FP4BlockScaleMoERunner.run_moe()`
5. C++ → `FP4BlockScaleMoeRunner::run()` → CUDA kernel launch
6. CUDA kernel executes on GPU
7. Return path: CUDA → C++ → Python → User

### Registered Ops in trtllm_gen_custom_ops.py

| Op Name | Kernel | Description |
|---------|--------|-------------|
| `trtllm::fp4_block_scale_moe_runner` | FP4 MOE GEMM | E2M1 quantized MOE |
| `trtllm::fp8_block_scale_moe_runner` | FP8 MOE GEMM | E4M3 quantized MOE |
| `trtllm::mxe4m3_mxe2m1_block_scale_moe_runner` | MX FP4/FP8 MOE | Mixed precision MOE |
| `trtllm::e4m3_mxe2m1_block_scale_moe_runner` | E4M3/MX E2M1 MOE | E4M3 input, MX E2M1 weights |
| `trtllm::bf16_mxe2m1_block_scale_moe_runner` | BF16/MX E2M1 MOE | BF16 input, MX E2M1 weights |
| `trtllm::fp8_fp4_block_scale_moe_runner` | FP8/FP4 MOE | FP8 input, FP4 weights |

---

## Custom Ops Dispatch: userbuffers_custom_ops

### Overview

User buffers enable **zero-copy collective communication** by allocating tensors in NCCL-registered memory.

**Key File**: [custom_ops/userbuffers_custom_ops.py](custom_ops/userbuffers_custom_ops.py)

### The Problem

Standard tensor allocation:
```python
x = torch.randn(1024, 4096, device='cuda')  # Regular CUDA memory
nccl_allreduce(x)  # Requires copy to NCCL buffer → overhead!
```

User buffers solution:
```python
ub_tensor = torch.ops.trtllm.create_userbuffers_tensor([1024, 4096], torch.float16)
nccl_allreduce(ub_tensor)  # Zero-copy! Already in NCCL memory
```

### Python Custom Ops

```python
# custom_ops/userbuffers_custom_ops.py:1-45
import torch

# Op 1: Allocate tensor in user buffer memory
@torch.library.custom_op("trtllm::copy_to_userbuffers", mutates_args=())
def copy_to_userbuffers(a: torch.Tensor) -> torch.Tensor:
    """Copy tensor to NCCL user buffer memory."""
    ub_tensor = torch.ops.trtllm.create_userbuffers_tensor(a.shape, a.dtype)
    return ub_tensor.copy_(a, non_blocking=True)

@copy_to_userbuffers.register_fake
def _(a) -> torch.Tensor:
    return torch.empty_like(a)


# Op 2: Add with output in user buffer (avoid in-place for torch.compile)
@torch.library.custom_op("trtllm::add_to_ub", mutates_args=())
def add_to_ub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add two tensors, store result in user buffer."""
    shape = [max(i, j) for i, j in zip(a.shape, b.shape)]
    ub_tensor = torch.ops.trtllm.create_userbuffers_tensor(shape, a.dtype)
    return torch.add(a, b, out=ub_tensor)

@add_to_ub.register_fake
def _(a, b) -> torch.Tensor:
    shape = [max(i, j) for i, j in zip(a.shape, b.shape)]
    return a.new_empty(shape, dtype=a.dtype)


# Op 3: Matmul with output in user buffer
@torch.library.custom_op("trtllm::matmul_to_ub", mutates_args=())
def matmul_to_ub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matmul with result in user buffer."""
    shape = list(a.shape)
    shape[-1] = b.shape[-1]
    ub_tensor = torch.ops.trtllm.create_userbuffers_tensor(shape, a.dtype)
    return torch.matmul(a, b, out=ub_tensor)

@matmul_to_ub.register_fake
def _(a, b) -> torch.Tensor:
    shape = list(a.shape)
    shape[-1] = b.shape[-1]
    return a.new_empty(shape, dtype=a.dtype)
```

### C++ Implementation: create_userbuffers_tensor

**File**: [cpp/tensorrt_llm/thop/userbuffersTensor.cpp](../../cpp/tensorrt_llm/thop/userbuffersTensor.cpp)

```cpp
// userbuffersTensor.cpp:16-53
namespace torch_ext {

std::pair<torch::Tensor, tensorrt_llm::runtime::ub::UBBuffer>
create_userbuffers_tensor(at::IntArrayRef shape, torch::ScalarType dtype) {
    // 1. Calculate buffer size
    int64_t buffer_size = std::accumulate(
        shape.begin(), shape.end(), 1, std::multiplies<int64_t>()
    ) * torch::elementSize(dtype);

    // 2. Allocate from NCCL user buffer pool
    auto [ptr, ub] = tensorrt_llm::runtime::ub::UserBuffersManager
        ::get_instance()
        .allocate_userbuffers(buffer_size);

    // 3. Wrap as PyTorch tensor with custom deleter
    auto& deleter = ptr.get_deleter();
    return std::make_pair(
        torch::from_blob(
            ptr.release(), shape, deleter,
            torch::dtype(dtype).device(torch::kCUDA)
        ),
        ub  // UBBuffer handle (metadata)
    );
}

// Simplified interface for PyTorch (only return tensor)
torch::Tensor create_userbuffers_tensor_op(
    at::IntArrayRef shape, torch::ScalarType dtype
) {
    return create_userbuffers_tensor(shape, dtype).first;
}

} // namespace torch_ext

// Register with PyTorch
TORCH_LIBRARY_FRAGMENT(trtllm, m) {
    m.def("create_userbuffers_tensor", &torch_ext::create_userbuffers_tensor_op);
}
```

### User Buffers Manager (C++)

**File**: [cpp/tensorrt_llm/kernels/userbuffers/userbuffersManager.h](../../cpp/tensorrt_llm/kernels/userbuffers/userbuffersManager.h)

```cpp
namespace tensorrt_llm::runtime::ub {

class UserBuffersManager {
public:
    static UserBuffersManager& get_instance();  // Singleton

    // Allocate from user buffer pool
    std::pair<std::unique_ptr<void, Deleter>, UBBuffer>
    allocate_userbuffers(size_t bytes) {
        // 1. Get buffer from NCCL
        UBBuffer ub = ub_allocate(bytes);

        // 2. Return smart pointer with custom deleter
        auto ptr = std::unique_ptr<void, Deleter>(
            ub.addr,
            [ub](void*) { ub_deallocate(ub.addr); }
        );

        return {std::move(ptr), ub};
    }
};

// NCCL interface
UBBuffer ub_allocate(size_t bytes);
void ub_deallocate(void* addr);
bool ub_supported();

} // namespace
```

### Usage in Fusion Patterns

**File**: [cpp/tensorrt_llm/thop/allreduceOp.cpp](../../cpp/tensorrt_llm/thop/allreduceOp.cpp)

```cpp
// Fused AllReduce + RMSNorm using user buffers
torch::Tensor allreduce_residual_norm_userbuffers(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight
) {
    // 1. Allocate output in user buffer
    auto [norm_out, ub_buffer] = torch_ext::create_userbuffers_tensor(
        input.sizes(), input.scalar_type()
    );

    // 2. Launch fused kernel: AllReduce + Add + RMSNorm
    // - AllReduce writes directly to norm_out (user buffer)
    // - Add residual in-place
    // - Apply RMSNorm
    launch_fused_ar_residual_norm_kernel(
        input.data_ptr(), residual.data_ptr(), weight.data_ptr(),
        norm_out.data_ptr(),  // ← user buffer, NCCL can write directly
        ub_buffer,            // ← NCCL handle
        stream
    );

    return norm_out;  // Already in user buffer memory
}
```

### Python Binding for User Buffers API

**File**: [cpp/tensorrt_llm/pybind/userbuffers/bindings.cpp](../../cpp/tensorrt_llm/pybind/userbuffers/bindings.cpp)

```cpp
// bindings.cpp:28-50
void UserBufferBindings::initBindings(pybind11::module_& m) {
    // UBBuffer metadata class
    py::class_<tub::UBBuffer>(m, "UBBuffer")
        .def_readonly("size", &tub::UBBuffer::size)
        .def_property_readonly("addr",
            [](tub::UBBuffer& self) { return reinterpret_cast<intptr_t>(self.addr); })
        .def_readonly("handle", &tub::UBBuffer::handle);

    // User buffer functions
    m.def("ub_initialize", [](int tp_size) { tub::ub_initialize(tp_size); });
    m.def("ub_allocate", [](size_t bytes) { return tub::ub_allocate(bytes); });
    m.def("ub_deallocate", [](intptr_t addr) {
        return tub::ub_deallocate(reinterpret_cast<void*>(addr));
    });
    m.def("ub_supported", &tub::ub_supported);
}
```

Accessible in Python:
```python
from tensorrt_llm.bindings.internal import userbuffers

# Check support
if userbuffers.ub_supported():
    # Initialize for TP=4
    userbuffers.ub_initialize(tp_size=4)

    # Allocate buffer
    ub = userbuffers.ub_allocate(1024 * 1024)  # 1MB
    print(f"Allocated UB at {ub.addr}, handle={ub.handle}")
```

### Complete Dispatch Path for User Buffers

```
Python: torch.ops.trtllm.copy_to_userbuffers(tensor)
    ↓
PyTorch Dispatcher
    ↓
Python: copy_to_userbuffers() → create_userbuffers_tensor()
    ↓
C++: TORCH_LIBRARY_FRAGMENT(trtllm, m).def("create_userbuffers_tensor", ...)
    ↓
C++: create_userbuffers_tensor_op() → UserBuffersManager::allocate_userbuffers()
    ↓
C++: ub_allocate() → NCCL user buffer API
    ↓
NCCL: allocate from registered memory pool
    ↓
C++: return torch::from_blob(ptr, ...) ← Wraps as PyTorch tensor
    ↓
Python: returns tensor backed by user buffer memory
```

### Why This Matters: Performance Impact

**Without user buffers**:
```python
x = torch.randn(8192, 4096, device='cuda')  # Regular memory
# AllReduce internally does:
# 1. Copy x to NCCL buffer
# 2. AllReduce
# 3. Copy result back to x
result = dist.all_reduce(x)  # 2x memory copy overhead
```

**With user buffers**:
```python
x = torch.ops.trtllm.copy_to_userbuffers(
    torch.randn(8192, 4096, device='cuda')
)
# AllReduce directly on user buffer:
# 1. AllReduce (no copy!)
result = dist.all_reduce(x)  # Zero-copy!
```

**Measured speedup**: 1.3-1.5x for AllReduce-heavy workloads (Transformers with TP).

---

## AutoTuner Deep Dive

The AutoTuner enables **runtime kernel selection** based on empirical profiling.

**Key File**: [autotuner.py](autotuner.py)

### Architecture

```
┌──────────────────┐
│  Custom Op Call  │  fp4_block_scale_moe_runner(...)
└────────┬─────────┘
         │
         ↓
┌─────────────────────────┐
│  AutoTuner.choose_one() │
│  - Cache lookup          │
│  - Profile runners       │
│  - Select best tactic    │
└────────┬────────────────┘
         │
         ↓ (if cache hit)
┌─────────────────────────┐
│  Return cached result   │
│  (runner_id, tactic)    │
└─────────────────────────┘
         │
         ↓ (if cache miss)
┌─────────────────────────┐
│  Profile all tactics    │
│  1. Warmup (3 iters)    │
│  2. Delay stream        │ ← Eliminate host overhead
│  3. Time (10 iters)     │
│  4. Select minimum      │
└────────┬────────────────┘
         │
         ↓
┌─────────────────────────┐
│  Cache result           │
│  key: (op, runner,      │
│        hash(attrs),     │
│        input_shapes)    │
└─────────────────────────┘
```

### Python API

```python
# autotuner.py:542-649
class AutoTuner:
    """Singleton autotuner for kernel selection."""

    def choose_one(
        self,
        custom_op: str,                    # Op name
        runners: List[TunableRunner],      # Candidate implementations
        tuning_config: TuningConfig,       # Dynamic shape specs
        inputs: List[torch.Tensor],        # Input tensors
        **kwargs
    ) -> Tuple[TunableRunner, int]:
        """Select best runner and tactic via profiling."""

        input_shapes = self._get_input_sizes(inputs)

        # 1. Check cache
        is_cache_hit, runner_id, tactic, _ = \
            self.profiling_cache.search_cache(
                custom_op, runners, input_shapes, tuning_config
            )

        if not self.is_tuning_mode or is_cache_hit:
            return (runners[runner_id], tactic)

        # 2. Generate optimization profiles (different input shapes)
        profiles = self._optimization_profiles(tuning_config, inputs)

        # 3. Profile each profile
        for profile in profiles:
            tensors = self._prepare_input_tensors(profile, inputs)

            # Profile all runners and tactics
            runner_id, tactic, min_time, _ = self._profile_runners(
                custom_op, runners, tensors, profile, tuning_config
            )

            # Cache result
            cache_key = self.profiling_cache.get_cache_key(...)
            self.profiling_cache[cache_key] = (runner_id, tactic, min_time)

        # 4. Return best for current input shape
        _, runner_id, tactic, _ = self.profiling_cache.search_cache(...)
        return (runners[runner_id], tactic)
```

### Profiling Implementation

```python
# autotuner.py:719-767
def _profile_single_kernel(
    self,
    runner: TunableRunner,
    inputs: List[torch.Tensor],
    tactic: int,
    **kwargs
) -> float:
    """Profile a single kernel tactic."""

    stream = torch.cuda.current_stream()

    # 1. Warmup (avoid cold start overhead)
    for _ in range(self.warmup):
        runner(inputs, tactic=tactic, **kwargs)
    stream.synchronize()

    # 2. Delay kernel to eliminate host overhead
    #    This is critical: host Python overhead can dominate
    #    for small kernels, skewing measurements.
    delay_kernel(self.stream_delay_micro_secs, stream)

    # 3. Time execution
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream=stream)
    for _ in range(self.repeat):
        runner(inputs, tactic=tactic, **kwargs)
    end.record(stream=stream)
    stream.synchronize()

    avg_time = start.elapsed_time(end) / self.repeat
    return avg_time
```

### C++ Delay Kernel (Critical for Accurate Profiling)

**File**: `cpp/tensorrt_llm/kernels/delayStream.cu`

```cpp
// delayStream.cu:23-36
__global__ void delayStreamKernel(long long delay_micro_secs) {
    for (int i = 0; i < delay_micro_secs; ++i) {
        // Largest __nanosleep is 1ms, so loop for longer delays
        __nanosleep(1000);  // 1 microsecond
    }
}

void invokeDelayStreamKernel(long long delay_micro_secs, cudaStream_t stream) {
    delayStreamKernel<<<1, 1, 0, stream>>>(delay_micro_secs);
}
```

**Bound to Python**:
```python
from tensorrt_llm.bindings.internal.runtime import delay_kernel

# Usage in autotuner.py:750
delay_kernel(1000, stream)  # 1ms delay before profiled kernel
```

**Why this matters**: Without the delay, host-side overhead (Python interpreter, cudaLaunchKernel, etc.) dominates timing for fast kernels (<100μs), leading to incorrect tactic selection. The delay kernel "pushes" the profiled kernel launch far enough from the timing start that host overhead is negligible.

### Tuning Configuration and Dynamic Shapes

```python
# autotuner.py:22-99
@dataclass(slots=True, unsafe_hash=True)
class DynamicTensorSpec:
    """Specification for a dynamic tensor dimension."""
    input_idx: int                                # Which input tensor
    dim_idx: int                                  # Which dimension
    gen_tuning_buckets: Union[Tuple[int], Callable]  # Values to profile
    map_to_tuning_buckets: Callable              # Rounding function for inference

@dataclass(kw_only=True)
class TuningConfig:
    """Configuration for autotuning."""
    dynamic_tensor_specs: Tuple[DynamicTensorSpec, ...]
    constraint_specs: Tuple[ConstraintSpec, ...]  # Dependent dimensions
    tune_max_num_tokens: int = None


# Example from trtllm_gen_custom_ops.py:162-174
class FP4BlockScaleMoERunner(TunableRunner):

    @classmethod
    def get_dynamic_tensor_specs(cls) -> Tuple[DynamicTensorSpec, ...]:
        HIDDEN_STATES_IDX = 2
        TUNED_DIM = 0  # num_tokens dimension
        MAX_PROFILE_BUCKET = 4096

        # Profile at powers of 2: [8, 16, 32, 64, ..., 4096]
        m_values = get_last_power_of_2_num_tokens_buckets(MAX_PROFILE_BUCKET)

        # At inference, round num_tokens to nearest power of 2
        round_rule = lambda x: min(last_positive_power_of_2(x), MAX_PROFILE_BUCKET)

        return (DynamicTensorSpec(
            HIDDEN_STATES_IDX, TUNED_DIM,
            m_values, round_rule
        ),)
```

### Cache Serialization

```python
# autotuner.py:380-508
class AutoTunerProfilingCache:

    def save_cache(self, file_path: Path) -> None:
        """Save cache to JSON."""
        serializable = self._serialize_cache_to_json()
        with open(file_path, 'w') as f:
            json.dump(serializable, f, indent=2)

    def _serialize_cache_to_json(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "lib_version": self.lib_version,
                "device_name": torch.cuda.get_device_name(),
                "device_capability": torch.cuda.get_device_capability(),
            },
            "cache_data": {
                str(key): {
                    "runner_id": runner_id,
                    "tactic": tactic,
                    "min_time": min_time
                }
                for key, (runner_id, tactic, min_time) in self.cache.items()
            }
        }
```

**Example cache file**:
```json
{
  "metadata": {
    "lib_version": "0.13.0",
    "device_name": "NVIDIA H100 80GB HBM3",
    "device_capability": [9, 0]
  },
  "cache_data": {
    "('trtllm::fp4_block_scale_moe_runner', 'FP4BlockScaleMoERunner', -123456789, ((1024, 4096), (1024, 64), ...))": {
      "runner_id": 0,
      "tactic": 42,
      "min_time": 0.234
    }
  }
}
```

### Heuristics and Algorithms

1. **Tactic Selection**: Exhaustive search over valid tactics, selecting minimum time
   - No ML-based search (yet)
   - Tactics are kernel configs (CTA tile size, warp arrangement, etc.)

2. **Profile Generation**: Cartesian product of dynamic dimension values
   ```python
   # autotuner.py:827-849
   dim_grids = itertools.product(*[spec.gen_tuning_buckets for spec in specs])
   for opt_point in dim_grids:
       profile = create_profile(opt_point)
       generated_profiles.append(profile)
   ```

3. **Nearest Profile Matching** (inference time):
   ```python
   # autotuner.py:852-891
   def _find_nearest_profile(shapes, dynamic_tensor_specs):
       """Round input shapes to nearest profiled config."""
       base_profile = list(list(shape) for shape in shapes)

       for spec in dynamic_tensor_specs:
           # Apply rounding function
           base_profile[spec.input_idx][spec.dim_idx] = \
               spec.map_to_tuning_buckets(
                   base_profile[spec.input_idx][spec.dim_idx]
               )

       return tuple(tuple(shape) for shape in base_profile)
   ```
   Example: Input with 1000 tokens → rounds to 1024 (nearest power of 2)

4. **Tile Token Dimension Calculation** (MOE-specific):
   ```python
   # trtllm_gen_custom_ops.py:16-35
   def calculate_tile_tokens_dim(num_tokens, num_experts, top_k):
       """Calculate optimal CTA tile size for MOE kernel."""
       # Assume perfect load balancing
       tokens_per_expert = num_tokens * top_k // num_experts

       # Round to next power of 2
       tile_tokens_dim = next_positive_power_of_2(tokens_per_expert)

       # Special case: use 192 for 128-256 range
       if 128 < tokens_per_expert < 256:
           tile_tokens_dim = 192

       # Clamp to [8, 128]
       return min(max(tile_tokens_dim, 8), 128)
   ```

---

## Code Snippets and Usage Examples

### Example 1: Using Compilation Mode with Custom Patterns

```python
import torch
from tensorrt_llm._torch.compilation import Backend
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = model.cuda().half()

# Create backend with user buffers and multi-stream
backend = Backend(
    enable_inductor=True,
    enable_userbuffers=True,   # Zero-copy collectives
    enable_piecewise_cuda_graph=True,  # CUDA graphs
    capture_num_tokens=[128, 256, 512, 1024],  # Graph capture points
    max_num_streams=4          # 4-way stream parallelism
)

# Compile model
compiled_model = torch.compile(model, backend=backend, mode="reduce-overhead")

# First call: trigger compilation + autotuning
input_ids = torch.randint(0, 32000, (1, 128), device='cuda')
with torch.no_grad():
    output = compiled_model(input_ids)  # ~30s compilation + tuning

# Subsequent calls: fast path
output = compiled_model(input_ids)  # <5ms
```

### Example 2: Using AutoDeploy Mode

```python
from tensorrt_llm._torch.auto_deploy import LLM, LlmArgs

# Configure AutoDeploy
args = LlmArgs(
    model="meta-llama/Llama-2-7b-hf",
    tp_size=2,                # Tensor parallelism
    enable_chunked_prefill=True,
    kv_cache_type="paged",
    max_batch_size=64,
)

# Initialize LLM (no compilation)
llm = LLM(args)

# Generate text
outputs = llm.generate(
    ["Hello, my name is", "Once upon a time"],
    max_tokens=100,
    temperature=0.7,
)

for output in outputs:
    print(output.text)
```

### Example 3: Using AutoTuner Directly

```python
import torch
from tensorrt_llm._torch.autotuner import (
    AutoTuner, TunableRunner, TuningConfig, DynamicTensorSpec,
    autotune
)

class MyCustomRunner(TunableRunner):
    def get_valid_tactics(self, inputs, profile, **kwargs):
        # Return list of valid tactic IDs
        return [0, 1, 2, 3]

    def forward(self, inputs, tactic=-1, **kwargs):
        # Execute kernel with selected tactic
        if tactic == 0:
            return self.kernel_impl_0(inputs)
        elif tactic == 1:
            return self.kernel_impl_1(inputs)
        # ...

# Configure tuning
tuning_config = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            input_idx=0, dim_idx=0,
            gen_tuning_buckets=(32, 64, 128, 256, 512),
            map_to_tuning_buckets=lambda x: ((x + 31) // 32) * 32
        ),
    )
)

# Tune and cache
with autotune(tune_mode=True, cache_path="./autotune_cache.json"):
    tuner = AutoTuner.get()

    inputs = [torch.randn(100, 512, device='cuda')]
    runner, tactic = tuner.choose_one(
        "my_custom_op",
        [MyCustomRunner()],
        tuning_config,
        inputs,
    )

    # Execute with best tactic
    output = runner(inputs, tactic=tactic)

# Later: load cache and run inference
with autotune(tune_mode=False, cache_path="./autotune_cache.json"):
    output = runner(inputs, tactic=tactic)  # Uses cached tactic
```

### Example 4: Registering a Custom Op with C++ Kernel

**Python side** (`my_custom_op.py`):
```python
import torch

@torch.library.custom_op("mylib::my_gemm", mutates_args=())
def my_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Custom GEMM kernel."""
    runner = torch.classes.mylib.MyGemmRunner()
    return runner.run(a, b)

@my_gemm.register_fake
def _(a, b) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
```

**C++ side** (`my_gemm_op.cpp`):
```cpp
#include <torch/extension.h>

namespace my_ns {

class MyGemmRunner {
public:
    torch::Tensor run(torch::Tensor a, torch::Tensor b) {
        // Launch CUDA kernel
        auto output = torch::empty({a.size(0), b.size(1)}, a.options());
        launch_my_gemm_kernel(
            a.data_ptr<float>(), b.data_ptr<float>(),
            output.data_ptr<float>(),
            a.size(0), a.size(1), b.size(1),
            at::cuda::getCurrentCUDAStream()
        );
        return output;
    }
};

} // namespace my_ns

TORCH_LIBRARY(mylib, m) {
    m.class_<my_ns::MyGemmRunner>("MyGemmRunner")
        .def(torch::init<>())
        .def("run", &my_ns::MyGemmRunner::run);
}
```

**CMakeLists.txt**:
```cmake
torch_library(my_custom_op
    MODULE
    my_gemm_op.cpp
    my_gemm_kernel.cu
)
```

### Example 5: Attention Backend Selection

```python
import torch
from tensorrt_llm._torch.attention_backend import get_attention_backend

# Configure attention backend
config = AttentionConfig(
    backend="flashinfer",  # or "vanilla"
    num_heads=32,
    head_dim=128,
    use_paged_kv=True,
    page_size=16,
)

# Get backend instance
attn_backend = get_attention_backend(config)

# Run attention
q = torch.randn(1024, 32, 128, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1024, 32, 128, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1024, 32, 128, device='cuda', dtype=torch.bfloat16)

output = attn_backend.forward(q, k, v, kv_cache, ...)
```

### Example 6: Minimal User Buffer Usage

```python
import torch
from tensorrt_llm.bindings.internal import userbuffers

# Initialize user buffers for TP=4
if userbuffers.ub_supported():
    userbuffers.ub_initialize(tp_size=4)

# Allocate tensor in user buffer
def alloc_ub_tensor(shape, dtype):
    return torch.ops.trtllm.create_userbuffers_tensor(shape, dtype)

# Use in communication
x = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
ub_x = torch.ops.trtllm.copy_to_userbuffers(x)

# AllReduce with zero-copy
import torch.distributed as dist
dist.all_reduce(ub_x)  # No intermediate copy!

# Add residual with result in user buffer
residual = torch.randn_like(ub_x)
result = torch.ops.trtllm.add_to_ub(ub_x, residual)
```

---

## Summary

The `tensorrt_llm/_torch` runtime provides:

1. **Two Execution Modes**:
   - **Compilation**: Full graph optimization via `torch.compile` + custom patterns
   - **AutoDeploy**: Direct execution with selective kernel replacement

2. **Custom Ops System**:
   - Python: `@torch.library.custom_op` for PyTorch registration
   - C++: `TORCH_LIBRARY_FRAGMENT` for binding to CUDA kernels
   - Dispatch: Python → PyTorch → C++ → CUDA

3. **AutoTuner**:
   - Profiles multiple kernel tactics at different input shapes
   - Caches best tactic per (op, runner, shape) tuple
   - Uses delay kernel to eliminate host overhead

4. **User Buffers**:
   - Zero-copy NCCL communication
   - Allocate tensors in NCCL-registered memory
   - Fuse AllReduce + Norm for 1.5x speedup

**Mental Model**: Think of `tensorrt_llm/_torch` as a **bridge** between PyTorch's eager execution and TensorRT's static optimization, with autotuning as the "glue" that selects the best kernel at runtime.

---

## Further Reading

- PyTorch Custom Operators: https://pytorch.org/docs/stable/notes/custom_operators.html
- torch.compile Deep Dive: https://pytorch.org/docs/stable/torch.compiler.html
- NCCL User Buffers: https://github.com/NVIDIA/nccl/issues/650
- TensorRT-LLM C++ Kernels: `cpp/tensorrt_llm/kernels/`

**Last Updated**: 2025-10-13
**Maintainer**: TensorRT-LLM Team
