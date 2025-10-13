**TensorRT-LLM _torch Architecture**

- Scope: Python-side runtime in `tensorrt_llm/_torch` — compilation, backends, custom ops, and autotuning.
- Goal: Explain how torch.compile is used across compile paths, how runtime “backends” are structured, how custom ops flow Python → C++ → CUDA, and how autotuning selects kernel tactics. Includes minimal runnable snippets in this folder.

**High-Level Map**
- `compilation/` — Custom torch.compile backend and graph passes used inside the library.
- `auto_deploy/` — Pluggable compile backends for “in-framework” deployment (torch.compile, CUDA graphs, or both).
- `attention_backend/` — Attention kernel backends (TRTLLM, FLASHINFER, VANILLA) used by high-level modules.
- `custom_ops/` — Python shims for custom ops, including those backed by TRT-LLM Gen kernels and UserBuffers.
- `autotuner.py` — Python autotuner that profiles multiple tactics and caches the fastest.
- `pyexecutor/` — Python runtime/executor that manages batching, CUDA graphs, and inference loop orchestration.

File references below are clickable links, e.g. [tensorrt_llm/_torch/compilation/backend.py](../tensorrt_llm/_torch/compilation/backend.py).

**Compilation vs AutoDeploy**
- `compilation` provides a torch.compile compiler backend for internal graph-level optimization and orchestration:
  - Entry: [tensorrt_llm/_torch/compilation/backend.py](../tensorrt_llm/_torch/compilation/backend.py) class `Backend`.
  - Applies pattern passes to an FX graph (`register_add_norm`, `register_ar_fusions`) then optionally:
    - Multi-stream scheduling, unless piecewise CUDA graph or Inductor is enabled.
    - Piecewise CUDA graph capture via `piecewise_optimizer`.
    - Inductor codegen via `compile_fx(...)` when `enable_inductor=True`.
  - Integrated with AOTAutograd: `aot_module_simplified(..., fw_compiler=self.optimize, ...)` in [tensorrt_llm/_torch/compilation/backend.py](../tensorrt_llm/_torch/compilation/backend.py).
  - Detects input token count from `FakeTensor` metadata to drive piecewise graph capture.

- `auto_deploy` is an “outer” deployment-oriented wrapper with a backend registry that composes torch.compile and CUDA Graphs:
  - Registry and interface: [tensorrt_llm/_torch/auto_deploy/compile/compiler.py](../tensorrt_llm/_torch/auto_deploy/compile/compiler.py).
  - Backends:
    - `torch-simple`: no-op pass-through. [tensorrt_llm/_torch/auto_deploy/compile/backends/torch_simple.py](../tensorrt_llm/_torch/auto_deploy/compile/backends/torch_simple.py).
    - `torch-compile`: wraps the model with `torch.compile(dynamic=True)`. [tensorrt_llm/_torch/auto_deploy/compile/backends/torch_compile.py](../tensorrt_llm/_torch/auto_deploy/compile/backends/torch_compile.py).
    - `torch-cudagraph`: captures and replays CUDA graphs over a set of batch sizes; handles flattening, hashing static args, prefetching output buffers, and replay. [tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py](../tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py).
    - `torch-opt`: combines `torch.compile(dynamic=True)` with CUDA-graph capture. [tensorrt_llm/_torch/auto_deploy/compile/backends/torch_opt.py](../tensorrt_llm/_torch/auto_deploy/compile/backends/torch_opt.py).
  - AutoDeploy warms up with `autotune()` before capture to fix tactics and minimize host noise (see [tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py](../tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py)).

- Key differences:
  - compilation.Backend is a custom “compiler” used as `fw_compiler` within torch.compile, transforming the FX graph with TRT-LLM-specific passes and optionally handing off to Inductor; it returns an optimized graph/module.
  - auto_deploy backends operate at a higher level around the nn.Module, choosing between torch.compile, CUDA graph capture, or both; they do not inject FX-level transforms of TRT-LLM patterns beyond what torch.compile/Inductor already applies.
  - Both use torch.compile, but compilation.Backend owns passes and piecewise CUDA-graph orchestration inside the compiler; auto_deploy focuses on deployment ergonomics and batched CUDA-graph replay.

**Runtime Backends (Python)**
- Attention backends ([tensorrt_llm/_torch/attention_backend](../tensorrt_llm/_torch/attention_backend)):
  - `TRTLLM` backend: Python wrapper calls into `tensorrt_llm.bindings.internal.thop` attention ops and multiple C++ THOP kernels; optimized for TRT-LLM kernels and memory layouts. See [tensorrt_llm/_torch/attention_backend/trtllm.py](../tensorrt_llm/_torch/attention_backend/trtllm.py) → `thop.attention(...)`.
  - `FLASHINFER` backend: integration with FlashInfer kernels (enabled when installed) [tensorrt_llm/_torch/attention_backend/flashinfer.py](../tensorrt_llm/_torch/attention_backend/flashinfer.py).
  - `VANILLA` backend: a reference/PyTorch fall-back [tensorrt_llm/_torch/attention_backend/vanilla.py](../tensorrt_llm/_torch/attention_backend/vanilla.py).
  - torch.compile compatibility: only TRTLLM and FLASHINFER backends are supported for compile-time in-place op variants [tensorrt_llm/_torch/modules/attention.py](../tensorrt_llm/_torch/modules/attention.py).

- Model execution backends (deployment):
  - `auto_deploy` compile backends described above are used when constructing the AutoDeploy LLM wrapper [tensorrt_llm/_torch/auto_deploy/llm.py](../tensorrt_llm/_torch/auto_deploy/llm.py).
  - The executor also uses CUDA-graph runner utilities in [tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py](../tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py) to manage capture/replay in production scheduling.

- Interplay with torch.compile:
  - Attention modules and fused ops expose torch.library custom ops with `.register_fake` shape/type functions so Inductor/Dynamo can trace graphs while treating kernels as extern calls [tensorrt_llm/_torch/custom_ops](../tensorrt_llm/_torch/custom_ops).
  - compilation.Backend uses AOTAutograd to substitute its FX optimizer. auto_deploy may wrap the already-compiled module in a CUDA graph for replay.

**Custom Ops: Python → C++ → CUDA**

This section traces ops in [tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py](../tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py) that wrap TRT-LLM Gen kernels for MoE. Four families are exposed:

- `trtllm::fp4_block_scale_moe_runner` — FP4 hidden activations with block-scaling, returns multiple tensors (e.g., outputs and optionally gate outputs).
- `trtllm::fp8_block_scale_moe_runner` — FP8 activations with block scaling.
- `trtllm::e4m3_mxe2m1_block_scale_moe_runner` — mixed e4m3 activations with mxe2m1 weights.
- `trtllm::fp8_fp4_block_scale_moe_runner` — mixed FP8/FP4 MoE path.

Python registration and dispatch:
- Each custom op is declared with `@torch.library.custom_op("trtllm::...", mutates_args=())`, with accompanying `.register_fake` to enable torch.compile tracing `tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py:150, 292, 520, 1040`.
- Calls flow through a TunableRunner implementation (e.g., `FP4BlockScaleMoERunner`) which:
  - Computes a tile size heuristic (tokens per expert rounded to power-of-two) `calculate_tile_tokens_dim(...)`.
  - Instantiates a Torch custom class bound in C++: `torch.classes.trtllm.FP4BlockScaleMoERunner(...)` (one per tile size) `tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py:57, 107`.
  - Queries valid tactics via `get_valid_configs(...)` which maps to C++ `Runner::getValidConfigIndices(...)` for the given shapes.
  - Runs via `.run_moe(...)` with the chosen tactic.
- The Python AutoTuner (`autotuner.py`) is used to select the best tactic per shape/profile; see below.

C++ registration and kernel launch:
- The corresponding C++ custom classes are registered via `TORCH_LIBRARY_FRAGMENT(trtllm, m)` with `.class_<...>("...Runner").def("run_moe", &...)`:
  - FP4: [cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp](../cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp) registers `FP4BlockScaleMoERunner` and `FP8FP4BlockScaleMoERunner`.
  - FP8-only: [cpp/tensorrt_llm/thop/fp8BlockScaleMoe.cpp](../cpp/tensorrt_llm/thop/fp8BlockScaleMoe.cpp) registers `FP8BlockScaleMoERunner`.
  - Mixed MX* variants: [cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp](../cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp) registers `MxE4m3MxE2m1BlockScaleMoERunner` and `Bf16MxE2m1...`.
- The `.run_moe(...)` implementations:
  - Validate/normalize inputs (routing logits vs precomputed topk, groups, shapes) `cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp:32`.
  - Allocate intermediate workspaces (expert indices/weights, GEMM outputs/scales) on CUDA.
  - Invoke TRT-LLM Gen Runners (e.g., `tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner`) with current CUDA stream `at::cuda::getCurrentCUDAStream(...)`.
  - Return output tensors; tactic is the “config index” that selects a concrete kernel config.

Putting it together — MoE runner end-to-end path with links:
- Python calls op, e.g. [tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py](../tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py)
  - Creates a Python `TunableRunner` and invokes [autotuner.py: AutoTuner.choose_one](../tensorrt_llm/_torch/autotuner.py) to pick tactic.
  - Calls into C++ Torch class: `torch.classes.trtllm.FP8BlockScaleMoERunner(...).run_moe(...)`.
- C++ THOP wrapper (Torch custom class):
  - FP8: [cpp/tensorrt_llm/thop/fp8BlockScaleMoe.cpp](../cpp/tensorrt_llm/thop/fp8BlockScaleMoe.cpp) `.def("run_moe", ...)`
  - FP4/Mixed: [cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp](../cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp), [cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp](../cpp/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp)
  - Forwards to CUDA MoE runner: `tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::MoE::Runner::run(...)`.
- CUDA MoE orchestration: [cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h](../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h), [runner.cu](../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu)
  - Routing: [RoutingDeepSeek.cu](../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/RoutingDeepSeek.cu), [RoutingRenormalize.cu](../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/RoutingRenormalize.cu), [RoutingLlama4.cu](../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/RoutingLlama4.cu), types in [RoutingKernel.h](../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/RoutingKernel.h)
  - Permute + GEMM1: [blockScaleMoe/runner.h (PermuteGemm1::Runner)](../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h) → [batchedGemm/KernelRunner.h](../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.h) / [KernelRunner.cpp](../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/KernelRunner.cpp)
    - Final GEMM kernel launch via [BatchedGemmInterface.h](../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/BatchedGemmInterface.h) and [KernelMetaInfo.h](../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/KernelMetaInfo.h) using [CudaKernelLauncher.h](../cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/trtllm/gen/CudaKernelLauncher.h)
  - Activation (DeepSeek only): [blockScaleMoe/DevKernel.cu](../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu)
  - GEMM2: [blockScaleMoe/runner.h (Gemm2::Runner)](../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h) → same GEMM path
  - Finalize (unpermute + combine): [blockScaleMoe/DevKernel.cu](../cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu)

How ops are registered with Torch:
- Python level: `@torch.library.custom_op` creates an aten op name in the `trtllm` namespace and provides a Python implementation and a `.register_fake` for Dynamo/Inductor.
- C++ level: `TORCH_LIBRARY_FRAGMENT(trtllm, m)` registers functions (`m.def(...)`) and custom classes (`m.class_<...>(...)`); device-specific impls are bound with `TORCH_LIBRARY_IMPL(trtllm, CUDA, m)` in other files.

UserBuffers custom ops
- Python ops: `tensorrt_llm/_torch/custom_ops/userbuffers_custom_ops.py:1` defines:
  - `trtllm::copy_to_userbuffers(a)` — allocates a UserBuffers-backed CUDA tensor and copies `a` into it.
  - `trtllm::add_to_ub(a, b)` — computes `a + b` into a UserBuffers-backed output via `out=`.
  - `trtllm::matmul_to_ub(a, b)` — computes `a @ b` into a UserBuffers-backed output via `out=`.
- C++ binding used by these Python ops:
  - `trtllm::create_userbuffers_tensor(shape, dtype)` is registered in `cpp/tensorrt_llm/thop/userbuffersTensor.cpp:42`.
  - It calls `UserBuffersManager::allocate_userbuffers(...)` and wraps the memory with `torch::from_blob(...)` plus a custom deleter returning memory to the UB manager `cpp/tensorrt_llm/thop/userbuffersTensor.cpp:12`.
- Dispatch:
  - `copy_to_userbuffers` → `torch.ops.trtllm.create_userbuffers_tensor(...)` to allocate UB-backed output, then `.copy_(...)` GPU-to-GPU.
  - `add_to_ub` / `matmul_to_ub` allocate UB-backed `out` then call stock PyTorch `torch.add`/`torch.matmul` into `out`, ensuring the result resides in UB-managed memory without additional copies.

Autotuner internals
- Python API: `autotune(...)` context manager toggles tuning on and persists/loads a JSON cache per rank `tensorrt_llm/_torch/autotuner.py:221`.
- Tuning loop: `AutoTuner.choose_one(...)` builds optimization profiles over dynamic dims (grid over buckets), queries each runner for `get_valid_tactics(...)`, then profiles each tactic to find the fastest `tensorrt_llm/_torch/autotuner.py:542`.
- Profiling:
  - Warmup N times, then measure average time over `repeat` iterations using CUDA events `tensorrt_llm/_torch/autotuner.py:704`.
  - Inserts a tiny synthetic kernel before timing via `delay_kernel(...)` to decouple host overhead `tensorrt_llm/_torch/autotuner.py:750`.
- Heuristics and constraints:
  - Dynamic bucket generators include powers-of-two token counts `tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py:29` and helpers in `.../utils.py`.
  - ConstraintSpecs map dependent dimensions (e.g., routing/logit length matches num tokens) `tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py:200, 520`.
- Caching:
  - Cache key encodes `(custom_op, runner-id, rounded input-shapes/profile)`; entries store `(runner_index, tactic, min_time)` and survive process restarts via JSON `tensorrt_llm/_torch/autotuner.py:333, 618`.
- Fallback:
  - If no cache entry and tuning disabled, a “fallback” tactic `-1` and default runner are used.

**Minimal Snippets**

- Torch compile (internal compiler) vs AutoDeploy backends

  Example uses a toy module and compiles two ways. Requires CUDA for meaningful speedups.

  1) Internal compiler backend with custom passes:

  ```python
  import torch, torch.nn as nn
  from tensorrt_llm._torch.compilation import Backend

  class Toy(nn.Module):
      def __init__(self, d=1024):
          super().__init__()
          self.lin1, self.lin2 = nn.Linear(d, d), nn.Linear(d, d)
      def forward(self, x):
          return self.lin2(torch.nn.functional.relu(self.lin1(x)))

  model = Toy().cuda().eval()
  backend = Backend(enable_inductor=True, max_num_streams=1)
  compiled = torch.compile(model, backend=backend)  # FX passes + Inductor
  x = torch.randn(32, 1024, device='cuda')
  y = compiled(x)
  ```

  2) AutoDeploy `torch-opt` (torch.compile + CUDA graphs):

  ```python
  import torch, torch.nn as nn
  from tensorrt_llm._torch.auto_deploy.compile.backends.torch_opt import TorchOptCompiler

  model = Toy().cuda().eval()
  # Choose capture batch sizes; can omit to let backend choose heuristics
  compiler = TorchOptCompiler(model, args=(), kwargs={}, max_batch_size=128, cuda_graph_batch_sizes=[1, 32, 128])
  captured = compiler.compile()
  # First call triggers capture; later calls replay when shapes match
  y = captured(torch.randn(32, 1024, device='cuda'))
  ```

- End-to-end TRTLLM Gen custom op

  ```python
  import torch
  from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import fp8_block_scale_moe_runner
  from tensorrt_llm._torch.autotuner import autotune

  # Shapes: [tokens, hidden], experts, inter size
  T, H, E, I = 256, 4096, 64, 1536
  hidden_states = torch.randn(T, H, device='cuda', dtype=torch.bfloat16)
  h_scale      = torch.ones(T, device='cuda', dtype=torch.float32)  # per-row (example)
  w1 = torch.randn(I, H, device='cuda', dtype=torch.float8_e4m3fn)
  s1 = torch.ones(I, device='cuda', dtype=torch.float32)
  w2 = torch.randn(H, I, device='cuda', dtype=torch.float8_e4m3fn)
  s2 = torch.ones(H, device='cuda', dtype=torch.float32)
  logits = torch.randn(T, E, device='cuda', dtype=torch.bfloat16)

  with autotune(cache_path='codex-tensorrt-docs/autotune_cache.json'):
      out = fp8_block_scale_moe_runner(
          logits, None, hidden_states, h_scale, w1, s1, w2, s2,
          num_experts=E, top_k=4, n_group=None, topk_group=None,
          intermediate_size=I, local_expert_offset=0, local_num_experts=E,
          routed_scaling_factor=None, routing_method_type=0,
      )
  print(out.shape)
  ```

- UserBuffers ops

  ```python
  import torch
  from tensorrt_llm._torch.custom_ops import userbuffers_custom_ops as ub

  a = torch.randn(32, 1024, device='cuda', dtype=torch.bfloat16)
  b = torch.randn(32, 1024, device='cuda', dtype=torch.bfloat16)

  # Allocate UB-backed tensor and copy to it
  a_ub = ub.copy_to_userbuffers(a)

  # Compute into UB-backed outputs
  c = ub.add_to_ub(a_ub, b)
  w = torch.randn(1024, 4096, device='cuda', dtype=torch.bfloat16)
  z = ub.matmul_to_ub(c, w)  # result in UB memory
  ```

- Autotuner, standalone sketch

  ```python
  from tensorrt_llm._torch.autotuner import AutoTuner, autotune

  tuner = AutoTuner.get()
  # Clear cache if needed
  tuner.clear_cache(); tuner.reset_statistics()
  # Wrap any op calls inside autotune() to populate cache
  with autotune(cache_path='codex-tensorrt-docs/autotune_cache.json'):
      # call ops which internally use AutoTuner.choose_one(...)
      ...
  print(tuner.stats)
  ```

**Notes and Limitations**
- Only a subset of custom backends/ops are torch.compile-compatible; attention explicitly restricts compile-time backends to TRTLLM and FLASHINFER ([tensorrt_llm/_torch/modules/attention.py](../tensorrt_llm/_torch/modules/attention.py)).
- Unknown/custom ops appear as extern calls to Inductor; fake registrations ensure tracing and shape propagation, but fusion is limited across these boundaries.
- CUDA Graph capture imposes shape and static-input hashing constraints; see `CapturedGraph` in [tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py](../tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py) for details on rounding and replay.

**Where To Look Next**
- Attention kernels and THOP registrations: [cpp/tensorrt_llm/thop/attentionOp.cpp](../cpp/tensorrt_llm/thop/attentionOp.cpp).
- Additional custom ops and runners: [tensorrt_llm/_torch/custom_ops/torch_custom_ops.py](../tensorrt_llm/_torch/custom_ops/torch_custom_ops.py) maps to multiple [cpp/tensorrt_llm/thop](../cpp/tensorrt_llm/thop) files.
- UserBuffers manager and memory semantics: [cpp/tensorrt_llm/kernels/userbuffers/userbuffersManager.h](../cpp/tensorrt_llm/kernels/userbuffers/userbuffersManager.h) (also see [cpp/tensorrt_llm/thop/userbuffersTensor.h](../cpp/tensorrt_llm/thop/userbuffersTensor.h)).
