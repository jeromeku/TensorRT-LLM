TensorRT‑LLM Autotuning Trace: Python → Bindings → C++/CUDA
===========================================================

This document traces, in detail, how autotuning selects and applies GEMM tactics for the fused MoE path. It shows the full dispatch from Python custom ops to C++ runners and CUTLASS/CUDA kernels, with annotated code excerpts and clickable source spans.

Table of Contents
-----------------

1. [Overview](#overview)
2. [Python Entrypoint (fused_moe)](#python-entrypoint-fused_moe)
3. [MoERunner (tunable runner wrapper)](#moerunner-tunable-runner-wrapper)
4. [AutoTuner.choose_one (profiling loop)](#autotunerchoose_one-profiling-loop)
5. [C++ FusedMoeRunner profiling API](#c-fusedmoerunner-profiling-api)
6. [GemmProfilerBackend: init/prepare/runProfiler](#gemmprofilerbackend-initpreparerunprofiler)
7. [CUTLASS MoE: grouped GEMM dispatch](#cutlass-moe-grouped-gemm-dispatch)
8. [Tactics → Execution (final run)](#tactics--execution-final-run)
9. [Dataflow Diagrams](#dataflow-diagrams)


## Overview

Goal: choose fast GEMM configs (GEMM1 & GEMM2) given the current shapes/dtypes/quantization, then run fused MoE with those tactics.

```
fused_moe (Python)
  ├─ build MoERunner (torch.classes.trtllm.FusedMoeRunner)
  ├─ AutoTuner.choose_one("...gemm1") → profile tactics via MoERunner.forward
  │    └─ C++ FusedMoeRunner.runGemmProfile → GemmProfilerBackend.init/prepare/runProfiler
  ├─ AutoTuner.choose_one("...gemm2") → same
  └─ fused_moe_runner.run_moe(..., [best1, best2]) → actual execution
```


## Python Entrypoint (fused_moe)

The custom op `trtllm::fused_moe` drives autotuning for GEMM1 and GEMM2, then launches the fused run using the selected tactics.

- File: [tensorrt_llm/_torch/custom_ops/torch_custom_ops.py:124-173](../tensorrt_llm/_torch/custom_ops/torch_custom_ops.py#L124-L173), [tensorrt_llm/_torch/custom_ops/torch_custom_ops.py:196-245](../tensorrt_llm/_torch/custom_ops/torch_custom_ops.py#L196-L245)

```py
@torch.library.custom_op("trtllm::fused_moe", mutates_args=())
def fused_moe(...):
    tuner = AutoTuner.get()                             # acquire singleton autotuner
    # Choose GEMM1 tactic
    _, gemm_tactic_1 = tuner.choose_one(
        "trtllm::fused_moe::gemm1", [moe_runner], MoERunner.tuning_config,
        [tuner_input, fc1_expert_weights, fc1_expert_biases, fc2_expert_weights, fc2_expert_biases], gemm_idx=1)
    # Choose GEMM2 tactic
    _, gemm_tactic_2 = tuner.choose_one(
        "trtllm::fused_moe::gemm2", [moe_runner], MoERunner.tuning_config,
        [tuner_input, fc1_expert_weights, fc1_expert_biases, fc2_expert_weights, fc2_expert_biases], gemm_idx=2)
    # Execute with best tactics
    output = moe_runner.fused_moe_runner.run_moe(
        input, token_selected_experts, token_final_scales, fc1_expert_weights, fc1_expert_biases,
        fc2_expert_weights, fc2_expert_biases, quant_scales, input_sf, swizzled_input_sf,
        swiglu_alpha, swiglu_beta, swiglu_limit, tp_size, tp_rank, ep_size, ep_rank,
        cluster_size, cluster_rank, enable_alltoall, min_latency_mode,
        [gemm_tactic_1, gemm_tactic_2], unpadded_hidden_size)
    return output if min_latency_mode else [output]
```

Notes
- `tuner_input` is chosen to match non‑alltoall shapes for warmup consistency.
- `MoERunner.tuning_config` defines the dynamic M (num_tokens) sampling for profiling.


## MoERunner (tunable runner wrapper)

`MoERunner` wraps a `torch.classes.trtllm.FusedMoeRunner` instance and exposes the `TunableRunner` interface to AutoTuner.

- File: [tensorrt_llm/_torch/custom_ops/torch_custom_ops.py:27-35](../tensorrt_llm/_torch/custom_ops/torch_custom_ops.py#L27-L35), [tensorrt_llm/_torch/custom_ops/torch_custom_ops.py:37-92](../tensorrt_llm/_torch/custom_ops/torch_custom_ops.py#L37-L92), [tensorrt_llm/_torch/custom_ops/torch_custom_ops.py:94-121](../tensorrt_llm/_torch/custom_ops/torch_custom_ops.py#L94-L121)

```py
class MoERunner(TunableRunner):
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(DynamicTensorSpec(0, 0, get_last_power_of_2_num_tokens_buckets(8192),
                                lambda x: min(last_positive_power_of_2(x), 8192)), ),
        tune_max_num_tokens=8192,
    )
    def __init__(..., top_k, tp_size, tp_rank, ep_size, ep_rank, ...):
        # cache / construct torch.classes.trtllm.FusedMoeRunner
        self.fused_moe_runner = torch.classes.trtllm.FusedMoeRunner(...)
    def get_valid_tactics(self, inputs, profile, **kwargs):
        return range(self.fused_moe_runner.get_tactic_num(kwargs["gemm_idx"]))
    def forward(self, inputs, gemm_idx=0, tactic=-1, do_preparation=False):
        # Delegate a single profiling run to C++
        self.fused_moe_runner.run_gemm_profile(x, fc1_w, fc1_b, fc2_w, fc2_b, self.top_k, ...,
                                               self.enable_alltoall, self.min_latency_mode,
                                               gemm_idx, tactic, do_preparation, self.unpadded_hidden_size)
```

Notes
- `get_valid_tactics` queries the C++ runner for candidate tactic count per GEMM.
- `forward` calls `run_gemm_profile` to time a single tactic.


## AutoTuner.choose_one (profiling loop)

The autotuner tries candidate tactics and picks the best, caching results.

- File: [tensorrt_llm/_torch/autotuner.py:511-620](../tensorrt_llm/_torch/autotuner.py#L511-L620), [tensorrt_llm/_torch/autotuner.py:651-707](../tensorrt_llm/_torch/autotuner.py#L651-L707), [tensorrt_llm/_torch/autotuner.py:719-720](../tensorrt_llm/_torch/autotuner.py#L719-L720)

```py
_, runner_id, tactic, _ = self.profiling_cache.search_cache(custom_op, runners, input_shapes, tuning_config)
if self.is_tuning_mode:                                 # Warmup mode → profile candidates
    profiles = self._optimization_profiles(tuning_config, inputs)
    # For each profile → profile runners/tactics
    best_runner_id, best_tactic, min_time, failures = self._profile_runners(
        custom_op, runners, inputs, profile, tuning_config, **kwargs)
    # Cache the winner and continue
# Else (inference): use cached or fallback tactic
```

Inner profiling loop (critical path): tensorrt_llm/_torch/autotuner.py#L651-L707

```py
for runner_id, runner in enumerate(runners):
    valid_tactics = runner.get_valid_tactics(input_tensors, profile, **kwargs)
    if "do_preparation" in runner_arg_names and len(valid_tactics) > 0:
        runner(input_tensors, tactic=-1, do_preparation=True, **kwargs)  # pre‑alloc workspace/templates
    for tac in valid_tactics:
        try:
            time_measured = self._profile_single_kernel(runner, input_tensors, tac, **kwargs)
        except Exception:                                    # record failure but keep going
            time_measured = float('inf')
        if time_measured < min_time:
            min_time, best_runner_id, best_tactic = time_measured, runner_id, tac
```

Timing details (CUDA events and warmup): tensorrt_llm/_torch/autotuner.py#L719-L720 and earlier

```py
# _profile_single_kernel: warmup runs, then CUDA event timing across repeats
for _ in range(self.warmup): runner(inputs, tactic=tactic, **kwargs)
start.record();  for _ in range(self.repeat): runner(inputs, tactic=tactic, **kwargs);  end.record();
avg_time = start.elapsed_time(end) / self.repeat
```


## C++ FusedMoeRunner profiling API

The PyTorch custom class exposes `get_tactic_num` and `run_gemm_profile` for Python autotuner usage.

- File: [cpp/tensorrt_llm/thop/moeOp.cpp:1151-1156](../cpp/tensorrt_llm/thop/moeOp.cpp#L1151-L1156) (registration), [cpp/tensorrt_llm/thop/moeOp.cpp:614-620](../cpp/tensorrt_llm/thop/moeOp.cpp#L614-L620) (getTacticNum), [cpp/tensorrt_llm/thop/moeOp.cpp:622-711](../cpp/tensorrt_llm/thop/moeOp.cpp#L622-L711) (runGemmProfile)

Key steps inside `runGemmProfile`:

```cpp
// 1) Compute shapes; pick expert counts; derive group size for W4/W8, etc.
// 2) mProfiler->init(...): set SM, activation/weight/output dtypes, inter/hidden sizes, ep/tp/cluster config.
// 3) Allocate profile workspace (cudaMalloc) → mProfiler->prepare(...)
// 4) mProfiler->runProfiler(num_rows, profile, mProfileWorkspace, expert_weights_ptr, stream);
```

 Snippets: `cpp/tensorrt_llm/thop/moeOp.cpp#L600` and `cpp/tensorrt_llm/thop/moeOp.cpp#L709`

```cpp
mKernelRunner->runMoe(input.const_data_ptr(), ..., output.data_ptr(), ..., stream);  // used in min_latency path
...
// Profile specific tactic. Assuming at least one preparation phase executed already.
mProfiler->runProfiler(num_rows, profile, mProfileWorkspace, expert_weights_ptr, stream);
```

Notes
- The FusedMoeRunner ctor also pre‑populates candidate profiles via `mGemm1Profiles = mKernelRunner->getTactics(...)`.


## GemmProfilerBackend: init/prepare/runProfiler

This helper computes workspace needs, builds TMA WS inputs, and iterates candidate configs.

- File: [cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h:864-1140](../cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h#L864-L1140) (GemmProfilerBackend)
- Hot‑path: [cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4466-4568](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L4466-L4568)

Highlights (annotated):

```cpp
// init(...): record SM, num experts, inter/hidden sizes, scaling types (MXFPX/NVFP4), parallelism config
// prepare(...): compute per‑sample routing buffers, TMA WS inputs, quant params, and allocate per‑config workspaces
// runProfiler(...):
//   - Configure TMA WS inputs per sample and fusion mode (FINALIZE on Hopper if allowed)
//   - Select config (tactic) and call mInterface->gemm1 / gemm2 with that config
//   - Measure execution via CUDA timing from Python side
```

 Example (configuring TMA WS inputs, then calling gemm1 + gemm2): `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L3626`

```cpp
Self::gemm1(moe_gemm_runner_, blockscale_gemm_runner, input_activations, fc1_result_, glu_inter_result_,
            expert_first_token_offset_, gemm1_tma_ws_input, fc1_expert_weights, fc1_expert_biases,
            num_valid_tokens_ptr, fc1_int_scales, fc1_fp8_dequant,
            quant_params, num_rows, expanded_num_rows, hidden_size, inter_size, num_experts_per_node, ...);
auto gemm2_input = applyPrequantScale(...);
Self::gemm2(moe_gemm_runner_, blockscale_gemm_runner, gemm2_input, final_output, nullptr,
            expert_first_token_offset_, gemm2_tma_ws_input, fc2_expert_weights, fc2_expert_biases, fc2_int_scales,
            fc2_fp8_dequant, ..., quant_params, token_topk_unpermuted_scales, ...);
```


## CUTLASS MoE: grouped GEMM dispatch

Grouped GEMM entry points select the architecture‑specific kernels and epilogues based on `EpilogueTag` and activation type.

- File: [cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:939-966](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h#L939-L966)

```cpp
void MoeGemmRunner<...>::runGemm(GroupedGemmInput inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs) {
    dispatchToArch<EpilogueTag>(inputs, hopper_inputs);
}
void MoeGemmRunner<...>::moeGemmBiasAct(GroupedGemmInput inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs) {
    switch (inputs.activation_type) {
        case ActivationType::Swiglu: runGemm<cutlass_extensions::EpilogueOpDefaultSilu>(inputs, hopper_inputs); break;
        // ...
    }
}
```


## Tactics → Execution (final run)

After both GEMM tactics are selected, `fused_moe` calls `FusedMoeRunner.run_moe` with `[gemm_tactic_1, gemm_tactic_2]` to execute the high‑level pipeline.

- File: [tensorrt_llm/_torch/custom_ops/torch_custom_ops.py:218-245](../tensorrt_llm/_torch/custom_ops/torch_custom_ops.py#L218-L245)

```py
run_moe = moe_runner.fused_moe_runner.run_moe_min_latency if min_latency_mode else moe_runner.fused_moe_runner.run_moe
output = run_moe(input, token_selected_experts, token_final_scales, fc1_expert_weights, fc1_expert_biases,
                 fc2_expert_weights, fc2_expert_biases, quant_scales, input_sf, swizzled_input_sf, swiglu_alpha,
                 swiglu_beta, swiglu_limit, tp_size, tp_rank, ep_size, ep_rank, cluster_size, cluster_rank,
                 enable_alltoall, min_latency_mode, [gemm_tactic_1, gemm_tactic_2], unpadded_hidden_size)
```

The C++ FusedMoeRunner (non‑profiling run) performs routing maps, expands rows, dispatches GEMM1/2, and handles finalize fusion or standalone finalize.


## Dataflow Diagrams

Autotuning control-flow (high‑level)

```
fused_moe
  ├─ build MoERunner
  ├─ choose_one("gemm1")
  │   ├─ MoERunner.get_valid_tactics → FusedMoeRunner.get_tactic_num
  │   ├─ (prep) MoERunner.forward(..., do_preparation=True) → run_gemm_profile(..., tactic=-1)
  │   └─ repeat for each tactic: MoERunner.forward(..., tactic=k) → run_gemm_profile(..., tactic=k)
  ├─ choose_one("gemm2") [same]
  └─ run_moe(..., [best1, best2])
```

Profiling inner‑loop (detail)

```
AutoTuner._profile_single_kernel
  ├─ warmup:   for i in 1..warmup:   runner.forward(tactic)
  ├─ timing:   record start; for i in 1..repeat: runner.forward(tactic); record end
  └─ avg_time = (end - start)/repeat
```

Workspace/tactical preparation (C++)

```
FusedMoeRunner.runGemmProfile
  ├─ GemmProfilerBackend.init
  ├─ allocate profile workspace (cudaMalloc)
  ├─ GemmProfilerBackend.prepare → build TMA WS, quant params, routing samples
  └─ GemmProfilerBackend.runProfiler → for sample/config:
       ├─ configure TMA WS (fusion, swap AB)
       ├─ mInterface->gemm1(inputs, config)
       └─ mInterface->gemm2(inputs, config)
```

This completes the end‑to‑end autotuning trace.
