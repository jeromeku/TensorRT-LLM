TensorRT‑LLM Fused MoE: Cutlass & CuteDSL Deep Dives
====================================================

This is a literate, end‑to‑end walkthrough of the fused MoE call stack for the Cutlass and CuteDSL paths. It mixes explanatory prose with annotated code excerpts and VSCode‑style source links for fast navigation.

Table of Contents
-----------------

1. [Architecture Overview](#architecture-overview)
2. [Cutlass Path Deep Dive](#cutlass-path-deep-dive)
   - [Python Entry & Routing](#python-entry--routing)
   - [Optional All‑to‑All Dispatch](#optional-alltoall-dispatch)
   - [Fused MoE Compute (Autotuned GEMM1/2)](#fused-moe-compute-autotuned-gemm12)
   - [Finalize & Combine](#finalize--combine)
   - [Autotuning Hooks](#autotuning-hooks)
3. [CuteDSL Path Deep Dive](#cutedsl-path-deep-dive)
   - [Python Entry & Routing](#python-entry--routing-1)
   - [Permute / Expand (Token Dispatch in‑kernel)](#permute--expand-token-dispatch-inkernel)
   - [GEMMs & Activation](#gemms--activation)
   - [Finalize](#finalize)
4. [Dataflow Diagrams](#dataflow-diagrams)
5. [Key Files Index](#key-files-index)

Trace Cards (Selected Call Sites)
---------------------------------

- Backend selection (create_moe): [tensorrt_llm/_torch/modules/fused_moe/create_moe.py:22-57](../tensorrt_llm/_torch/modules/fused_moe/create_moe.py#L22-L57)
  ```py
  def get_moe_cls(model_config: ModelConfig, override_quant_config: Optional[QuantConfig] = None) -> Type[MoE]:
      moe_backend = model_config.moe_backend
      # ... selects CutlassFusedMoE / CuteDslFusedMoE / ... based on config
  ```

- Interface forward wrapper (dispatch to custom op or forward_impl): [tensorrt_llm/_torch/modules/fused_moe/interface.py:236-260](../tensorrt_llm/_torch/modules/fused_moe/interface.py#L236-L260)
  ```py
  def forward(self, x, router_logits, do_finalize=True, output_dtype=None, all_rank_num_tokens=None, use_dp_padding=None):
      if self.register_to_config and is_torch_compiling():
          # routes through moe_custom_op(...) for torch.compile
          res = moe_custom_op(self.layer_idx_str, hidden_states, x_sf, is_swizzled, router_logits, do_finalize, ...)
          return res[0] if do_finalize else res
      else:
          return self.forward_impl(x, router_logits, do_finalize=do_finalize, output_dtype=output_dtype, ...)
  ```

- Cutlass: all‑to‑all fused send + memset ids + fused_moe call site: [tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:366-401](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py#L366-L401)
  ```py
  # Dispatch activations, scales and indices in a single fused op
  x, x_sf, token_selected_experts, token_final_scales = MnnvlMoe.mnnvl_moe_alltoallv([...], alltoall_info, ...)
  torch.ops.trtllm.memset_expert_ids(token_selected_experts, alltoall_info.recv_rank_count_cumsum, max_num_token, top_k, self.num_experts, self.ep_size)
  # Fused MoE op that drives GEMM1→SwiGLU→GEMM2 (+ optional fused finalize)
  final_hidden_states = torch.ops.trtllm.fused_moe(x, token_selected_experts, token_final_scales, self.w3_w1_weight.view(weight_dtype), ...)
  ```

- Fused op (autotune then run): [tensorrt_llm/_torch/custom_ops/torch_custom_ops.py:196-245](../tensorrt_llm/_torch/custom_ops/torch_custom_ops.py#L196-L245)
  ```py
  _, gemm_tactic_1 = tuner.choose_one("trtllm::fused_moe::gemm1", [moe_runner], MoERunner.tuning_config, [...], gemm_idx=1)
  _, gemm_tactic_2 = tuner.choose_one("trtllm::fused_moe::gemm2", [moe_runner], MoERunner.tuning_config, [...], gemm_idx=2)
  run_moe = moe_runner.fused_moe_runner.run_moe_min_latency if min_latency_mode else moe_runner.fused_moe_runner.run_moe
  output = run_moe(input, token_selected_experts, token_final_scales, fc1_expert_weights, ..., [gemm_tactic_1, gemm_tactic_2], unpadded_hidden_size)
  ```

- C++ FusedMoeRunner: forward to CUTLASS runner: [cpp/tensorrt_llm/thop/moeOp.cpp:596-607](../cpp/tensorrt_llm/thop/moeOp.cpp#L596-L607)
  ```cpp
  mKernelRunner->runMoe(input.const_data_ptr(), input_sf_ptr, swizzled_input_sf,
      reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()), token_final_scales_ptr,
      fc1_expert_weights.const_data_ptr(), fc1_bias_ptr, activation_params, fc2_expert_weights.const_data_ptr(),
      fc2_bias_ptr, quant_params, num_rows, hidden_size, inter_size, num_experts_total, experts_per_token,
      static_cast<char*>(workspace_info.workspace.data_ptr()), output.data_ptr(),
      static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, /*enable_alltoall*/ false, lora_params,
      mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, stream);
  ```

- CUTLASS MoE: TMA WS + GEMM1 + GEMM2 (+ finalize): [cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3626-3643](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L3626-L3643)
  ```cpp
  Self::gemm1(moe_gemm_runner_, blockscale_gemm_runner, input_activations, fc1_result_, glu_inter_result_, expert_first_token_offset_, gemm1_tma_ws_input, fc1_expert_weights, fc1_expert_biases, ...);
  auto gemm2_input = applyPrequantScale(...);
  Self::gemm2(moe_gemm_runner_, blockscale_gemm_runner, gemm2_input, final_output, nullptr, expert_first_token_offset_, gemm2_tma_ws_input, fc2_expert_weights, fc2_expert_biases, ...);
  ```

- Grouped GEMM entrypoints: [cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h:939-966](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h#L939-L966)
  ```cpp
  void MoeGemmRunner<...>::runGemm(GroupedGemmInput inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs) {
      dispatchToArch<EpilogueTag>(inputs, hopper_inputs);
  }
  void MoeGemmRunner<...>::moeGemmBiasAct(GroupedGemmInput inputs, TmaWarpSpecializedGroupedGemmInput hopper_inputs) {
      switch (inputs.activation_type) { case ActivationType::Swiglu: runGemm<cutlass_extensions::EpilogueOpDefaultSilu>(inputs, hopper_inputs); ... }
  }
  ```

- Routing CUDA bindings & registration: [cpp/tensorrt_llm/thop/customMoeRoutingOp.cpp:124-146](../cpp/tensorrt_llm/thop/customMoeRoutingOp.cpp#L124-L146)
  ```cpp
  TORCH_LIBRARY_FRAGMENT(trtllm, m) {
      m.def("renorm_moe_routing_op(Tensor router_logits, SymInt topk, ScalarType? output_dtype=None) -> (Tensor, Tensor)");
  }
  TORCH_LIBRARY_IMPL(trtllm, CUDA, m) {
      m.impl("renorm_moe_routing_op", &torch_ext::renorm_moe_routing_op);
  }
  TORCH_LIBRARY_FRAGMENT(trtllm, m) {
      m.def("default_moe_routing_op(Tensor router_logits, SymInt topk, ScalarType? output_dtype=None) -> (Tensor, Tensor)");
  }
  TORCH_LIBRARY_IMPL(trtllm, CUDA, m) {
      m.impl("default_moe_routing_op", &torch_ext::default_moe_routing_op);
  }
  ```

- Routing CUDA kernel launcher (specializes kernel by expert/top‑k sizes): [cpp/tensorrt_llm/kernels/customMoeRoutingKernels.cu:212-231](../cpp/tensorrt_llm/kernels/customMoeRoutingKernels.cu#L212-L231)
  ```cpp
  template <typename InputT, typename OutputT, typename IdxT, bool DoSoftmaxBeforeTopK>
  void invokeCustomMoeRouting(InputT* routerLogits, OutputT* topkValues, IdxT* topkIndices, int64_t numTokens,
                              int64_t numExperts, int64_t topK, cudaStream_t stream) {
      const uint32_t numBlocks = std::min(...);
      uint32_t maxNumExperts = nextPowerOfTwo(numExperts) < 32 ? 32 : nextPowerOfTwo(numExperts);
      auto* kernelInstance = &customMoeRoutingKernel<InputT, OutputT, IdxT, 128, 8, DoSoftmaxBeforeTopK>;
      switch (maxNumExperts) { CASE(32) CASE(64) CASE(96) CASE(128) default: kernelInstance = nullptr; }
      TLLM_CHECK_WITH_INFO(kernelInstance != nullptr, "Can not find corresponding kernel instance.");
      // launch ...
  }
  ```

- All‑to‑all prepare bindings: [cpp/tensorrt_llm/thop/moeCommOp.cpp:295-305](../cpp/tensorrt_llm/thop/moeCommOp.cpp#L295-L305)
  ```cpp
  TORCH_LIBRARY_FRAGMENT(trtllm, m) {
      m.def("mnnvl_moe_alltoallv_prepare_without_allgather(Tensor experts_ids, Tensor? experts_statics, Tensor allWorkspace, int max_token_count_per_rank, int ep_rank, int ep_size, int expert_count, int slot_count, int top_k) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor?)");
  }
  TORCH_LIBRARY_IMPL(trtllm, CUDA, m) {
      m.impl("mnnvl_moe_alltoallv_prepare_without_allgather", &torch_ext::moePrepareOp);
  }
  ```

- Python helpers for MNNVL (prepare/dispatch/combine): [tensorrt_llm/_mnnvl_utils.py:401-446](../tensorrt_llm/_mnnvl_utils.py#L401-L446), [tensorrt_llm/_mnnvl_utils.py:462-520](../tensorrt_llm/_mnnvl_utils.py#L462-L520)
  ```py
  (local_send_rank_count_cumsum, local_send_rank_indices, local_recv_rank_count_cumsum, local_recv_rank_indices, backward_local_recv_rank_indices, gathered_expert_statics) = torch.ops.trtllm.mnnvl_moe_alltoallv_prepare_without_allgather(...)
  alltoall_info = MoEAlltoallInfo(...)
  return alltoall_info, gathered_expert_statics
  ...
  (local_gather_indices, send_rank_count_cumsum, send_rank_local_indices, recv_rank_count_cumsum, recv_rank_local_indices, backward_recv_rank_local_indices) = torch.ops.trtllm.moe_comm_prepare_indices(...)
  torch.ops.trtllm.moe_local_gather(...)
  ```

- Autotuner inner loop (profiling tactics and caching best): [tensorrt_llm/_torch/autotuner.py:651-707](../tensorrt_llm/_torch/autotuner.py#L651-L707)
  ```py
  for runner_id, runner in enumerate(runners):
      valid_tactics = runner.get_valid_tactics(input_tensors, profile, **kwargs)
      if "do_preparation" in runner_arg_names and len(valid_tactics) > 0:
          runner(input_tensors, tactic=-1, do_preparation=True, **kwargs)
      for tac in valid_tactics:
          try:
              time_measured = self._profile_single_kernel(runner, input_tensors, tac, **kwargs)
          except Exception:
              # record failure; continue
          if time_measured < min_time:
              min_time = time_measured; best_runner_id, best_tactic = runner_id, tac
  ```

- Expand token rows (duplicate by top‑k): [cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:1585-1592](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L1585-L1592)
  ```cpp
  void expandInputRowsKernelLauncher(InputActivationsType const* unpermuted_input, ExpandedActivationsType* permuted_output,
      float const* unpermuted_scales, float* permuted_scales, int const* permuted_row_to_unpermuted_row,
      int64_t const num_rows, int64_t const hidden_size, int const k, int const num_experts_per_node, ...);
  ```

- CuteDSL: moe_permute_op binding & runPermute template: [cpp/tensorrt_llm/thop/moeUtilOp.cpp:38-88](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L38-L88)
  ```cpp
  // runPermute<T>(...)
  fused_prologue_result = cutlass_kernels::fusedBuildExpertMapsSortFirstToken(...);
  if (!fused_prologue_result) { threeStepBuildExpertMapsSortFirstToken(...); }
  cutlass_kernels::expandInputRowsKernelLauncher(input_activations, permuted_data_, token_topk_unpermuted_scales, ...);
  ```

- CuteDSL staged compute (quant → GEMM1 → SwiGLU → quant → GEMM2): [tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py:211-228](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L211-L228)
  ```py
  act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(permuted_data_tensor)
  h1 = cute_dsl_fp8_group_blockwise_gemm_ref(a=act_input_fp8, b=self.w3_w1_weight.view(weight_dtype), a_sf=act_input_sf, b_sf=self.quant_scales[0], offset_array=expert_first_token_offset_tensor)
  h2 = swiglu_fused_moe(h1)
  act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(h2)
  h3 = cute_dsl_fp8_group_blockwise_gemm_ref(a=act_input_fp8, b=self.w2_weight.view(weight_dtype), a_sf=act_input_sf, b_sf=self.quant_scales[1], offset_array=expert_first_token_offset_tensor)
  ```

- Finalize (staged path): [cpp/tensorrt_llm/thop/moeUtilOp.cpp:248-260](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L248-L260)
  ```cpp
  // run_moe_finalize_scale_op(...)
  cutlass_kernels::finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
      static_cast<UnfusedGemmOutputType const*>(gemm2_output), final_output, biases, unpermuted_final_scales,
      unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row, token_selected_experts,
      expert_first_token_offset, num_rows, hidden_size, unpadded_hidden_size, experts_per_token, num_experts_per_node,
      parallelism_config, enable_alltoall, stream);
  ```

## Architecture Overview

```
┌─ MoE.forward ─────────────────────────────────────────────────────────────┐
│ choose backend (Cutlass/CuteDSL/…)                                        │
│ route top‑k experts + scales (custom CUDA for small n_experts/topk)       │
│ optional EP all‑to‑all dispatch (fused comm)                               │
│ MoE compute (GEMM1→SwiGLU→GEMM2; fused or staged)                          │
│ finalize (fused epilogue or separate finalize kernel)                      │
│ TP reduce‑scatter/all‑reduce if needed                                     │
└───────────────────────────────────────────────────────────────────────────┘
```


## Cutlass Path Deep Dive

### Python Entry & Routing

Backend selection and construction happens here:

- create_moe backend selector: `tensorrt_llm/_torch/modules/fused_moe/create_moe.py#L22`

The uniform forward wrapper lives in the interface. It either calls a custom op for torch.compile or falls back to `forward_impl` of the concrete backend:

- interface forwarding: `tensorrt_llm/_torch/modules/fused_moe/interface.py#L151`

Within Cutlass, `forward_impl` computes the per‑rank token count and orchestrates chunking. The interesting part for the fused path is in `forward_chunk`, which performs routing and (optionally) dispatch before invoking the fused op.

Annotated: routing + fused op call

- call site (fused_moe): `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py#L393`

```py
# CutlassFusedMoE.forward_chunk(...)
# 1) Perform routing: token_selected_experts (int32), token_final_scales (fp32)
# 2) Optional dispatch (alltoall / allgather)
# 3) Fused MoE op: routes → GEMM1 → SwiGLU → GEMM2 → (maybe fused finalize)
final_hidden_states = torch.ops.trtllm.fused_moe(
    x,
    token_selected_experts,
    token_final_scales,
    self.w3_w1_weight.view(weight_dtype),
    self.w3_w1_bias,
    self.w2_weight.view(weight_dtype),
    self.w2_bias,
    output_dtype,
    quant_scales=self.quant_scales,
    input_sf=x_sf,
    swizzled_input_sf=is_sf_swizzled,
    swiglu_alpha=self.swiglu_alpha,
    swiglu_beta=self.swiglu_beta,
    swiglu_limit=self.swiglu_limit,
    tp_size=self.tp_size,
    tp_rank=self.tp_rank,
    ep_size=self.ep_size,
    ep_rank=self.ep_rank,
    cluster_size=self.cluster_size,
    cluster_rank=self.cluster_rank,
    enable_alltoall=self.enable_alltoall,
    use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
    use_w4_group_scaling=use_w4_group_scaling,
    use_int8_woq_per_channel=use_int8_woq_per_channel,
    use_mxfp8_act_scaling=use_mxfp8_act_scaling,
    min_latency_mode=False,
    use_fused_finalize=self.use_fused_finalize,
    tune_max_num_tokens=self.tune_max_num_tokens,
    tuner_num_tokens=tuner_num_tokens,
    tuner_top_k=tuner_top_k,
    unpadded_hidden_size=self.unpadded_hidden_size,
)
# The custom op returns a list when min_latency_mode is False → unwrap.
final_hidden_states = final_hidden_states[0]
```

Routing itself is performed via either plain PyTorch or small‑N CUDA ops when `n_experts ≤ 128` and `topk ≤ 8`:

- default (softmax→topk): [tensorrt_llm/_torch/modules/fused_moe/routing.py](../tensorrt_llm/_torch/modules/fused_moe/routing.py#L206-L214)
- renormalize (topk→softmax): [tensorrt_llm/_torch/modules/fused_moe/routing.py](../tensorrt_llm/_torch/modules/fused_moe/routing.py#L254-L263)
- CUDA bindings: [cpp/tensorrt_llm/thop/customMoeRoutingOp.cpp](../cpp/tensorrt_llm/thop/customMoeRoutingOp.cpp#L111-L170)
- CUDA kernel launcher: [cpp/tensorrt_llm/kernels/customMoeRoutingKernels.cu](../cpp/tensorrt_llm/kernels/customMoeRoutingKernels.cu#L213-L260)

### Optional All‑to‑All Dispatch

When EP size is larger than top‑k and DP is enabled, tokens are dispatched across EP ranks before computation, using a fused alltoall path (MNNVL):

- enable & workspace: [tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py#L186-L206)
- prepare (no‑allgather variant): [tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py#L355-L361)
- alltoall fused send: [tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py#L367-L371)
- normalize ids: [tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py#L372-L375)
- DP allgather fallback: [tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py#L376-L387)

The comm ops are bound in C++:

- prepare binding: [cpp/tensorrt_llm/thop/moeCommOp.cpp](../cpp/tensorrt_llm/thop/moeCommOp.cpp#L298-L305)
- alltoall op: [cpp/tensorrt_llm/thop/moeCommOp.cpp](../cpp/tensorrt_llm/thop/moeCommOp.cpp#L255-L262)
- memset expert ids: [cpp/tensorrt_llm/thop/moeCommOp.cpp](../cpp/tensorrt_llm/thop/moeCommOp.cpp#L311-L318)

### Fused MoE Compute (Autotuned GEMM1/2)

The fused op first autotunes GEMM1/GEMM2 tactics, then runs the MoE. This is orchestrated in Python to keep configuration close to the model and device state:

- custom op entry: [tensorrt_llm/_torch/custom_ops/torch_custom_ops.py](../tensorrt_llm/_torch/custom_ops/torch_custom_ops.py#L197-L241)

```py
@torch.library.custom_op("trtllm::fused_moe", mutates_args=())
def fused_moe(...):
    tuner = AutoTuner.get()
    # choose One tactic for GEMM1
    _, gemm_tactic_1 = tuner.choose_one(
        "trtllm::fused_moe::gemm1", [moe_runner], MoERunner.tuning_config,
        [tuner_input, fc1_expert_weights, fc1_expert_biases, fc2_expert_weights, fc2_expert_biases],
        gemm_idx=1,
    )
    # choose One tactic for GEMM2
    _, gemm_tactic_2 = tuner.choose_one(
        "trtllm::fused_moe::gemm2", [moe_runner], MoERunner.tuning_config,
        [tuner_input, fc1_expert_weights, fc1_expert_biases, fc2_expert_weights, fc2_expert_biases],
        gemm_idx=2,
    )
    # run via C++ runner with selected tactics
    output = moe_runner.fused_moe_runner.run_moe(
        input, token_selected_experts, token_final_scales,
        fc1_expert_weights, fc1_expert_biases,
        fc2_expert_weights, fc2_expert_biases,
        quant_scales, input_sf, swizzled_input_sf,
        swiglu_alpha, swiglu_beta, swiglu_limit,
        tp_size, tp_rank, ep_size, ep_rank,
        cluster_size, cluster_rank,
        enable_alltoall, min_latency_mode,
        [gemm_tactic_1, gemm_tactic_2],
        unpadded_hidden_size,
    )
    return output if min_latency_mode else [output]
```

On the C++ side, `FusedMoeRunner::runMoe` validates and forwards into the CUTLASS MoE runner:

- cpp entry: [cpp/tensorrt_llm/thop/moeOp.cpp](../cpp/tensorrt_llm/thop/moeOp.cpp#L200-L520)

```cpp
// FusedMoeRunner::runMoe(...)
// 1) Validate inputs & shapes
// 2) Assemble activation/quant/fusion params and workspaces
// 3) Delegate to CUTLASS runner → runMoe
mKernelRunner->runMoe(
    input.const_data_ptr(),
    /*input_sf*/ input_sf.has_value() ? input_sf.value().const_data_ptr() : nullptr,
    swizzled_input_sf,
    reinterpret_cast<int const*>(token_selected_experts.const_data_ptr()),
    token_final_scales ? reinterpret_cast<float const*>(token_final_scales.value().const_data_ptr()) : nullptr,
    fc1_expert_weights.const_data_ptr(),
    fc1_expert_biases.has_value() ? fc1_expert_biases.value().const_data_ptr() : nullptr,
    activation_params,
    fc2_expert_weights.const_data_ptr(),
    fc2_expert_biases.has_value() ? fc2_expert_biases.value().const_data_ptr() : nullptr,
    quant_params,
    num_rows, hidden_size, unpadded_hidden_size_val, inter_size, num_experts_total, experts_per_token,
    static_cast<char*>(workspace_info.workspace.data_ptr()),
    output.data_ptr(), static_cast<int*>(workspace_info.src_to_dest_map),
    parallelism_config, enable_alltoall,
    /*use_lora*/ false, lora_params,
    mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, stream);
```

Inside the CUTLASS runner, `CutlassMoeFCRunner::runMoe` sets up TMA WS (Hopper/Blackwell) and dispatches grouped GEMMs for GEMM1 and GEMM2, optionally fusing finalize into GEMM2:

- runMoe interface + template: [cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h](../cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h#L600-L780)

```cpp
// CutlassMoeFCRunner::runMoe(...)
// 1) compute TMA WS inputs & strides when using TMA WS
// 2) GEMM1 (may fuse bias+activation): moeGemmBiasAct(...)
// 3) GEMM2 (may fuse finalize): moeGemm(...)
// 4) if not fused finalize → separate finalizeMoeRoutingKernelLauncher(...)
```

Supporting kernels visible in the CU file:

- setup TMA WS & compute strides (kernel launch): [cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L3626-L3668), [..](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L3727-L3768)
- expand/permute token rows (duplicating by top‑k): [cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L1585-L1690)

### Finalize & Combine

If GEMM2 did not fuse finalize, a standalone finalize kernel reduces top‑k outputs with routing scales and unpermutes rows back to token order:

- finalize kernel: [cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L1884-L1960)

When EP all‑to‑all is used, a symmetric combine returns outputs to original ranks and reduces across the expert dimension as needed:

- combine (Python): [tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py#L431-L441)

Finally, TP reduce‑scatter/all‑reduce harmonizes outputs across tensor parallel ranks:

- helper: [tensorrt_llm/_torch/modules/fused_moe/interface.py](../tensorrt_llm/_torch/modules/fused_moe/interface.py#L336-L353)

### Autotuning Hooks

Python‑side autotuner (profiles candidates, caches best tactic):

- autotuner: [tensorrt_llm/_torch/autotuner.py](../tensorrt_llm/_torch/autotuner.py#L511-L720)

FusedMoeRunner exposes profiling entry points used by the Python autotuner:

- get tactic count: [cpp/tensorrt_llm/thop/moeOp.cpp](../cpp/tensorrt_llm/thop/moeOp.cpp#L614-L620)
- runGemmProfile (prep + run): [cpp/tensorrt_llm/thop/moeOp.cpp](../cpp/tensorrt_llm/thop/moeOp.cpp#L622-L760)
- profiler workspace & dispatch: [cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L4466-L4568)


## CuteDSL Path Deep Dive

### Python Entry & Routing

CuteDSL reuses the same interface and routing. The backend class derives from CutlassFusedMoE to share configuration logic, but uses a staged flow for compute:

- backend class: [tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L95-L131)

### Permute / Expand (Token Dispatch in‑kernel)

CuteDSL uses a `moe_permute_op` custom op to build expert maps (fused or 3‑step), expand/permute rows, and produce the mapping tensors required by later stages:

- call site: [tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L193-L215)
- binding: [cpp/tensorrt_llm/thop/moeUtilOp.cpp](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L90-L160)
- template dispatcher (build maps + expand): [cpp/tensorrt_llm/thop/moeUtilOp.cpp](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L38-L88)

Annotated: build maps + expand

```cpp
// runPermute<T>(...)
// 1) Try fused expert map build; else fall back to 3‑step builder
bool fused_prologue_result = cutlass_kernels::fusedBuildExpertMapsSortFirstToken(/*...*/);
if (!fused_prologue_result) {
    cutlass_kernels::threeStepBuildExpertMapsSortFirstToken(/*...*/);
}
// 2) Expand & permute input rows from [num_tokens, H] to [num_tokens*topk, H]
cutlass_kernels::expandInputRowsKernelLauncher(/*...*/);
```

Kernels:

- fused expert map builder: [cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L526-L613)
- 3‑step builder: [cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L877-L980)
- expand rows: [cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu](../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L1585-L1690)

### GEMMs & Activation

CuteDSL path then quantizes activations and uses reference group blockwise GEMMs implemented in Python (for clarity/tests). The arithmetic mirrors what CUTLASS kernels do in fused mode:

- quantize + GEMM1 + SwiGLU + GEMM2: [tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L216-L240)

```py
# Quantize activations → FP8 group/blockwise
act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(permuted_data_tensor)
# GEMM1 (w3_w1), reference CuteDSL path
h1 = cute_dsl_fp8_group_blockwise_gemm_ref(
    a=act_input_fp8, b=self.w3_w1_weight.view(weight_dtype),
    a_sf=act_input_sf, b_sf=self.quant_scales[0], offset_array=expert_first_token_offset_tensor)
# SwiGLU
h2 = swiglu_fused_moe(h1)
# Quantize activations again if needed and GEMM2 (w2)
act_input_fp8, act_input_sf = torch.ops.trtllm.fp8_quantize_1x128(h2)
h3 = cute_dsl_fp8_group_blockwise_gemm_ref(
    a=act_input_fp8, b=self.w2_weight.view(weight_dtype),
    a_sf=act_input_sf, b_sf=self.quant_scales[1], offset_array=expert_first_token_offset_tensor)
```

### Finalize

Finally, CuteDSL calls the same finalize custom op used by the staged CUTLASS path. It reduces across top‑k per token using routing scales and unpermutes back to token order:

- call site: [tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py](../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L241-L259)
- binding: [cpp/tensorrt_llm/thop/moeUtilOp.cpp](../cpp/tensorrt_llm/thop/moeUtilOp.cpp#L248-L340)

```py
final_hidden_states = torch.ops.trtllm.moe_finalize_scale_op(
    h3, None, token_final_scales,
    unpermuted_row_to_permuted_row_tensor,
    permuted_row_to_unpermuted_row_tensor,
    token_selected_experts,
    expert_first_token_offset_tensor,
    False,  # enable_alltoall
    x.shape[0],            # num_rows
    x.shape[1],            # hidden_size (possibly padded)
    self.unpadded_hidden_size,
    self.routing_method.top_k,
    self.expert_size_per_partition,
    self.tp_size, self.tp_rank, self.ep_size, self.ep_rank,
)
```


## Dataflow Diagrams

Cutlass (Fused) Path

```
route → [A2A?] → fused_moe (autotuned) → [finalize fused? else kernel] → [A2A combine?] → TP reduce

CUDA highlights inside fused_moe/runMoe:
  build maps → expand → TMA WS → GEMM1+SwiGLU → GEMM2(+finalize?) → finalize kernel (if needed)
```

CuteDSL (Staged) Path

```
route → permute/expand → quant → GEMM1 → SwiGLU → quant → GEMM2 → finalize → TP reduce
```


## Key Files Index

- Backend selection: `tensorrt_llm/_torch/modules/fused_moe/create_moe.py#L22`
- MoE interface/forward wrapper: `tensorrt_llm/_torch/modules/fused_moe/interface.py#L151`
- Routing methods: `tensorrt_llm/_torch/modules/fused_moe/routing.py#L74`
- Cutlass backend: `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py#L222`
- CuteDSL backend: `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py#L84`
- Fused MoE op (Python): `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py#L197`
- AutoTuner core: `tensorrt_llm/_torch/autotuner.py#L511`
- FusedMoeRunner: `cpp/tensorrt_llm/thop/moeOp.cpp#L57`
- CUTLASS MoE interfaces: `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h#L445`
- Grouped GEMM dispatch: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h#L939`
- Expert map/expand/finalize kernels: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu#L480`
- All‑to‑all ops: `cpp/tensorrt_llm/thop/moeCommOp.cpp#L240`
