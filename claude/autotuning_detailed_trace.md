# Complete Autotuning Trace: TensorRT-LLM AutoTuner

This document provides an exhaustive, line-by-line trace of the autotuning mechanism in TensorRT-LLM, from Python user code through to kernel selection and execution.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Overview](#component-overview)
3. [Complete Call Flow](#complete-call-flow)
4. [Detailed Traces](#detailed-traces)
   - [Initialization & Setup](#initialization--setup)
   - [First Invocation (Tuning Mode)](#first-invocation-tuning-mode)
   - [Subsequent Invocations (Inference Mode)](#subsequent-invocations-inference-mode)
5. [Data Structures](#data-structures)
6. [Cache Mechanism](#cache-mechanism)
7. [Example Walkthrough](#example-walkthrough)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTOTUNING ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      USER CODE (Python)                          │
│                                                                  │
│  with autotune(tune_mode=True, cache_path="cache.json"):        │
│      output = custom_op(input, weight, ...)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              CUSTOM OP IMPLEMENTATION (Python)                   │
│                                                                  │
│  @torch.library.custom_op("trtllm::my_op", ...)                 │
│  def my_op(input, weight, ...):                                 │
│      tuner = AutoTuner.get()                                    │
│      runner, tactic = tuner.choose_one(                         │
│          "trtllm::my_op",                                        │
│          [MyRunner()],                                           │
│          tuning_config,                                          │
│          [input, weight, ...]                                    │
│      )                                                           │
│      return runner(inputs, tactic=tactic)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AutoTuner.choose_one()                        │
│                                                                  │
│  If tuning_mode:                                                 │
│    ┌──────────────────────────────────────────────────────┐    │
│    │ 1. Generate optimization profiles                    │    │
│    │ 2. For each profile:                                 │    │
│    │    a. Check cache                                    │    │
│    │    b. If miss: Profile all (runner, tactic) pairs   │    │
│    │    c. Select best and cache result                  │    │
│    └──────────────────────────────────────────────────────┘    │
│  Else:                                                           │
│    ┌──────────────────────────────────────────────────────┐    │
│    │ 1. Search cache for matching config                  │    │
│    │ 2. Return cached (runner, tactic)                    │    │
│    │ 3. Or fallback if cache miss                         │    │
│    └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 RUNNER IMPLEMENTATION                            │
│                                                                  │
│  class MyRunner(TunableRunner):                                 │
│      def get_valid_tactics(self, inputs, profile):              │
│          return list_of_valid_tactics                           │
│                                                                  │
│      def forward(self, inputs, tactic):                         │
│          # Execute kernel with tactic                           │
│          return result                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   KERNEL EXECUTION                               │
│                                                                  │
│  C++ / CUDA kernel launch with selected configuration           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Overview

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `AutoTuner` | [autotuner.py:511-953](../tensorrt_llm/_torch/autotuner.py#L511-L953) | Main singleton coordinating autotuning |
| `TunableRunner` | [autotuner.py:150-207](../tensorrt_llm/_torch/autotuner.py#L150-L207) | Abstract base for kernel implementations |
| `TuningConfig` | [autotuner.py:52-98](../tensorrt_llm/_torch/autotuner.py#L52-L98) | Configuration for dynamic dimensions |
| `AutoTunerProfilingCache` | [autotuner.py:298-508](../tensorrt_llm/_torch/autotuner.py#L298-L508) | Cache for profiling results |
| `OptimizationProfile` | [autotuner.py:123-139](../tensorrt_llm/_torch/autotuner.py#L123-L139) | Dimension ranges for profiling |
| `DynamicTensorSpec` | [autotuner.py:22-35](../tensorrt_llm/_torch/autotuner.py#L22-L35) | Specification for dynamic dimensions |

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `autotune()` | [autotuner.py:210-242](../tensorrt_llm/_torch/autotuner.py#L210-L242) | Context manager for tuning mode |
| `choose_one()` | [autotuner.py:542-649](../tensorrt_llm/_torch/autotuner.py#L542-L649) | Main entry point for tactic selection |
| `_profile_runners()` | [autotuner.py:651-707](../tensorrt_llm/_torch/autotuner.py#L651-L707) | Profile all runner/tactic combinations |
| `_profile_single_kernel()` | [autotuner.py:719-767](../tensorrt_llm/_torch/autotuner.py#L719-L767) | Profile one kernel with timing |
| `_optimization_profiles()` | [autotuner.py:769-850](../tensorrt_llm/_torch/autotuner.py#L769-L850) | Generate profiles via cartesian product |

---

## Complete Call Flow

### High-Level Flow Diagram

```
User Code
   │
   ├─→ autotune(tune_mode=True, cache_path="...")
   │     │
   │     └─→ AutoTuner.get().is_tuning_mode = True
   │
   ├─→ custom_op(inputs)
   │     │
   │     ├─→ Runner instantiation
   │     │     └─→ MyRunner(config_params)
   │     │
   │     ├─→ AutoTuner.get().choose_one(op_name, [runner], config, inputs)
   │     │     │
   │     │     ├─→ IF tuning_mode:
   │     │     │     ├─→ _optimization_profiles(config, inputs)
   │     │     │     │     └─→ Generate cartesian product of dimension buckets
   │     │     │     │
   │     │     │     └─→ For each profile:
   │     │     │           ├─→ _prepare_input_tensors(profile, inputs)
   │     │     │           │     └─→ Create tensors with profile shapes
   │     │     │           │
   │     │     │           ├─→ profiling_cache.search_cache(...)
   │     │     │           │     └─→ Check if this profile already tuned
   │     │     │           │
   │     │     │           └─→ IF cache miss:
   │     │     │                 └─→ _profile_runners(...)
   │     │     │                       │
   │     │     │                       └─→ For each runner:
   │     │     │                             ├─→ runner.get_valid_tactics(...)
   │     │     │                             │     └─→ Returns list of valid tactics
   │     │     │                             │
   │     │     │                             └─→ For each tactic:
   │     │     │                                   ├─→ _profile_single_kernel(...)
   │     │     │                                   │     ├─→ Warmup iterations
   │     │     │                                   │     ├─→ Timed iterations
   │     │     │                                   │     └─→ Return avg time
   │     │     │                                   │
   │     │     │                                   └─→ Select best (runner, tactic)
   │     │     │                                         └─→ Cache result
   │     │     │
   │     │     └─→ ELSE (inference mode):
   │     │           └─→ profiling_cache.search_cache(...)
   │     │                 ├─→ Return cached (runner_id, tactic)
   │     │                 └─→ Or fallback tactic if cache miss
   │     │
   │     └─→ runner.forward(inputs, tactic=best_tactic)
   │           └─→ Kernel execution
   │
   └─→ autotune.__exit__()
         ├─→ AutoTuner.get().is_tuning_mode = old_mode
         └─→ profiling_cache.save_cache(cache_path)
```

---

## Detailed Traces

### Initialization & Setup

#### Step 1: User enters tuning context

**Location:** [`tensorrt_llm/_torch/autotuner.py:210-242`](../tensorrt_llm/_torch/autotuner.py#L210-L242)

```python
# Line 210-242
@contextlib.contextmanager
def autotune(tune_mode: bool = True, cache_path: str = None, rank: int = 0):
    """Context manager for enabling autotuning mode.

    Args:
        tune_mode: Whether to enable tuning (default: True)
        cache_path: Path to save/load cache (optional)
        rank: GPU rank for distributed setups (default: 0)
    """
    # Check if rank-specific cache file exists
    tune_required = tune_mode
    if cache_path is not None:
        cache_path_no_ext = os.path.splitext(cache_path)[0]
        cache_path_no_ext_rank = cache_path_no_ext + f".rank{rank}.json"
        file_exists = os.path.exists(cache_path_no_ext_rank)

        # Don't tune if cache exists
        tune_required = tune_required and not os.path.exists(cache_path)

        # Load existing cache if available
        if file_exists:
            logger.info(
                f"[Autotuner] Loading cache from {cache_path_no_ext_rank}")
            AutoTuner.get().profiling_cache.load_cache(cache_path_no_ext_rank)

    # Save old tuning mode
    old_mode = AutoTuner.get().is_tuning_mode
    AutoTuner.get().is_tuning_mode = tune_required

    autotune_enabled = tune_required and not old_mode
    if autotune_enabled:
        logger.info("[Autotuner] Autotuning process starts ...")

    try:
        yield  # User code executes here
    finally:
        # Restore old mode
        AutoTuner.get().is_tuning_mode = old_mode

        if autotune_enabled:
            logger.info("[Autotuner] Autotuning process ends")

        # Save cache to disk
        if cache_path is not None:
            logger.info(f"[Autotuner] Saving cache to {cache_path_no_ext_rank}")
            AutoTuner.get().profiling_cache.save_cache(cache_path_no_ext_rank)
```

**What happens:**
1. Creates rank-specific cache filename: `cache.rank0.json`
2. Loads existing cache if available (skips tuning)
3. Sets `AutoTuner.is_tuning_mode = True` if cache not found
4. After user code, saves cache to disk

#### Step 2: AutoTuner singleton initialization

**Location:** [`tensorrt_llm/_torch/autotuner.py:511-540`](../tensorrt_llm/_torch/autotuner.py#L511-L540)

```python
# Line 511-540
class AutoTuner:
    """AutoTuner for optimizing TensorRT LLM operations."""
    _instance = None  # Singleton instance

    def __init__(self, warmup=3, repeat=10, stream_delay_micro_secs=1000):
        """Initialize AutoTuner with profiling parameters.

        Args:
            warmup: Number of warmup iterations (default: 3)
            repeat: Number of profiling iterations (default: 10)
            stream_delay_micro_secs: Delay before profiled kernel (default: 1000)
        """
        self.repeat = repeat
        self.warmup = warmup
        self.stream_delay_micro_secs = stream_delay_micro_secs

        # Create profiling cache
        self.profiling_cache = AutoTunerProfilingCache()

        # Tuning mode flag
        self.is_tuning_mode = False

        # Statistics tracking
        self.stats = AutoTunerStatistics()

        self.profiling_debug = True

    @classmethod
    def get(cls):
        """Get singleton AutoTuner instance."""
        if cls._instance is None:
            cls._instance = AutoTuner()
        return cls._instance
```

**Key members:**
- `repeat=10`: Each tactic benchmarked 10 times
- `warmup=3`: 3 warmup runs before timing
- `stream_delay_micro_secs=1000`: 1ms delay to reduce host overhead noise
- `profiling_cache`: Stores (op, runner, shape) → (runner_id, tactic, time)
- `is_tuning_mode`: Controls tuning vs inference behavior

---

### First Invocation (Tuning Mode)

#### Step 3: Custom op called with inputs

**Example Location:** [`tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py:288-313`](../tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py#L288-L313)

```python
# Line 288-313
@torch.library.custom_op("trtllm::cute_dsl_nvfp4_gemm_blackwell",
                         mutates_args=(),
                         device_types="cuda")
def cute_dsl_nvfp4_gemm_blackwell(
    input: torch.Tensor,           # [m, k]
    weight: torch.Tensor,          # [n, k]
    input_scale: torch.Tensor,     # [m, k//16]
    weight_scale: torch.Tensor,    # [n, k//16]
    alpha: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """FP4 GEMM on Blackwell (SM100) using CuTe DSL.

    This custom op uses autotuning to select optimal kernel configuration.
    """

    # Get global AutoTuner singleton
    tuner = AutoTuner.get()

    # Instantiate runner with operation-specific config
    cute_dsl_nvfp4_gemm_blackwell_runner = CuteDSLNVFP4BlackwellLinear(
        alpha, output_dtype)

    # Choose best runner and tactic
    _, best_tactic = tuner.choose_one(
        "trtllm::cute_dsl_nvfp4_gemm_blackwell",  # Operation name
        [cute_dsl_nvfp4_gemm_blackwell_runner],   # List of runners
        CuteDSLNVFP4BlackwellLinear.tuning_config, # Tuning config
        [input, weight, input_scale, weight_scale], # Inputs for profiling
    )

    # Execute with best tactic
    return cute_dsl_nvfp4_gemm_blackwell_runner(
        inputs=[input, weight, input_scale, weight_scale],
        tactic=best_tactic,
    )
```

**What happens:**
1. Gets singleton `AutoTuner` instance
2. Creates `TunableRunner` instance with op-specific parameters
3. Calls `choose_one()` to select best (runner, tactic)
4. Executes runner with selected tactic

#### Step 4: Runner definition

**Example Location:** [`tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py:31-126`](../tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py#L31-L126)

```python
# Line 31-126
class CuteDSLNVFP4BlackwellLinear(TunableRunner):
    """Runner for FP4 GEMM using CuTe DSL on Blackwell."""

    # Shared kernel cache across all instances
    kernel_dict = dict()

    # Tuning configuration (class-level)
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                0, 0,  # input_idx=0, dim_idx=0 (batch/token dimension)
                get_last_power_of_2_num_tokens_buckets,  # Generate buckets
                last_positive_power_of_2  # Map to bucket
            ),
        ),
        constraint_specs=(
            ConstraintSpec(2, 0, fp4_scale_infer_shape),  # Scale shape constraint
        ),
    )

    def __init__(self, alpha: float, output_dtype: torch.dtype):
        super().__init__()
        self.alpha = alpha
        self.output_dtype = output_dtype
        assert output_dtype == torch.bfloat16

        if get_sm_version() != 100:
            raise ValueError(
                f"SM version {get_sm_version()} is not supported, "
                "it only supports SM 100"
            )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[Tuple[int, int]]:
        """Generate list of valid (mma_tiler, cluster_shape, swap_ab) configs.

        Returns:
            List of tactic tuples, each representing a valid kernel config
        """
        assert inputs[0].dim() == 2
        assert inputs[1].dim() == 2

        m = inputs[0].shape[0]
        n = inputs[1].shape[0]
        k = inputs[0].shape[1]
        real_k = k * 2  # FP4 packing: 2 values per byte

        batch_size = 1
        sf_vec_size = 16
        a_major = "k"  # Input layout
        b_major = "k"  # Weight layout

        # Candidate configurations
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
                    # Determine output layout
                    if swap_ab:
                        c_major = "m"
                        kernel_m = n
                        kernel_n = m
                    else:
                        c_major = "n"
                        kernel_m = m
                        kernel_n = n

                    # Check if kernel can implement this config
                    if Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
                        cutlass.Float4E2M1FN,   # ab_dtype
                        cutlass.Float8E4M3FN,   # sf_dtype
                        sf_vec_size,
                        cutlass.BFloat16,       # c_dtype
                        mma_tiler_mn,
                        cluster_shape_mn,
                        kernel_m, kernel_n, real_k,
                        batch_size,
                        a_major, b_major, c_major,
                    ):
                        valid_tactics.append(
                            (mma_tiler_mn, cluster_shape_mn, swap_ab)
                        )

        return valid_tactics

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic,
    ) -> torch.Tensor:
        """Execute GEMM with specified tactic.

        Args:
            inputs: [input, weight, input_scale, weight_scale]
            tactic: (mma_tiler_mn, cluster_shape_mn, swap_ab) tuple

        Returns:
            Output tensor [m, n] in bf16
        """
        # Parse tactic
        if isinstance(tactic, tuple):
            mma_tiler_mn, cluster_shape_mn, swap_ab = tactic
        else:
            # Fallback configuration
            mma_tiler_mn, cluster_shape_mn, swap_ab = [
                (128, 128), (1, 1), False
            ]

        a_tensor, b_tensor, a_sf_tensor, b_sf_tensor = inputs
        m, k, n = a_tensor.shape[0], a_tensor.shape[1], b_tensor.shape[0]

        # Allocate output
        c_tensor = torch.empty(
            *(m, n),
            dtype=self.output_dtype,
            device="cuda"
        )

        if swap_ab:
            c_tensor = c_tensor.permute(1, 0)

        # Get or compile kernel
        CACHE_KEY = (sf_vec_size, mma_tiler_mn, cluster_shape_mn, swap_ab)

        if CACHE_KEY not in CuteDSLNVFP4BlackwellLinear.kernel_dict:
            # Compile new kernel using CuTe DSL
            gemm = Sm100BlockScaledPersistentDenseGemmKernelWrapper(
                sf_vec_size, mma_tiler_mn, cluster_shape_mn
            )

            # Compile with concrete shapes
            compiled_gemm = cute.compile(gemm, kernel_m, kernel_n, real_k, ...)

            # Cache compiled kernel
            CuteDSLNVFP4BlackwellLinear.kernel_dict[CACHE_KEY] = compiled_gemm
        else:
            compiled_gemm = CuteDSLNVFP4BlackwellLinear.kernel_dict[CACHE_KEY]

        # Launch kernel
        compiled_gemm(kernel_m, kernel_n, real_k, ...)

        if swap_ab:
            c_tensor = c_tensor.permute(1, 0)

        return c_tensor
```

**Key points:**
- `get_valid_tactics()`: Generates cartesian product of configurations
  - 6 MMA tiler shapes × 9 cluster shapes × 2 swap options = **108 tactics**
  - Filtered by `can_implement()` check
- `forward()`: Compiles kernel on first use, caches for reuse
- Tactic is a tuple: `(mma_tiler_mn, cluster_shape_mn, swap_ab)`

#### Step 5: AutoTuner.choose_one() entry

**Location:** [`tensorrt_llm/_torch/autotuner.py:542-649`](../tensorrt_llm/_torch/autotuner.py#L542-L649)

```python
# Line 542-649
def choose_one(
    self,
    custom_op: str,                    # "trtllm::cute_dsl_nvfp4_gemm_blackwell"
    runners: List[TunableRunner],      # [CuteDSLNVFP4BlackwellLinear(...)]
    tuning_config: TuningConfig,       # Dynamic dimension config
    inputs: List[torch.Tensor],        # [input, weight, input_scale, weight_scale]
    **kwargs,
) -> Tuple:
    """Choose best runner and tactic through performance profiling.

    Returns:
        (best_runner, best_tactic)
    """

    # Get input shapes
    input_shapes = tuple(self._get_input_sizes(inputs))
    # Example: ((1024, 4096), (14336, 4096), (1024, 256), (14336, 256))

    # ========== INFERENCE MODE ==========
    if not self.is_tuning_mode:
        # Search cache
        is_cache_hit, best_runner_id, best_tactic, min_time = \
            self.profiling_cache.search_cache(
                custom_op, runners, input_shapes, tuning_config
            )

        best_runner = runners[best_runner_id]

        if not is_cache_hit:
            logger.warning_once(
                f"[AutoTunner] Using fallback tactic, cache miss on "
                f"input shapes={input_shapes}",
                key=custom_op
            )

        return (best_runner, best_tactic)

    # ========== TUNING MODE ==========
    assert len(runners) > 0, "At least one runner is required"
    assert all([isinstance(r, TunableRunner) for r in runners]), \
        "All runners must be subclass of TunableRunner"

    # Generate optimization profiles
    profiles = self._optimization_profiles(tuning_config, inputs)
    # Example: 8 profiles for token buckets [128, 256, 512, 1024, 2048, 4096, ...]

    # Record total configs to try
    self.stats.tuned_op_total_configs[custom_op] = len(profiles)

    new_tuning_failure_occured = False

    # Profile each optimization profile
    for p in profiles:
        # Create tensors matching profile shapes
        tensors = self._prepare_input_tensors(p, inputs)

        # Check if this profile already cached
        is_cache_hit, *_ = self.profiling_cache.search_cache(
            custom_op, runners, p.get_opt_shapes(), tuning_config
        )

        if not is_cache_hit:
            # Profile all (runner, tactic) combinations
            best_runner_id, best_tactic, min_time, has_tuning_failure_occured = \
                self._profile_runners(
                    custom_op, runners, tensors, p, tuning_config, **kwargs
                )

            if best_runner_id is not None:
                # Valid tactic found, cache it
                cache_key = self.profiling_cache.get_cache_key(
                    custom_op,
                    runners[best_runner_id],
                    p.get_opt_shapes(),
                    tuning_config
                )

                self.profiling_cache[cache_key] = (
                    best_runner_id, best_tactic, min_time
                )

                self.stats.tuned_op_successful_configs[custom_op] = \
                    self.stats.tuned_op_successful_configs.get(custom_op, 0) + 1

                logger.debug(
                    f"[Autotuner] Profiling runner={runners[best_runner_id]}, "
                    f"tactic={best_tactic} for cache_key={cache_key}."
                )
            else:
                logger.warning(
                    f"[Autotuner] No valid runner/tactic found for "
                    f"custom_op={custom_op}, input_shapes={input_shapes}"
                )

            new_tuning_failure_occured = \
                new_tuning_failure_occured or has_tuning_failure_occured

    # Log tuning failures
    if new_tuning_failure_occured:
        logger.warning(
            f"[Autotuner] New tuning error occurs: "
            f"Total failed profiling tactics: "
            f"{len(self.stats.failed_profiling_count[custom_op])} "
            f"for custom_op={custom_op}"
        )

    # Get best runner and tactic from cache
    _, runner_id, tactic, _ = self.profiling_cache.search_cache(
        custom_op, runners, input_shapes, tuning_config
    )

    return (runners[runner_id], tactic)
```

**Flow:**
1. **Inference mode:** Direct cache lookup, return cached (runner, tactic)
2. **Tuning mode:**
   - Generate multiple optimization profiles
   - For each profile:
     - Check cache
     - If miss: Profile all (runner, tactic) pairs
     - Cache best result
   - Return best overall

#### Step 6: Generate optimization profiles

**Location:** [`tensorrt_llm/_torch/autotuner.py:769-850`](../tensorrt_llm/_torch/autotuner.py#L769-L850)

```python
# Line 769-850
def _optimization_profiles(
    self,
    tuning_config: TuningConfig,
    inputs: List[torch.Tensor]
) -> List[OptimizationProfile]:
    """Generate optimization profiles via cartesian product of dynamic dims.

    Args:
        tuning_config: Configuration with dynamic_tensor_specs
        inputs: Current input tensors

    Returns:
        List of OptimizationProfile objects
    """

    # Create base profile from actual input shapes
    base_profile = OptimizationProfile(
        [[StaticDim(x) for x in t.size()]
         if isinstance(t, torch.Tensor) else [StaticDim(0)]
         for t in inputs]
    )
    # Example base_profile.shapes:
    # [[StaticDim(1024), StaticDim(4096)],      # input
    #  [StaticDim(14336), StaticDim(4096)],     # weight
    #  [StaticDim(1024), StaticDim(256)],       # input_scale
    #  [StaticDim(14336), StaticDim(256)]]      # weight_scale

    generated_profiles: List[OptimizationProfile] = []
    dynamic_dims = []

    # Process each dynamic tensor spec
    for spec in tuning_config.dynamic_tensor_specs:
        # spec.gen_tuning_buckets can be:
        # - A function: lambda max_tokens: [128, 256, 512, ...]
        # - A list/tuple: (128, 256, 512, 1024)

        if inspect.isfunction(spec.gen_tuning_buckets):
            if tuning_config.tune_max_num_tokens is None:
                # Use current input size
                opt_shapes = spec.gen_tuning_buckets(
                    base_profile.shapes[spec.input_idx][spec.dim_idx].val
                )
            else:
                # Use max_num_tokens
                opt_shapes = spec.gen_tuning_buckets(
                    tuning_config.tune_max_num_tokens
                )
        else:
            # Use provided list
            opt_shapes = spec.gen_tuning_buckets

        # Add current input value to bucket set
        opt_shapes = set(opt_shapes)
        opt_shapes.add(
            spec.map_to_tuning_buckets(
                base_profile.shapes[spec.input_idx][spec.dim_idx].val
            )
        )
        opt_shapes = sorted(list(opt_shapes))

        # Create ranges: [min, opt, max]
        # max for bucket i is opt for bucket i+1, except last bucket (inf)
        opt_shapes_max = tuple(opt_shapes[1:]) + (float('inf'), )
        opt_shapes_max = {
            v1: v2 for v1, v2 in zip(opt_shapes, opt_shapes_max)
        }

        dynamic_dims.append(
            (spec.input_idx, spec.dim_idx, opt_shapes_max, opt_shapes)
        )

    # Example dynamic_dims:
    # [(0, 0, {128: 256, 256: 512, 512: 1024, 1024: inf}, [128, 256, 512, 1024])]
    #   │  │   └─ max values                                └─ opt values
    #   │  └─ dim_idx (token dimension)
    #   └─ input_idx (input tensor)

    # Cartesian product of all dynamic dimensions
    dim_grids = itertools.product(*[d[-1] for d in dynamic_dims])
    # Example: If 1 dynamic dim with 4 buckets → 4 profiles
    #          If 2 dynamic dims with 4×3 buckets → 12 profiles

    for opt_point in dim_grids:
        # opt_point example: (512,) for 1D case
        p = copy.deepcopy(base_profile)

        for pos, (input_idx, dim_idx, opt_shapes_max, opt_shapes) in \
                enumerate(dynamic_dims):
            opt_value = opt_point[pos]
            min_value = opt_value  # min = opt for now
            max_value = opt_shapes_max[opt_value]

            # Replace static dim with dynamic dim
            p.shapes[input_idx][dim_idx] = DynamicDim(
                min_value, opt_value, max_value
            )

        # Apply constraint specs (for dependent dimensions)
        for spec in tuning_config.constraint_specs:
            min_value = opt_value = max_value = spec.infer_shape(
                p.get_opt_shapes()
            )

            if p.shapes[spec.input_idx] == [StaticDim(0)]:
                continue  # Skip optional inputs

            p.shapes[spec.input_idx][spec.dim_idx] = DynamicDim(
                min_value, opt_value, max_value
            )

        generated_profiles.append(p)
        logger.debug(f"[Autotuner] Generated profile: {p}")

    return generated_profiles
```

**Example output for FP4 GEMM:**

```python
# Input: [1024, 4096] with dynamic_tensor_specs on dim 0
# Buckets: [128, 256, 512, 1024, 2048, 4096]

Generated profiles:
1. shapes=[[DynamicDim(128, 128, 256), StaticDim(4096)], ...]
2. shapes=[[DynamicDim(256, 256, 512), StaticDim(4096)], ...]
3. shapes=[[DynamicDim(512, 512, 1024), StaticDim(4096)], ...]
4. shapes=[[DynamicDim(1024, 1024, 2048), StaticDim(4096)], ...]
5. shapes=[[DynamicDim(2048, 2048, 4096), StaticDim(4096)], ...]
6. shapes=[[DynamicDim(4096, 4096, inf), StaticDim(4096)], ...]
```

#### Step 7: Profile runners for each profile

**Location:** [`tensorrt_llm/_torch/autotuner.py:651-707`](../tensorrt_llm/_torch/autotuner.py#L651-L707)

```python
# Line 651-707
def _profile_runners(
    self,
    custom_op: str,
    runners: List[TunableRunner],
    input_tensors: List[torch.Tensor],
    profile: OptimizationProfile,
    tuning_config: TuningConfig,
    **kwargs,
) -> Tuple[int, Any, float, bool]:
    """Profile all (runner, tactic) combinations for given profile.

    Returns:
        (best_runner_id, best_tactic, min_time, has_failure)
    """
    min_time = float('inf')
    has_tuning_failure_occured = False
    best_runner_id, best_tactic = None, None

    # Iterate over all runners
    for runner_id, runner in enumerate(runners):
        # Check if runner.forward() accepts do_preparation
        runner_arg_names = {
            p.name
            for p in inspect.signature(runner.forward).parameters.values()
        }

        # Get valid tactics for this runner
        valid_tactics = runner.get_valid_tactics(
            input_tensors, profile, **kwargs
        )
        # Example: [(mma_128_128, cluster_1_1, False),
        #           (mma_256_128, cluster_2_2, True), ...]

        # Call preparation phase if supported
        if "do_preparation" in runner_arg_names and len(valid_tactics) > 0:
            runner(
                input_tensors,
                tactic=-1,
                do_preparation=True,
                **kwargs,
            )

        # Profile each tactic
        for tac in valid_tactics:
            try:
                # Profile this (runner, tactic) combination
                time_measured = self._profile_single_kernel(
                    runner, input_tensors, tac, **kwargs
                )
            except Exception as e:
                # Handle profiling failures
                shapes = self._get_input_sizes(input_tensors)
                logger.warning(
                    f"[Autotuner] Failed when profiling runner={runner}, "
                    f"tactic={tac}, shapes={shapes}. "
                    "Set TLLM_LOG_LEVEL=DEBUG for more details."
                )
                logger.debug(f"[Autotuner] Exception captured: {e}")

                # Record failed combination
                if custom_op not in self.stats.failed_profiling_count:
                    self.stats.failed_profiling_count[custom_op] = set()
                self.stats.failed_profiling_count[custom_op].add(
                    self.profiling_cache.get_cache_key(
                        custom_op, runner, profile.get_opt_shapes(),
                        tuning_config
                    )
                )

                # Mark as failed
                time_measured = float('inf')
                has_tuning_failure_occured = True

            # Update best if faster
            if time_measured < min_time:
                min_time = time_measured
                best_runner_id, best_tactic = runner_id, tac

    return best_runner_id, best_tactic, min_time, has_tuning_failure_occured
```

**Flow:**
1. For each runner:
   - Call `runner.get_valid_tactics()` to get candidate tactics
   - For each tactic:
     - Profile with `_profile_single_kernel()`
     - Handle exceptions gracefully
     - Track best time
2. Return best (runner_id, tactic, time)

#### Step 8: Profile single kernel

**Location:** [`tensorrt_llm/_torch/autotuner.py:719-767`](../tensorrt_llm/_torch/autotuner.py#L719-L767)

```python
# Line 719-767
def _profile_single_kernel(
    self,
    runner: TunableRunner,
    inputs: List[torch.Tensor],
    tactic: Any,
    **kwargs,
) -> float:
    """Profile single kernel implementation with accurate timing.

    Args:
        runner: TunableRunner instance
        inputs: Input tensors
        tactic: Tactic to profile

    Returns:
        Average execution time in milliseconds
    """
    stream = torch.cuda.current_stream()

    # ========== WARMUP PHASE ==========
    # Run kernel multiple times without timing
    # Purpose: Prime caches, compile kernels, stabilize GPU state
    for _ in range(self.warmup):  # default: 3 iterations
        runner(inputs, tactic=tactic, **kwargs)
    stream.synchronize()

    # ========== DELAY INJECTION ==========
    # Inject delay on CUDA stream before profiled kernel
    # Purpose: Reduce host-side overhead noise in measurements
    # The delay ensures kernel launch latency doesn't affect timing
    delay_kernel(self.stream_delay_micro_secs, stream)  # default: 1000 μs

    # ========== TIMING PHASE ==========
    # Create CUDA events for accurate GPU timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Record start event
    start.record(stream=stream)

    # Run kernel multiple times for averaging
    for _ in range(self.repeat):  # default: 10 iterations
        runner(inputs, tactic=tactic, **kwargs)

    # Record end event
    end.record(stream=stream)

    # Wait for all kernels to complete
    stream.synchronize()

    # Calculate average time
    avg_time = start.elapsed_time(end) / self.repeat  # ms

    # Log profiling result
    shapes = self._get_input_sizes(inputs)
    logger.debug(
        f"[Autotuner] Profiled runner={runner}, tactic={tactic}, "
        f"shapes={shapes}: {avg_time:.6f}ms."
    )

    return avg_time
```

**Key techniques:**
1. **Warmup:** 3 iterations to prime caches and compile kernels
2. **Delay injection:** 1ms delay to eliminate host overhead
3. **CUDA events:** GPU-side timing for accurate measurements
4. **Averaging:** 10 iterations to reduce variance

**Example output:**
```
[Autotuner] Profiled runner=CuteDSLNVFP4BlackwellLinear,
            tactic=(mma_128_128, cluster_1_1, False),
            shapes=[(1024, 4096), ...]: 0.245123ms
```

#### Step 9: Cache result

**Location:** [`tensorrt_llm/_torch/autotuner.py:612-617`](../tensorrt_llm/_torch/autotuner.py#L612-L617)

```python
# Line 612-617 (from choose_one)
if best_runner_id is not None:
    # Build cache key
    cache_key = self.profiling_cache.get_cache_key(
        custom_op,
        runners[best_runner_id],
        p.get_opt_shapes(),
        tuning_config
    )

    # Store (runner_id, tactic, time) in cache
    self.profiling_cache[cache_key] = (
        best_runner_id, best_tactic, min_time
    )
```

**Cache key structure:**

**Location:** [`tensorrt_llm/_torch/autotuner.py:358-375`](../tensorrt_llm/_torch/autotuner.py#L358-L375)

```python
# Line 358-375
def get_cache_key(
    self,
    custom_op: str,
    runner: TunableRunner,
    input_shapes: Tuple[torch.Size],
    tuning_config: TuningConfig,
) -> Tuple:
    """Build cache key from operation, runner, and shapes.

    Returns:
        Tuple: (custom_op, runner_class_name, runner_hash, profile_tuple)
    """
    return (
        custom_op,                               # "trtllm::cute_dsl_nvfp4_gemm_blackwell"
        runner.__class__.__name__,               # "CuteDSLNVFP4BlackwellLinear"
        hash(runner),                            # Hash of runner attributes
        AutoTuner.get()._find_nearest_profile(  # Nearest profile bucket
            input_shapes,
            tuning_config.dynamic_tensor_specs,
            tuning_config.constraint_specs,
            tuning_config.tune_max_num_tokens,
        ),
    )
```

**Example cache key:**
```python
(
    "trtllm::cute_dsl_nvfp4_gemm_blackwell",
    "CuteDSLNVFP4BlackwellLinear",
    -1234567890,  # hash(runner)
    ((1024, 4096), (14336, 4096), (1024, -1), (14336, -1))
    #                               ^     ^      ^      ^
    #                               |     |      |      |
    #                               |     |      |      └─ Constrained dimension
    #                               |     |      └─ Constrained dimension
    #                               |     └─ Static dimension
    #                               └─ Bucketed dimension (1024)
)
```

#### Step 10: Execute with best tactic

After `choose_one()` returns, the custom op calls the runner:

```python
# Back in custom op (cute_dsl_custom_ops.py:310-313)
return cute_dsl_nvfp4_gemm_blackwell_runner(
    inputs=[input, weight, input_scale, weight_scale],
    tactic=best_tactic,  # e.g., (mma_256_128, cluster_2_2, False)
)
```

Runner executes with the selected tactic:

```python
# In CuteDSLNVFP4BlackwellLinear.forward()
# 1. Get or compile kernel for this tactic
CACHE_KEY = (sf_vec_size, mma_tiler_mn, cluster_shape_mn, swap_ab)
if CACHE_KEY not in kernel_dict:
    compiled_gemm = cute.compile(gemm, ...)
    kernel_dict[CACHE_KEY] = compiled_gemm

# 2. Launch kernel
compiled_gemm(kernel_m, kernel_n, real_k, ...)
```

---

### Subsequent Invocations (Inference Mode)

#### Step 11: Cache lookup on inference

**Location:** [`tensorrt_llm/_torch/autotuner.py:574-589`](../tensorrt_llm/_torch/autotuner.py#L574-L589)

```python
# Line 574-589 (from choose_one)
# Early return if not tuning mode
if not self.is_tuning_mode:
    # Search cache for matching config
    is_cache_hit, best_runner_id, best_tactic, min_time = \
        self.profiling_cache.search_cache(
            custom_op, runners, input_shapes, tuning_config
        )

    best_runner = runners[best_runner_id]

    # Log cache miss (shouldn't happen in production)
    if not is_cache_hit:
        logger.warning_once(
            f"[AutoTunner] Using fallback tactic, cache miss on "
            f"input shapes={input_shapes}",
            key=custom_op
        )

    return (best_runner, best_tactic)
```

**Cache search:**

**Location:** [`tensorrt_llm/_torch/autotuner.py:333-356`](../tensorrt_llm/_torch/autotuner.py#L333-L356)

```python
# Line 333-356
def search_cache(
    self,
    custom_op: str,
    runners: List[TunableRunner],
    input_shapes: Tuple[torch.Size],
    tuning_config: TuningConfig,
) -> Tuple[bool, int, int, float]:
    """Search cache for matching configuration.

    Returns:
        (is_cache_hit, runner_id, tactic, min_time)
    """
    # Try each runner
    for r in runners:
        # Build cache key
        cache_key = self.get_cache_key(
            custom_op, r, input_shapes, tuning_config
        )

        # Check if key exists in cache
        if cache_key in self.cache:
            return True, *self.cache[cache_key]
            # Returns: (True, runner_id, tactic, min_time)

    # Cache miss - return fallback
    return False, *self.fallback_entry()
    # Returns: (False, 0, -1, inf)
```

**Performance:**
- **Cache hit:** O(1) lookup, ~10 μs
- **Cache miss:** Returns fallback tactic (-1)

---

## Data Structures

### TuningConfig

**Location:** [`tensorrt_llm/_torch/autotuner.py:52-98`](../tensorrt_llm/_torch/autotuner.py#L52-L98)

```python
@dataclass(kw_only=True)
class TuningConfig:
    """Configuration for autotuning process."""

    dynamic_tensor_specs: Tuple[DynamicTensorSpec, ...] = ()
    constraint_specs: Tuple[ConstraintSpec, ...] = ()
    tune_max_num_tokens: int = None
```

### DynamicTensorSpec

**Location:** [`tensorrt_llm/_torch/autotuner.py:22-35`](../tensorrt_llm/_torch/autotuner.py#L22-L35)

```python
@dataclass(slots=True, unsafe_hash=True)
class DynamicTensorSpec:
    """Specification for dynamic tensor dimension."""

    input_idx: int                           # Which input tensor (0, 1, 2, ...)
    dim_idx: int                             # Which dimension (0=batch, 1=hidden, ...)
    gen_tuning_buckets: Union[Tuple[int], Callable] = ()  # Bucket values or generator
    map_to_tuning_buckets: Callable = lambda x: x         # Mapping function
```

**Example:**
```python
DynamicTensorSpec(
    input_idx=0,           # First input tensor
    dim_idx=0,             # Batch/token dimension
    gen_tuning_buckets=get_last_power_of_2_num_tokens_buckets,  # Function
    map_to_tuning_buckets=last_positive_power_of_2  # Round to power of 2
)

# If current input has 1000 tokens:
# map_to_tuning_buckets(1000) = 1024
# gen_tuning_buckets(1024) = [128, 256, 512, 1024, 2048, 4096]
```

### ConstraintSpec

**Location:** [`tensorrt_llm/_torch/autotuner.py:38-49`](../tensorrt_llm/_torch/autotuner.py#L38-L49)

```python
@dataclass(slots=True, unsafe_hash=True)
class ConstraintSpec:
    """Specification for constrained tensor dimension."""

    input_idx: int           # Which input tensor
    dim_idx: int             # Which dimension
    infer_shape: Callable    # Function to compute shape from other dims
```

**Example:**
```python
ConstraintSpec(
    input_idx=2,  # Scale tensor
    dim_idx=0,    # First dimension
    infer_shape=lambda shapes: shapes[0][0] * (shapes[0][1] // 16)
    # Scale size = num_tokens * (hidden_size // 16)
)
```

### OptimizationProfile

**Location:** [`tensorrt_llm/_torch/autotuner.py:123-139`](../tensorrt_llm/_torch/autotuner.py#L123-L139)

```python
@dataclass
class OptimizationProfile:
    """Ranges for all tensors and dimensions."""

    shapes: List[List[Dim]]  # Dim = Union[StaticDim, DynamicDim]

    def get_opt_shapes(self):
        """Extract opt values from all dimensions."""
        opt_shapes = []
        for t in self.shapes:
            opt_shapes.append(tuple([d._opt() for d in t]))
        return tuple(opt_shapes)
```

**Example:**
```python
OptimizationProfile(
    shapes=[
        [DynamicDim(min=1024, opt=1024, max=2048), StaticDim(val=4096)],
        [StaticDim(val=14336), StaticDim(val=4096)],
        [DynamicDim(min=1024, opt=1024, max=2048), StaticDim(val=256)],
        [StaticDim(val=14336), StaticDim(val=256)],
    ]
)

# get_opt_shapes() returns:
((1024, 4096), (14336, 4096), (1024, 256), (14336, 256))
```

---

## Cache Mechanism

### Cache Structure

```python
# AutoTunerProfilingCache.cache is a dict:
{
    cache_key: (runner_id, tactic, min_time),
    ...
}

# Example entry:
{
    (
        "trtllm::cute_dsl_nvfp4_gemm_blackwell",
        "CuteDSLNVFP4BlackwellLinear",
        -1234567890,
        ((1024, 4096), (14336, 4096), (1024, -1), (14336, -1))
    ): (0, (mma_256_128, cluster_2_2, False), 0.245)
}
```

### Serialization

**Location:** [`tensorrt_llm/_torch/autotuner.py:380-508`](../tensorrt_llm/_torch/autotuner.py#L380-L508)

```python
# Line 380-405: save_cache()
def save_cache(self, file_path: Union[str, Path]) -> None:
    """Save cache to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        serializable_cache = self._serialize_cache_to_json()
        with open(file_path, 'w') as f:
            json.dump(serializable_cache, f, indent=2, default=str)
        logger.info(
            f"[AutoTuner] Successfully saved cache to {file_path}"
        )
    except Exception as e:
        logger.error(f"[AutoTuner] Failed to save cache: {e}")
        raise

# Line 436-468: _serialize_cache_to_json()
def _serialize_cache_to_json(self) -> Dict[str, Any]:
    """Convert cache to JSON-serializable format."""
    serializable_cache = {
        "metadata": {
            "lib_version": self.lib_version,
            "creation_timestamp": self.creation_timestamp,
            "device_name": self.device_name,
            "device_capability": self.device_capability,
        },
        "cache_data": {},
    }

    for key, value in self.cache.items():
        # Convert tuple key to string
        key_str = str(key)

        runner_id, tactic, min_time = value

        serializable_cache["cache_data"][key_str] = {
            "runner_id": runner_id,
            "tactic": tactic,
            "min_time": min_time,
        }

    return serializable_cache
```

**Example JSON:**
```json
{
  "metadata": {
    "lib_version": "0.14.0",
    "creation_timestamp": 1704067200.0,
    "device_name": "NVIDIA H100 80GB HBM3",
    "device_capability": [9, 0]
  },
  "cache_data": {
    "('trtllm::cute_dsl_nvfp4_gemm_blackwell', 'CuteDSLNVFP4BlackwellLinear', -1234567890, ((1024, 4096), (14336, 4096), (1024, -1), (14336, -1)))": {
      "runner_id": 0,
      "tactic": "(mma_256_128, cluster_2_2, False)",
      "min_time": 0.245123
    }
  }
}
```

---

## Example Walkthrough

### Complete FP4 GEMM Autotuning Example

```python
# ========== SETUP PHASE ==========
# User code
with autotune(tune_mode=True, cache_path="./cache/fp4_gemm.json"):
    # Autotune context: is_tuning_mode = True

    # Create input tensors
    input = torch.randn(1000, 4096, dtype=torch.uint8, device='cuda')    # FP4 packed
    weight = torch.randn(14336, 4096, dtype=torch.uint8, device='cuda')  # FP4 packed
    input_scale = torch.randn(1000, 256, dtype=torch.float8_e4m3fn, device='cuda')
    weight_scale = torch.randn(14336, 256, dtype=torch.float8_e4m3fn, device='cuda')

    # Call custom op
    output = torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(
        input, weight, input_scale, weight_scale, 1.0, torch.bfloat16
    )

# ========== CUSTOM OP EXECUTION ==========
# In cute_dsl_nvfp4_gemm_blackwell():

# 1. Get AutoTuner singleton
tuner = AutoTuner.get()  # is_tuning_mode = True

# 2. Create runner
runner = CuteDSLNVFP4BlackwellLinear(alpha=1.0, output_dtype=torch.bfloat16)

# 3. Choose best tactic
_, best_tactic = tuner.choose_one(
    "trtllm::cute_dsl_nvfp4_gemm_blackwell",
    [runner],
    CuteDSLNVFP4BlackwellLinear.tuning_config,
    [input, weight, input_scale, weight_scale],
)

# ========== AUTOTUNER.CHOOSE_ONE() ==========
# Input shapes: ((1000, 4096), (14336, 4096), (1000, 256), (14336, 256))

# Generate optimization profiles
# Dynamic spec: input_idx=0, dim_idx=0, map=last_power_of_2
# 1000 → 1024 (bucket)
# Profiles generated: [128, 256, 512, 1024, 2048, 4096] = 6 profiles

# ========== PROFILE LOOP ==========
# Profile 1: m=128
profiles = [
    OptimizationProfile(shapes=[
        [DynamicDim(128, 128, 256), StaticDim(4096)],
        [StaticDim(14336), StaticDim(4096)],
        [DynamicDim(128, 128, 256), StaticDim(256)],
        [StaticDim(14336), StaticDim(256)],
    ]),
    # ... 5 more profiles
]

for profile in profiles:  # 6 iterations
    # Prepare tensors with profile shapes
    tensors = _prepare_input_tensors(profile, inputs)
    # Creates: [torch.zeros(128, 4096), torch.zeros(14336, 4096), ...]

    # Check cache (first time = miss)
    is_cache_hit = profiling_cache.search_cache(...)  # False

    # Profile all runners
    best_runner_id, best_tactic, min_time = _profile_runners(...)

    # ========== PROFILE_RUNNERS() ==========
    # Get valid tactics from runner
    valid_tactics = runner.get_valid_tactics(tensors, profile)
    # Returns: ~50-100 tactics after filtering
    # Example tactics:
    # [
    #   ((256, 128), (1, 1), False),
    #   ((256, 128), (2, 2), False),
    #   ((128, 256), (1, 2), True),
    #   ...
    # ]

    min_time = inf
    best_tactic = None

    for tactic in valid_tactics:  # ~50-100 iterations
        # ========== PROFILE_SINGLE_KERNEL() ==========
        stream = torch.cuda.current_stream()

        # Warmup: 3 iterations
        for _ in range(3):
            runner(tensors, tactic=tactic)
        stream.synchronize()

        # Delay injection
        delay_kernel(1000, stream)  # 1ms delay

        # Timing: 10 iterations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record(stream=stream)
        for _ in range(10):
            runner(tensors, tactic=tactic)
        end.record(stream=stream)
        stream.synchronize()

        time_measured = start.elapsed_time(end) / 10  # Average
        # Example: 0.245 ms

        logger.debug(
            f"[Autotuner] Profiled tactic={tactic}: {time_measured:.6f}ms"
        )

        # Update best
        if time_measured < min_time:
            min_time = time_measured
            best_tactic = tactic

    # ========== CACHE RESULT ==========
    cache_key = (
        "trtllm::cute_dsl_nvfp4_gemm_blackwell",
        "CuteDSLNVFP4BlackwellLinear",
        hash(runner),
        ((128, 4096), (14336, 4096), (128, -1), (14336, -1))
    )
    profiling_cache[cache_key] = (0, best_tactic, min_time)
    # Example: (0, ((256, 128), (2, 2), False), 0.245)

# After all profiles: 6 cache entries created

# ========== FINAL SELECTION ==========
# Get best for current input shape (1000 → 1024)
_, runner_id, tactic, _ = profiling_cache.search_cache(
    "trtllm::cute_dsl_nvfp4_gemm_blackwell",
    [runner],
    ((1000, 4096), ...),  # Maps to 1024 bucket
    tuning_config
)
# Returns: (0, ((256, 128), (2, 2), False))

# ========== KERNEL EXECUTION ==========
# In runner.forward(inputs, tactic=(mma_256_128, cluster_2_2, False))
CACHE_KEY = (16, (256, 128), (2, 2), False)

if CACHE_KEY not in kernel_dict:
    # Compile kernel using CuTe DSL
    gemm = Sm100BlockScaledPersistentDenseGemmKernelWrapper(
        sf_vec_size=16,
        mma_tiler_mn=(256, 128),
        cluster_shape_mn=(2, 2)
    )
    compiled_gemm = cute.compile(gemm, m=1000, n=14336, k=8192, ...)
    kernel_dict[CACHE_KEY] = compiled_gemm

# Launch compiled kernel
compiled_gemm(1000, 14336, 8192, ...)

# ========== CLEANUP ==========
# Context exit: save cache to disk
# File: ./cache/fp4_gemm.rank0.json
# Contains: 6 cache entries (one per profile bucket)
```

**Summary:**
- **Profiles generated:** 6 (one per token bucket)
- **Tactics per profile:** ~50-100 (after filtering)
- **Total profiling runs:** 6 × 75 = **~450 kernel launches**
- **Warmup per tactic:** 3 iterations
- **Timing per tactic:** 10 iterations
- **Total kernel launches:** 450 × (3 + 10) = **~5,850 kernel launches**
- **Cache entries:** 6
- **Tuning time:** ~30-60 seconds (depending on hardware)

**Inference (subsequent runs):**
- **Cache lookup:** O(1), ~10 μs
- **No profiling**
- **Direct kernel execution**

---

## Summary

### Key Takeaways

1. **Singleton Pattern:** AutoTuner is a global singleton accessed via `AutoTuner.get()`

2. **Two Modes:**
   - **Tuning mode:** Profile all tactics, cache results
   - **Inference mode:** Direct cache lookup, no profiling

3. **Profile Generation:** Cartesian product of dynamic dimension buckets
   - Example: 6 token buckets = 6 profiles to tune

4. **Tactic Selection:** Per profile, benchmark all valid (runner, tactic) pairs
   - Example: 75 tactics × 6 profiles = 450 combinations

5. **Accurate Timing:**
   - Warmup iterations (3)
   - Delay injection (1ms)
   - Multiple timing iterations (10)
   - CUDA events for GPU-side timing

6. **Caching:**
   - Cache key: `(op_name, runner_class, runner_hash, bucketed_shapes)`
   - Persistent: Saved to JSON file
   - Rank-specific: `cache.rank0.json`, `cache.rank1.json`, etc.

7. **Fallback:** If cache miss in inference, uses tactic `-1` (fallback implementation)

### Performance Impact

**Tuning overhead:**
- First run: 30-60 seconds per operation
- Amortized: 0 (cached)

**Inference speedup:**
- Optimal kernel selection
- 2-10x faster than naive implementation
- Consistent performance across shapes

### Best Practices

1. **Always cache:** Use `autotune(cache_path="...")` to persist results
2. **Tune comprehensively:** Use representative input shapes
3. **Handle failures:** Implement fallback tactic (`-1`)
4. **Monitor stats:** Check `AutoTuner.stats` for failures
5. **Profile-aware:** Design ops with bucketing in mind

This completes the comprehensive autotuning trace!
