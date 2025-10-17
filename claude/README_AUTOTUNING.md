# Autotuning Documentation

This file provides a guide to the autotuning documentation in this directory.

## Quick Links

### Main Documentation

📘 **[Complete Autotuning Trace](autotuning_detailed_trace.md)**
- **Comprehensive line-by-line trace** of the entire autotuning system
- **Every function call documented** with source links and line numbers
- **Complete data flow** from Python → C++ → CUDA
- **Copious annotated code snippets** at each step
- **Detailed example walkthrough** with actual numbers

### Related Documentation

- **[Fused MoE Call Stack Trace](fused_moe_callstack_trace.md)** - Shows how MoE uses autotuning
- **[Detailed Kernel Trace](detailed_kernel_trace.md)** - Kernel implementations that get autotuned
- **[Visual Guide & Reference](visual_guide_and_reference.md)** - Practical usage and debugging

## What's Covered

The autotuning documentation provides exhaustive coverage of:

### 1. Architecture Overview
- Complete system architecture diagram
- Component relationships
- Data flow visualization

### 2. Initialization & Setup
```python
with autotune(tune_mode=True, cache_path="cache.json"):
    output = custom_op(input, weight, ...)
```
- Context manager setup
- AutoTuner singleton initialization
- Cache loading/saving

### 3. First Invocation (Tuning Mode)
- Custom op implementation
- Runner definition with `get_valid_tactics()`
- `AutoTuner.choose_one()` detailed flow
- Profile generation via cartesian product
- Runner profiling loop
- Single kernel profiling with accurate timing
- Cache result storage

### 4. Subsequent Invocations (Inference Mode)
- Cache lookup mechanism
- Fallback handling
- Near-zero overhead

### 5. Data Structures
- `TuningConfig` - Dynamic dimension configuration
- `DynamicTensorSpec` - Specification for dynamic dims
- `ConstraintSpec` - Dependent dimension constraints
- `OptimizationProfile` - Dimension ranges
- `AutoTunerProfilingCache` - Result storage

### 6. Complete Example Walkthrough
- FP4 GEMM autotuning from start to finish
- Actual numbers: 6 profiles × 75 tactics = 450 combinations
- Timing breakdown: ~5,850 total kernel launches
- Cache generation and reuse

## Key Insights

### Performance Numbers

**Tuning Phase:**
- **First run:** 30-60 seconds per operation
- **Kernel launches:** ~5,850 for comprehensive tuning
- **Profiles:** 6-10 typical (power-of-2 token buckets)
- **Tactics per profile:** 50-100 (after hardware filtering)

**Inference Phase:**
- **Cache lookup:** O(1), ~10 μs overhead
- **Speedup:** 2-10x vs naive implementation
- **Overhead:** Effectively zero after caching

### Critical Implementation Details

1. **Accurate Timing:**
   ```python
   # Warmup: 3 iterations
   for _ in range(3):
       runner(inputs, tactic=tactic)

   # Delay injection: 1ms to reduce host overhead
   delay_kernel(1000, stream)

   # Timing: 10 iterations with CUDA events
   start.record(stream)
   for _ in range(10):
       runner(inputs, tactic=tactic)
   end.record(stream)

   avg_time = start.elapsed_time(end) / 10
   ```

2. **Cache Key Structure:**
   ```python
   cache_key = (
       custom_op_name,           # "trtllm::my_op"
       runner_class_name,        # "MyRunner"
       hash(runner_attributes),  # -1234567890
       bucketed_shapes,          # ((1024, 4096), ...)
   )
   ```

3. **Profile Generation:**
   - Cartesian product of dynamic dimension buckets
   - Constraint specs for dependent dimensions
   - Power-of-2 bucketing for token dimensions

## File Organization

```
claude/
├── README.md                          # Main navigation
├── README_AUTOTUNING.md              # This file
├── autotuning_detailed_trace.md      # Complete autotuning trace ⭐
├── fused_moe_callstack_trace.md      # MoE architecture
├── detailed_kernel_trace.md          # Kernel implementations
└── visual_guide_and_reference.md     # Practical guide
```

## Usage Examples

### Basic Autotuning

```python
from tensorrt_llm._torch.autotuner import autotune

# Enable tuning and cache results
with autotune(tune_mode=True, cache_path="./cache/my_op.json"):
    # First run: profiles all tactics
    output = torch.ops.trtllm.my_op(input, weight)

# Subsequent runs: uses cached tactics
output = torch.ops.trtllm.my_op(input, weight)  # Fast!
```

### Implementing a Tunable Op

```python
from tensorrt_llm._torch.autotuner import (
    AutoTuner, TunableRunner, TuningConfig, DynamicTensorSpec
)

class MyRunner(TunableRunner):
    # Define tuning config
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=0,
                dim_idx=0,
                gen_tuning_buckets=get_power_of_2_buckets,
                map_to_tuning_buckets=next_power_of_2
            ),
        ),
    )

    def get_valid_tactics(self, inputs, profile, **kwargs):
        """Return list of valid tactic configurations."""
        # Generate candidate tactics based on input shapes
        tactics = []
        for block_size in [128, 256, 512]:
            for num_warps in [4, 8, 16]:
                if self.can_implement(block_size, num_warps, inputs):
                    tactics.append((block_size, num_warps))
        return tactics

    def forward(self, inputs, tactic=-1, **kwargs):
        """Execute kernel with given tactic."""
        if tactic == -1:
            # Fallback implementation
            block_size, num_warps = 128, 4
        else:
            block_size, num_warps = tactic

        # Launch kernel with configuration
        return self.launch_kernel(inputs, block_size, num_warps)

@torch.library.custom_op("trtllm::my_op", mutates_args=())
def my_op(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    tuner = AutoTuner.get()
    runner = MyRunner()

    _, best_tactic = tuner.choose_one(
        "trtllm::my_op",
        [runner],
        MyRunner.tuning_config,
        [input, weight],
    )

    return runner(inputs=[input, weight], tactic=best_tactic)
```

## Debugging

### Enable Detailed Logging

```python
import os
os.environ['TLLM_LOG_LEVEL'] = 'DEBUG'

# Shows detailed profiling results:
# [Autotuner] Profiled runner=MyRunner, tactic=(128, 4),
#             shapes=[(1024, 4096)]: 0.245123ms
```

### Check Statistics

```python
tuner = AutoTuner.get()
print(tuner.stats)

# Output:
# Cache misses: 0
# Tuned operations:
#   trtllm::my_op:
#     - Total configs tried: 6
#     - Successful configs: 6
#     - Failed profiling count: 0
#     - Success rate: 100.0%
```

### Inspect Cache

```python
tuner.print_profiling_cache()

# Output:
# [Autotuner] Cache contents:
# ('trtllm::my_op', 'MyRunner', -1234567890, ((1024, 4096),)):
#   (runner_id=0, tactic=(128, 4), min_time=0.245)
```

## Advanced Topics

### Multi-GPU Tuning

```python
# Rank-specific caching
with autotune(cache_path="cache.json", rank=torch.distributed.get_rank()):
    # Creates cache.rank0.json, cache.rank1.json, etc.
    output = torch.ops.trtllm.my_op(input, weight)
```

### Custom Bucket Functions

```python
def my_bucket_function(max_tokens):
    """Generate custom bucket sizes."""
    return [64, 128, 192, 256, 384, 512]

DynamicTensorSpec(
    input_idx=0,
    dim_idx=0,
    gen_tuning_buckets=my_bucket_function,
    map_to_tuning_buckets=lambda x: min([b for b in [64, 128, 192, ...] if b >= x])
)
```

### Constraint Specifications

```python
# Scale dimension depends on input dimension
ConstraintSpec(
    input_idx=1,  # Scale tensor
    dim_idx=0,    # First dimension
    infer_shape=lambda shapes: shapes[0][0] * (shapes[0][1] // 16)
    # scale_size = num_tokens * (hidden_size // 16)
)
```

## Performance Tips

1. **Tune comprehensively:** Include all expected input shapes
2. **Use power-of-2 buckets:** Better memory alignment
3. **Cache aggressively:** Persistent storage across runs
4. **Implement fallback:** Always provide tactic `-1`
5. **Handle failures gracefully:** Log but don't crash
6. **Monitor statistics:** Check for unexpected cache misses

## Common Patterns

### Pattern 1: Token Dimension Tuning
```python
# Tune on batch/token dimension (common for LLM ops)
DynamicTensorSpec(
    input_idx=0,
    dim_idx=0,
    gen_tuning_buckets=get_power_of_2_num_tokens_buckets,
    map_to_tuning_buckets=last_positive_power_of_2
)
```

### Pattern 2: Multiple Runners
```python
# Try multiple implementation strategies
runners = [
    CUDAKernelRunner(),
    TritonKernelRunner(),
    CutlassKernelRunner(),
]

_, best_tactic = tuner.choose_one(
    "trtllm::my_op",
    runners,
    config,
    inputs,
)
# Autotuner selects best across all runners
```

### Pattern 3: Preparation Phase
```python
def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
    if do_preparation:
        # One-time setup excluded from profiling
        self.compile_kernels()
        return None

    # Normal execution
    return self.launch_kernel(inputs, tactic)
```

## References

### Source Files

- **AutoTuner:** [`tensorrt_llm/_torch/autotuner.py`](../tensorrt_llm/_torch/autotuner.py)
- **Tests:** [`tests/unittest/_torch/misc/test_autotuner.py`](../tests/unittest/_torch/misc/test_autotuner.py)
- **Custom Ops:** [`tensorrt_llm/_torch/custom_ops/`](../tensorrt_llm/_torch/custom_ops/)

### Key Functions

| Function | Line | Description |
|----------|------|-------------|
| `autotune()` | [autotuner.py:210](../tensorrt_llm/_torch/autotuner.py#L210) | Context manager |
| `choose_one()` | [autotuner.py:542](../tensorrt_llm/_torch/autotuner.py#L542) | Main selection logic |
| `_optimization_profiles()` | [autotuner.py:769](../tensorrt_llm/_torch/autotuner.py#L769) | Profile generation |
| `_profile_runners()` | [autotuner.py:651](../tensorrt_llm/_torch/autotuner.py#L651) | Runner profiling |
| `_profile_single_kernel()` | [autotuner.py:719](../tensorrt_llm/_torch/autotuner.py#L719) | Kernel timing |

## Next Steps

1. **Read:** [autotuning_detailed_trace.md](autotuning_detailed_trace.md) for complete details
2. **Implement:** Create your own `TunableRunner` subclass
3. **Tune:** Run with `autotune(tune_mode=True)`
4. **Deploy:** Use cached results in production
5. **Monitor:** Check statistics and cache hit rates

---

For questions or issues, refer to the detailed trace document or TensorRT-LLM documentation.
