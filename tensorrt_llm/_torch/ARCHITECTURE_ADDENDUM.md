# ARCHITECTURE.md Addendum: Clarifying Backend Modes

> **Important Update**: This document clarifies the backend modes described in ARCHITECTURE.md based on actual usage patterns found in the codebase.

## The Three Backend Modes (Corrected)

The original ARCHITECTURE.md incorrectly presented "Compilation" and "Auto Deploy" as two separate modes. In reality, TensorRT-LLM has **three distinct backend modes**:

### 1. TensorRT Backend (Default)

```python
from tensorrt_llm import LLM

llm = LLM(model_path)  # No backend= argument
```

- **What it is**: Traditional TRT-LLM with ahead-of-time engine compilation
- **When to use**: Production deployments, maximum performance after build
- **Pros**: Fastest inference, mature feature set
- **Cons**: Requires build step (`trtllm-build`), less flexible

### 2. PyTorch Backend (`backend="pytorch"`)

```python
from tensorrt_llm import LLM

# Without torch.compile
llm = LLM(model_path, backend="pytorch")

# WITH torch.compile (this is what ARCHITECTURE.md calls "Compilation Mode")
from tensorrt_llm.llmapi import TorchCompileConfig

llm = LLM(
    model_path,
    backend="pytorch",
    torch_compile_config=TorchCompileConfig(
        enable_fullgraph=True,
        enable_inductor=True,
        enable_piecewise_cuda_graph=True,
    )
)
```

- **What it is**: Pure PyTorch execution (eager or compiled)
- **When to use**: Debugging, models not yet supported by TRT builder, rapid iteration
- **torch.compile variant**: JIT compilation for production-level performance
- **Pros**: No build step, flexible, supports more models
- **Cons**: Slower than TRT (without compile), first-call compilation overhead (with compile)

**Key insight**: The "Compilation Mode" in ARCHITECTURE.md is actually **PyTorch backend + `TorchCompileConfig`**, not a separate mode!

### 3. AutoDeploy Backend (`backend="_autodeploy"`)

```python
from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM

llm = AutoDeployLLM(model=model_path)
```

- **What it is**: PyTorch with automatic graph transformations (sharding, KV-cache, etc.)
- **When to use**: Day-0 support for new HuggingFace models, prototype deployments
- **Status**: Prototype (subject to breaking changes)
- **Pros**: Zero manual implementation, automatic optimizations
- **Cons**: Experimental, may have rough edges

## How Backend Selection Works

**Source**: [llmapi/llm.py:134-151](../llmapi/llm.py)

```python
# From tensorrt_llm/llmapi/llm.py
class BaseLLM:
    def __init__(self, model, tokenizer=None, **kwargs):
        backend = kwargs.get('backend', None)

        if backend == "pytorch":
            logger.info("Using LLM with PyTorch backend")
            llm_args_cls = TorchLlmArgs
        elif backend == '_autodeploy':
            logger.info("Using LLM with AutoDeploy backend")
            from .._torch.auto_deploy.llm_args import LlmArgs as AutoDeployLlmArgs
            llm_args_cls = AutoDeployLlmArgs
        else:
            logger.info("Using LLM with TensorRT backend")
            llm_args_cls = TrtLlmArgs
```

## TorchCompileConfig: Enabling torch.compile

**Source**: [llmapi/llm_args.py](../llmapi/llm_args.py)

```python
class TorchCompileConfig(StrictBaseModel):
    """Configuration for torch.compile within PyTorch backend."""

    enable_fullgraph: bool = Field(default=True,
        description="Enable full graph compilation")

    enable_inductor: bool = Field(default=False,
        description="Enable inductor backend (Triton)")

    enable_piecewise_cuda_graph: bool = Field(default=False,
        description="Enable piecewise CUDA graph capture")

    capture_num_tokens: Optional[List[int]] = Field(default=None,
        description="Token counts to profile for CUDA graphs")

    enable_userbuffers: bool = Field(default=True,
        description="Use NCCL user buffers for zero-copy")

    max_num_streams: int = Field(default=1,
        description="Number of CUDA streams for parallelism")
```

This config is passed to the internal `Backend` class in `tensorrt_llm/_torch/compilation/backend.py`.

## Documentation & Examples

### Official Documentation

1. **AutoDeploy**: [docs/source/torch/auto_deploy/auto-deploy.md](../../docs/source/torch/auto_deploy/auto-deploy.md)
   - Well documented with examples
   - Prototype status clearly indicated

2. **PyTorch Backend**: Documented in LLM API
   - `backend` parameter description
   - `TorchCompileConfig` docstrings
   - No dedicated tutorial (yet)

3. **torch.compile integration**: Underdocumented
   - Scattered across API docs
   - No step-by-step guide

### Examples

#### 1. PyTorch Backend with torch.compile

**File**: [examples/llm-api/quickstart_advanced.py](../../examples/llm-api/quickstart_advanced.py)

```bash
python quickstart_advanced.py \
    --model_dir meta-llama/Llama-2-7b-hf \
    --use_torch_compile \
    --use_piecewise_cuda_graph \
    --tp_size 4
```

**Key flags**:
- `--use_torch_compile`: Enables `TorchCompileConfig(enable_fullgraph=True, enable_inductor=True)`
- `--use_piecewise_cuda_graph`: Adds CUDA graph capture

#### 2. AutoDeploy Backend

**File**: [examples/auto_deploy/build_and_run_ad.py](../../examples/auto_deploy/build_and_run_ad.py)

```bash
python build_and_run_ad.py \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --args.compile-backend torch-opt \
    --args.world-size 2
```

### Tests

1. **PyTorch Backend**: [tests/integration/defs/accuracy/test_llm_api_pytorch.py](../../tests/integration/defs/accuracy/test_llm_api_pytorch.py)
   ```python
   # Line 93-108
   @parametrize_with_ids("torch_compile", [False, True])
   def test_bfloat16(self, attn_backend, torch_compile):
       torch_compile_config = TorchCompileConfig(
           enable_fullgraph=True,
           enable_piecewise_cuda_graph=True,
           capture_num_tokens=[2048, 8192],
           max_num_streams=3
       ) if torch_compile else None

       with LLM(self.MODEL_PATH,
                torch_compile_config=torch_compile_config) as llm:
           task = GSM8K(self.MODEL_NAME)
           task.evaluate(llm)
   ```

2. **AutoDeploy**: [tests/integration/defs/accuracy/test_llm_api_autodeploy.py](../../tests/integration/defs/accuracy/test_llm_api_autodeploy.py)
   ```python
   # Line 73-76
   with AutoDeployLLM(model=self.MODEL_PATH,
                      tokenizer=self.MODEL_PATH,
                      world_size=world_size,
                      compile_backend="torch-opt") as llm:
       task.evaluate(llm)
   ```

## Corrected Comparison Table

| Aspect | TensorRT | PyTorch (Eager) | PyTorch (Compiled) | AutoDeploy |
|--------|----------|----------------|-------------------|-----------|
| **API** | `LLM(model)` | `LLM(model, backend="pytorch")` | `+ torch_compile_config=...` | `AutoDeployLLM(model)` |
| **Runtime** | C++ TRT | PyTorch | PyTorch + FX | PyTorch + Transforms |
| **Build Step** | Yes | No | No | No |
| **Optimization** | Static engine | None | Graph compile | Graph transforms |
| **First Call** | Fast | Fast | Slow (~30s) | Medium (~5s export) |
| **Subsequent** | Fastest | Medium | Fast | Fast |
| **CUDA Graphs** | Built-in | Manual | Piecewise auto | Limited |
| **Use Case** | Production | Debug/Dev | High-perf PyTorch | Day-0 models |
| **Status** | Stable | Stable | Stable | Prototype |

## Recommendation for Users

1. **Start with PyTorch backend** (`backend="pytorch"`):
   - Fast iteration, easy debugging
   - Supports most models out of the box

2. **Enable torch.compile for production** (`torch_compile_config=...`):
   - When you need TRT-level performance
   - Multi-GPU with user buffers
   - Custom fusion patterns

3. **Use AutoDeploy for new models**:
   - When model not yet in TRT-LLM
   - Rapid prototyping
   - Automatic optimizations

4. **Graduate to TensorRT backend**:
   - When model matures and has full TRT support
   - For absolute maximum performance
   - When build step is acceptable

## What ARCHITECTURE.md Got Wrong

1. **"Compilation Mode" is not a separate mode**: It's PyTorch backend + `TorchCompileConfig`
2. **"Auto Deploy" is not an alternative to Compilation**: It's a third backend option
3. **Missing the TensorRT backend**: The original doc focused only on PyTorch-based modes
4. **API usage**: The doc showed internal APIs (`Backend` class) instead of user-facing `LLM` API

## Integration with ARCHITECTURE.md

The rest of ARCHITECTURE.md (sections 3-7) remains accurate:
- Backend architecture and torch.compile integration (Section 3)
- Custom ops dispatch paths (Sections 4-5)
- AutoTuner implementation (Section 6)
- Code snippets (Section 7 - though examples should use `LLM` API)

The core technical content about how `torch.compile` works, custom ops registration, and autotuning is all correct. Only the framing of "two modes" needs updating.

## Conclusion

Think of it this way:
- **TensorRT backend**: The traditional way (build → run fast)
- **PyTorch backend**: The flexible way (run now, optionally compile)
  - Without `torch_compile_config`: Eager mode
  - With `torch_compile_config`: Compiled mode (what ARCHITECTURE.md calls "Compilation")
- **AutoDeploy backend**: The experimental way (automatic everything)

All three backends can coexist and serve different use cases!

---

**Last Updated**: 2025-10-13
**Related**: [ARCHITECTURE.md](ARCHITECTURE.md)
