# TensorRT-LLM Backend Comparison: Deep Dive

> Comprehensive analysis of the three execution backends in TensorRT-LLM: TensorRT, PyTorch, and AutoDeploy

## Table of Contents

1. [Quick Comparison](#quick-comparison)
2. [TensorRT Backend](#tensorrt-backend)
3. [PyTorch Backend](#pytorch-backend)
4. [AutoDeploy Backend](#autodeploy-backend)
5. [Side-by-Side Comparison](#side-by-side-comparison)
6. [When to Use Each Backend](#when-to-use-each-backend)
7. [Documentation Links](#documentation-links)

---

## Quick Comparison

| Aspect | TensorRT Backend | PyTorch Backend | AutoDeploy Backend |
|--------|------------------|-----------------|-------------------|
| **API** | `LLM(model)` | `LLM(model, backend="pytorch")` | `AutoDeployLLM(model)` |
| **Execution** | TensorRT engine (C++) | PyTorch eager/compiled | PyTorch + graph transforms |
| **Model Source** | TRT-LLM models | TRT-LLM _torch models | HuggingFace models |
| **Build Step** | Yes (trtllm-build) | No | No |
| **Model Format** | Custom layers | PyTorch nn.Module | Transformers models |
| **Attention** | TRT plugin | Backend-agnostic | Pattern-matched SDPA |
| **Custom Ops** | TRT plugins (C++) | torch.ops.trtllm.* | torch.ops.auto_deploy.* |
| **Quantization** | QuantMode + plugins | Module-level FP8/NVFP4 | Transform-based |
| **TP Sharding** | Mapping-based | TensorParallelMode | Graph sharding |
| **KV Cache** | Plugin-managed | Explicit tensors | Graph insertion |
| **Fusion** | Plugin-level | AllReduce + Norm | Pattern-based |
| **Status** | Production | Production | Prototype |

---

## TensorRT Backend

### Overview

The TensorRT backend uses **graph-based compilation** via TensorRT engine. Models are defined using custom layers that compile to TensorRT plugins.

### Model Structure

**Location**: [tensorrt_llm/models/llama/model.py](../models/llama/model.py)

#### Key Imports (Lines 25-27)
```python
from ...layers import (MOE, Attention, AttentionMaskType, ColumnLinear,
                       Embedding, FusedGatedMLP, GatedMLP,
                       PositionEmbeddingType, RmsNorm)
```

### Layer Implementations

#### 1. Attention Layer

**File**: [tensorrt_llm/layers/attention.py](../layers/attention.py)

**Configuration** (Lines 62-82):
```python
self.attention = Attention(
    local_layer_idx=self.local_layer_idx,
    hidden_size=config.hidden_size,
    attention_head_size=config.head_size,
    num_attention_heads=config.num_attention_heads,
    num_kv_heads=config.num_key_value_heads,
    max_position_embeddings=config.max_position_embeddings,
    dtype=config.dtype,
    attention_mask_type=AttentionMaskType.causal,
    bias=config.attn_bias,
    position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
    rotary_embedding_base=config.rotary_base,
    rotary_embedding_scaling=config.rotary_scaling,
    tp_group=config.mapping.tp_group,
    tp_size=config.mapping.tp_size,
    tp_rank=config.mapping.tp_rank,
    q_scaling=1.0 / config.attention_multiplier,
    quant_mode=config.quant_mode,
    cp_group=config.mapping.cp_group,
    cp_size=config.mapping.cp_size,
    cp_rank=config.mapping.cp_rank
)
```

**Implementation Details**:
- **Type**: Plugin-based (compiles to `gpt_attention` or `bert_attention` plugin)
- **Position Encoding**: RoPE GPT-NeoX style
- **Attention Mask**: Causal
- **KV Cache**: Plugin-managed via `KeyValueCacheParams`
- **Parallelism**: Tensor parallel (TP) and context parallel (CP)

#### 2. MLP/FFN Layer

**File**: [tensorrt_llm/layers/mlp.py](../layers/mlp.py)

**Configuration** (Lines 86-102):
```python
ClsMLP = GatedMLP
mlp_kwargs = {}
if config.moe.has_moe():
    ClsMLP = MOE
    mlp_kwargs = {
        "moe_config": config.moe,
        "mapping": config.mapping,
    }

self.mlp = ClsMLP(
    hidden_size=config.hidden_size,
    ffn_hidden_size=mlp_hidden_size,
    hidden_act=config.hidden_act,
    dtype=config.dtype,
    bias=config.mlp_bias,
    tp_group=config.mapping.tp_group,
    tp_size=config.mapping.tp_size,
    quant_mode=config.quant_mode,
    **mlp_kwargs
)
```

**Types**:
- `GatedMLP`: Standard SwiGLU FFN
- `FusedGatedMLP`: Optimized fused version
- `MOE`: Mixture of Experts

**MOE Implementation** ([layers/moe.py](../layers/moe.py) Lines 143-197):
- Uses `_moe_plugin` function → compiles to TensorRT MOE plugin
- Expert routing with top-k selection
- Normalization modes: RENORMALIZE, SPARSE_MIXER, DEVICE_LIMITED

#### 3. Normalization Layer

**File**: [tensorrt_llm/layers/normalization.py](../layers/normalization.py)

**Configuration** (Lines 55-57, 104-106):
```python
self.input_layernorm = RmsNorm(
    normalized_shape=config.hidden_size,
    eps=config.norm_epsilon,
    dtype=config.dtype
)

self.post_layernorm = RmsNorm(
    normalized_shape=config.hidden_size,
    eps=config.norm_epsilon,
    dtype=config.dtype
)
```

**Implementation**: Compiles to RMSNorm TensorRT plugin

### Custom Ops/Plugins

**TensorRT Plugins** (referenced in functional.py):
- `gpt_attention` / `bert_attention`: Attention plugins
- `_moe_plugin`: MOE plugin
- Quantization plugins (FP8, INT8, INT4, NVFP4)
- Fusion plugins (AllReduce + Norm, etc.)

**Functional Ops** (Lines 27-33):
```python
from ..functional import (
    allgather, bert_attention, gpt_attention, matmul,
    embedding, softmax, layernorm, ...
)
```

### Fusion Patterns

#### AllReduce + Norm + Quantization Fusion

**File**: [models/llama/model.py](../models/llama/model.py) Lines 166-178

```python
reduce_fusion_op = AllReduceFusionOp.NONE
if default_net().plugin_config.reduce_fusion:
    if default_net().plugin_config.user_buffer:
        if self.config.quant_mode.has_fp8_qdq():
            reduce_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8
        elif self.config.quant_mode.has_nvfp4():
            reduce_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4
        else:
            assert False, "UB must enabled with fp8 or nvfp4 model"
    else:
        reduce_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM
```

**Fused Operation** (Lines 234-255):
```python
if default_net().plugin_config.norm_quant_fusion:
    hidden_states, mlp_input = fused_layernorm(
        residual,
        weight=self.post_layernorm.weight.value,
        gamma_trans=self.mlp.fc.activation_global_scaling_factor.value,
        ...
    )
```

**What's Fused**:
1. Residual addition (`hidden_states + residual`)
2. RMSNorm
3. Quantization (FP8 or NVFP4)
4. AllReduce (optional)

**Performance Impact**: 1.5-2x speedup on multi-GPU

### Model Modifications

**None** - Models are defined from scratch using TRT-LLM layers:
- Input: Model definition in Python using `tensorrt_llm.layers`
- Output: TensorRT engine file (optimized binary)

---

## PyTorch Backend

### Overview

The PyTorch backend uses **native PyTorch execution** (eager or torch.compile). Models use standard PyTorch `nn.Module` with custom ops for optimized kernels.

### Model Structure

**Location**: [tensorrt_llm/_torch/models/modeling_llama.py](_torch/models/modeling_llama.py)

#### Key Imports (Lines 33-42)
```python
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import create_moe
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
```

### Layer Implementations

#### 1. Attention Module

**File**: [tensorrt_llm/_torch/modules/attention.py](_torch/modules/attention.py)

**Class Definition** (Lines 111-132):
```python
class Attention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        bias: bool,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        rope_fusion: Optional[bool] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        q_scaling: float = 1.0,
        attention_chunk_size: Optional[int] = None,
    ):
        super().__init__()
        ...
```

**QKV Projection** (Lines 225-240):
```python
self.qkv_proj = Linear(
    hidden_size,
    (num_attention_heads + 2 * num_key_value_heads) * head_dim,
    bias=bias,
    dtype=dtype,
    tensor_parallel_mode=TensorParallelMode.COLUMN,
    weights_loading_config=WeightsLoadingConfig(
        weight_mode=WeightMode.FUSED_QKV_LINEAR,
    ),
    ...
)
```

**Output Projection** (Lines 245-258):
```python
self.o_proj = Linear(
    num_attention_heads * head_dim,
    hidden_size,
    bias=dense_bias if dense_bias is not None else bias,
    dtype=dtype,
    tensor_parallel_mode=TensorParallelMode.ROW,
    ...
)
```

**Attention Backend** (Lines 261-262):
```python
self.attn_backend = config.attn_backend
attn_cls = get_attention_backend(self.attn_backend)
```

**Supported Backends**:
- **TRTLLM**: Custom CUDA kernels via `torch.ops.trtllm`
- **FlashInfer**: FlashInfer library
- **Torch SDPA**: PyTorch's scaled_dot_product_attention
- **Triton**: Triton kernels

**Custom Op** (Lines 80-108):
```python
@torch.library.custom_op("trtllm::attn_custom_op_inplace",
                         mutates_args=("output",))
def attn_custom_op_inplace(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    attention_mask: str,
    mrope_rotary_cos_sin: Optional[torch.Tensor],
    mrope_position_deltas: Optional[torch.Tensor],
    ...
) -> None:
    """Custom attention op with in-place output."""
```

**RoPE Fusion** (Lines 277-288):
```python
if self.rope_fusion:
    # Fused RoPE inside attention kernel
    q, k, v = self.forward_impl(q=qkv, ...)
else:
    # Separate RoPE application
    q, k, v = self.apply_rope(q, k, v, position_ids)
    attn_output = self.forward_impl(q=q, k=k, v=v, ...)
```

#### 2. GatedMLP Module

**File**: [tensorrt_llm/_torch/modules/gated_mlp.py](_torch/modules/gated_mlp.py)

**Structure** (Lines 19-96):
```python
class GatedMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        bias: bool,
        activation: Callable = F.silu,
        ...
    ):
        super().__init__()

        # Fused gate and up projection
        self.gate_up_proj = Linear(
            self.hidden_size,
            self.intermediate_size * 2,
            bias=bias,
            dtype=dtype,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_GATE_UP_LINEAR
            ),
        )

        # Down projection
        self.down_proj = Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=bias,
            dtype=dtype,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )
```

**Forward Pass** (Lines 136-153):
```python
def forward(self, x, all_reduce_params=None):
    # Fused gate + up projection
    h1 = self.gate_up_proj(x)

    # SwiGLU activation (with optional FP8 quantization)
    h2 = self._apply_activation(h1)

    # Down projection + AllReduce
    output = self.down_proj(h2, all_reduce_params=all_reduce_params)

    return output
```

**Activation** (Lines 110-134):
```python
def _apply_activation(self, gate_up: torch.Tensor):
    if self.is_fp8_quant:
        # FP8 quantized SwiGLU
        return swiglu(gate_up, self.gate_up_proj.weight_dtype, ...)
    else:
        # Standard SwiGLU
        h1, h2 = gate_up.chunk(2, dim=-1)
        return self.activation(h1) * h2
```

**MOE Implementation** ([modeling_llama.py](../tensorrt_llm/_torch/models/modeling_llama.py) Lines 250-347):
```python
class Llama4MoE(nn.Module):
    def __init__(self, ...):
        # Shared expert
        self.shared_expert = GatedMLP(...)

        # Routed experts
        self.experts = create_moe(
            routing_method=Llama4RenormalizeMoeRoutingMethod(top_k),
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ...
        )

        # Router
        self.router = Linear(hidden_size, num_experts, ...)
```

#### 3. RMSNorm Module

**File**: [tensorrt_llm/_torch/modules/rms_norm.py](_torch/modules/rms_norm.py)

**Implementation** (Lines 26-115):
```python
class RMSNorm(nn.Module):
    def __init__(self, *, hidden_size: int, eps: float, ...):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        # FlashInfer optimized path
        if IS_FLASHINFER_AVAILABLE:
            if residual is not None:
                return flashinfer_fused_add_rmsnorm(
                    hidden_states,
                    residual,
                    self.weight,
                    self.variance_epsilon,
                )
            else:
                return flashinfer_rmsnorm(
                    hidden_states,
                    self.weight,
                    self.variance_epsilon,
                )

        # Fallback: PyTorch implementation
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)

            if residual is not None:
                hidden_states = hidden_states + residual.to(torch.float32)

            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

            return hidden_states.to(input_dtype)
```

**Custom Ops**:
- `flashinfer_fused_add_rmsnorm`: Fused residual + RMSNorm
- `flashinfer_rmsnorm`: RMSNorm only
- `flashinfer_gemma_rmsnorm`: Gemma-specific variant

### Custom Ops/Kernels

**Attention Ops**:
- `torch.ops.trtllm.attn_custom_op_inplace`: Custom attention kernel
- FlashInfer: `flash_attn_with_kvcache`, `batch_decode_with_padded_kv_cache`

**Linear Ops**:
- Standard `F.linear` (eager mode)
- FP8/NVFP4 quantized linear (via custom ops)

**Norm Ops**:
- `flashinfer_fused_add_rmsnorm`: Fused residual + norm
- `flashinfer_rmsnorm`: Standalone norm

**MOE Ops**:
- `torch.ops.trtllm.fused_moe`: Fused MOE kernel
- `torch.ops.trtllm.fp4_block_scale_moe_runner`: FP4 quantized MOE

### Fusion Patterns

**File**: [modeling_llama.py](_torch/models/modeling_llama.py) Lines 396-609

#### Pre-MLP Fusion (Lines 503-525)
```python
if self.pre_feed_forward_fusion_op != AllReduceFusionOp.NONE:
    # Fused: AllReduce + Residual + RMSNorm + Quantization
    mlp_input = AllReduce(
        self.pre_ffn_norm.weight,
        attn_output,
        residual=residual,
        fusion_op=self.pre_feed_forward_fusion_op,
        fusion_scale=pre_fusion_scale,
        ...
    )
else:
    # Separate operations
    hidden_states = attn_output + residual
    mlp_input = self.pre_ffn_norm(hidden_states)
```

**Fusion Operations** (Lines 396-443):
```python
self.pre_feed_forward_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM

if self.is_nvfp4:
    self.pre_feed_forward_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4
elif self.is_fp8_quant:
    self.pre_feed_forward_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8
```

**What's Fused**:
1. AllReduce (TP communication)
2. Residual addition
3. RMSNorm
4. Quantization (FP8 or NVFP4)

### Model Modifications

**None** - Models are defined from scratch using PyTorch modules:
- Input: Custom model implementations in `tensorrt_llm/_torch/models/`
- Output: PyTorch `nn.Module` with custom ops

---

## AutoDeploy Backend

### Overview

The AutoDeploy backend applies **graph-level transformations** to HuggingFace models. It uses `torch.export` to capture the graph, then applies transforms for sharding, KV cache, fusion, etc.

### Workflow

1. **Load HuggingFace model** (no modifications)
2. **Export to FX graph** (`torch.export`)
3. **Apply transforms**:
   - Sharding (tensor parallelism)
   - KV cache insertion
   - Attention pattern matching
   - Quantization
   - Fusion
4. **Compile** (optional: `torch.compile`)
5. **Execute** with TRT-LLM runtime

**Location**: [tensorrt_llm/_torch/auto_deploy/](_torch/auto_deploy/)

### Graph Transformations

#### 1. Attention Transform

**File**: [transform/library/attention.py](_torch/auto_deploy/transform/library/attention.py)

**Purpose**: Replace PyTorch attention with optimized kernels

**Pattern 1: repeat_kv** (Lines 38-48)
```python
# Original HuggingFace pattern
def _repeat_kv_pattern(hidden_states, n_rep) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = torch.unsqueeze(hidden_states, 2)
    hidden_states = hidden_states.expand(...)
    return hidden_states.reshape(...)

# Replacement
def _repeat_kv_repl(hidden_states, n_rep) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_attention_repeat_kv(hidden_states, n_rep)
```

**Pattern 2: SDPA** (Lines 52-230)

Multiple patterns for `F.scaled_dot_product_attention`:

```python
# Pattern: SDPA with causal mask
def sdpa_pattern(query, key, value, attn_mask, dropout, is_causal, scaling):
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scaling
    attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return torch.matmul(attn_weight, value)

# Replacement
def sdpa_repl(query, key, value, attn_mask, dropout, is_causal, scaling):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query, key, value,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=True,
        scale=scaling,
    )
```

**What It Does**:
- Finds PyTorch attention patterns in the graph
- Replaces with `torch.ops.auto_deploy.torch_attention_sdpa`
- Enables backend-specific optimizations (FlashAttention, etc.)

#### 2. KV Cache Transform

**File**: [transform/library/kvcache.py](_torch/auto_deploy/transform/library/kvcache.py)

**Purpose**: Insert KV cache management into the graph

**Operations** (Lines 69-200):
```python
class InsertCachedAttention(BaseTransform):
    def transform(self, gm: GraphModule):
        # 1. Process metadata
        metadata_op = get_prepare_metadata_op()

        # 2. Insert cache nodes
        past_key = gm.graph.placeholder("past_key")
        past_value = gm.graph.placeholder("past_value")

        # 3. Replace attention ops
        for node in gm.graph.nodes:
            if node.target == torch.ops.auto_deploy.torch_attention_sdpa:
                # Replace with cached version
                cached_attn_op = get_cached_attention_op()
                new_node = gm.graph.call_function(
                    cached_attn_op,
                    args=(q, k, v, metadata, past_key, past_value, ...),
                )
```

**What It Does**:
- Adds `past_key` and `past_value` as graph inputs
- Inserts cache update logic
- Modifies attention calls to use cached K/V

#### 3. Quantization Transform

**File**: [transform/library/quantization.py](_torch/auto_deploy/transform/library/quantization.py)

**Purpose**: Apply quantization (FP8, NVFP4, etc.)

**Base Class** (Lines 39-100):
```python
class Quantization(BaseTransform):
    algo_name: str = None  # "FP8", "NVFP4", etc.

    @staticmethod
    def target_op():
        """Target quantization ops (e.g., torch.ops.auto_deploy.torch_fake_quant_fp8_linear)"""

    @staticmethod
    def quantize_weight(original_weight: torch.Tensor) -> torch.Tensor:
        """Quantize weight tensor."""

    @staticmethod
    def build_custom_args_for_linear(scale_getattrs: Dict[str, Node]) -> Tuple:
        """Build args for quantized linear op."""
```

**Transform Process** (Lines 102-127):
```python
def transform(self, gm: GraphModule):
    for node in gm.graph.nodes:
        if node.target == torch.nn.functional.linear:
            # 1. Quantize weight
            quant_weight = self.quantize_weight(original_weight)

            # 2. Insert quantized linear
            new_node = gm.graph.call_function(
                self.target_op(),
                args=(input, quant_weight, scale, ...)
            )

            # 3. Replace original node
            node.replace_all_uses_with(new_node)
```

**Algorithms**:
- **FP8**: `torch.ops.auto_deploy.torch_fake_quant_fp8_linear`
- **NVFP4**: `torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear`
- **INT8**: `torch.ops.auto_deploy.torch_fake_quant_int8_linear`

#### 4. Fusion Transform

**File**: [transform/library/fusion.py](_torch/auto_deploy/transform/library/fusion.py)

**Purpose**: Fuse multiple GEMMs with same input

**GEMM Fusion** (Lines 20-80):
```python
def _insert_fused_gemm(gm: GraphModule, idx: int, parent_node: Node, linear_nodes: List[Node]):
    """
    Fuse GEMMs: [y1, y2, y3] = [x @ w1.T, x @ w2.T, x @ w3.T]
    Into: y = x @ [w1; w2; w3].T; y1, y2, y3 = split(y)
    """
    # 1. Concatenate weights
    weights = [node.args[1] for node in linear_nodes]
    fused_weight = torch.cat(weights, dim=0)

    # 2. Single fused GEMM
    fused_output = F.linear(input, fused_weight)

    # 3. Split output
    y1, y2, y3 = torch.split(fused_output, [w1.shape[0], w2.shape[0], w3.shape[0]])
```

**Quantized Fusion** (Lines 95-219):
```python
class QuantizationFusionMixin(ABC):
    """Fuse quantized GEMMs with same input."""

    def fuse_rule(self, weights: List[Tensor], **scales) -> Tuple[Tensor, Dict]:
        # FP8: fuse weights and scales
        fused_weight = torch.cat(weights, dim=0)
        fused_scale = torch.cat(scales['input_scale'], dim=0)
        return fused_weight, {'input_scale': fused_scale, ...}
```

**Example**: Gate + Up projection in SwiGLU
```
Before:
  gate = linear(x, w_gate)
  up = linear(x, w_up)

After:
  gate_up = linear(x, [w_gate; w_up])
  gate, up = split(gate_up)
```

#### 5. Sharding Transform

**File**: [transform/library/sharding.py](_torch/auto_deploy/transform/library/sharding.py)

**Purpose**: Apply tensor parallelism (TP) sharding

**Algorithm** (Lines 1-17):
```python
"""
Sharding for Tensor Parallelism:
1. Initialize unsharded model
2. Shard the graph IR:
   a. Identify linear nodes for TP tuples
   b. Reduce/Shard weight shapes (row or column dimension)
   c. Add all_reduce nodes where necessary
   d. Add checkpoint loading hooks
3. Load checkpoint and allocate sharded tensors
"""
```

**Transform Process** (Lines 55-98):
```python
class Sharding(BaseTransform):
    def transform(self, gm: GraphModule):
        # 1. Detect sharding patterns
        tp_transforms = self._detect_column_row_sharding(gm)
        ep_transforms = self._detect_ep_sharding(gm)
        bmm_transforms = self._detect_bmm_sharding(gm)

        # 2. Apply TP transforms
        for transform in tp_transforms:
            self._apply_tp_transform(gm, transform)

        # 3. Apply EP transforms (expert parallelism)
        for transform in ep_transforms:
            self._apply_ep_transform(gm, transform)
```

**Column Sharding** (Lines 135-242):
```python
def _detect_column_row_sharding(self, gm):
    """
    Column shard: Split weight along columns
      w_sharded = w[:, start:end]
      y_sharded = x @ w_sharded
      # No AllReduce needed (yet)

    Row shard: Split weight along rows
      w_sharded = w[start:end, :]
      y_sharded = x @ w_sharded
      y = AllReduce(y_sharded)  # ← Insert AllReduce
    """
```

**Example**: QKV projection
```
Unsharded:
  qkv = linear(x, w_qkv)  # Shape: [B, S, 3 * H]

Column sharded (TP=4):
  qkv_0 = linear(x, w_qkv[:, :3*H//4])  # Rank 0
  qkv_1 = linear(x, w_qkv[:, 3*H//4:])  # Rank 1
  # No AllReduce (yet)
```

### Custom Ops

**Attention Ops**:
- `torch.ops.auto_deploy.torch_attention_repeat_kv`
- `torch.ops.auto_deploy.torch_attention_sdpa`
- `torch.ops.auto_deploy.torch_cached_attention`

**Linear Ops**:
- `torch.ops.auto_deploy.torch_fake_quant_fp8_linear`
- `torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear`

**Norm Ops**:
- FlashInfer RMSNorm (same as PyTorch backend)

**MOE Ops**:
- `torch.ops.auto_deploy.torch_fused_moe`

### Model Modifications

**Source**: HuggingFace Transformers models (unmodified)
**Modifications**: Applied via graph transformations

1. **Attention**: PyTorch SDPA → `torch.ops.auto_deploy.torch_attention_sdpa`
2. **KV Cache**: Insert cache inputs + cache update logic
3. **Quantization**: `F.linear` → `torch_fake_quant_*_linear`
4. **Fusion**: Multiple `F.linear` → Single fused `F.linear` + split
5. **Sharding**: Full weights → Sharded weights + AllReduce

---

## Side-by-Side Comparison

### 1. Attention Implementation

| Backend | Type | KV Cache | RoPE | Custom Op |
|---------|------|----------|------|-----------|
| TensorRT | Plugin | Plugin-managed | Plugin-level | `gpt_attention` |
| PyTorch | Backend-agnostic | Explicit tensors | Separate or fused | `trtllm::attn_custom_op` |
| AutoDeploy | Pattern-matched | Graph insertion | Graph-level | `auto_deploy::torch_attention_sdpa` |

**TensorRT**:
- Compiles to optimized attention plugin
- All logic (QKV proj, RoPE, SDPA, output proj) in plugin
- Fastest, least flexible

**PyTorch**:
- Modular: separate QKV proj, RoPE, attention, output proj
- Backend selection (TRTLLM/FlashInfer/Triton)
- Can fuse RoPE into attention
- Most flexible

**AutoDeploy**:
- Starts with HuggingFace SDPA
- Replaces with custom op via pattern matching
- Automatic KV cache insertion

### 2. MLP/FFN Implementation

| Backend | Gate+Up Fusion | Down Proj | AllReduce | Custom Op |
|---------|----------------|-----------|-----------|-----------|
| TensorRT | FusedGatedMLP | Separate | Plugin-level | N/A (plugin) |
| PyTorch | Fused weights | TensorParallelMode.ROW | Explicit AllReduce | N/A |
| AutoDeploy | Graph fusion | Graph sharding | Graph insertion | N/A |

**All backends** use SwiGLU activation:
```python
gate, up = split(gate_up_proj(x))
output = down_proj(silu(gate) * up)
```

**Weight Fusion**:
- TensorRT: `FusedGatedMLP` layer
- PyTorch: `WeightMode.FUSED_GATE_UP_LINEAR`
- AutoDeploy: Fusion transform concatenates weights

### 3. Normalization

| Backend | Implementation | Residual Fusion | Custom Op |
|---------|---------------|-----------------|-----------|
| TensorRT | RmsNorm layer | AllReduce fusion | Plugin |
| PyTorch | FlashInfer / PyTorch | AllReduce + Norm | `flashinfer_fused_add_rmsnorm` |
| AutoDeploy | Pattern-based | Graph-level | Same as PyTorch |

**Fusion Pattern** (All backends):
```python
# Fused: AllReduce + Residual + RMSNorm
output = fused_op(hidden_states, residual=residual, weight=norm_weight)

# Equivalent to:
hidden_states = AllReduce(hidden_states)
hidden_states = hidden_states + residual
output = RMSNorm(hidden_states, weight)
```

### 4. Quantization

| Backend | Method | Granularity | Custom Op |
|---------|--------|-------------|-----------|
| TensorRT | QuantMode flags | Plugin-level | Quant plugins |
| PyTorch | Module-level | Per-linear | FP8/NVFP4 Linear |
| AutoDeploy | Transform-based | Graph-level | `torch_fake_quant_*_linear` |

**FP8 Quantization Example**:

**TensorRT**:
```python
# Set QuantMode
config = LLaMAConfig(quant_mode=QuantMode.FP8_QDQ())
# Plugin handles quantization internally
```

**PyTorch**:
```python
# Module-level
linear = Linear(..., quant_mode="FP8")
output = linear(input)  # Auto-quantizes
```

**AutoDeploy**:
```python
# Graph-level transform
transform = FP8Quantization()
gm = transform(gm)  # Replaces F.linear with quant ops
```

### 5. Tensor Parallelism

| Backend | Method | Sharding | AllReduce | Communication |
|---------|--------|----------|-----------|---------------|
| TensorRT | Mapping-based | Layer init | Functional | NCCL |
| PyTorch | TensorParallelMode | Linear init | AllReduce module | NCCL |
| AutoDeploy | Graph transform | Graph sharding | Graph insertion | NCCL |

**Example: QKV Projection (TP=4)**

**TensorRT**:
```python
attention = Attention(
    tp_size=4,
    tp_rank=0,  # Rank 0
    ...
)
# Plugin handles sharding internally
```

**PyTorch**:
```python
qkv_proj = Linear(
    hidden_size,
    3 * num_heads * head_dim,
    tensor_parallel_mode=TensorParallelMode.COLUMN,  # Column shard
)
# Weight loading: each rank loads 1/4 of columns
```

**AutoDeploy**:
```python
# Sharding transform
transform = Sharding(tp_size=4)
gm = transform(gm)
# Inserts sharding logic into graph
```

### 6. MOE Implementation

| Backend | Expert Routing | Expert Compute | Top-k | Custom Op |
|---------|---------------|----------------|-------|-----------|
| TensorRT | MOE plugin | Plugin-managed | Plugin-level | `_moe_plugin` |
| PyTorch | `fused_moe` module | Per-expert GatedMLP | Module-level | `torch.ops.trtllm.fused_moe` |
| AutoDeploy | Pattern-matched | Graph-level | Transform | `torch.ops.auto_deploy.torch_fused_moe` |

**All backends** follow the same flow:
1. Router: `logits = router(x)`
2. Top-k: `topk_weights, topk_indices = topk(logits, k)`
3. Expert compute: `expert_outputs = [expert_i(x) for i in topk_indices]`
4. Combine: `output = sum(topk_weights * expert_outputs)`

**Optimization** (all backends):
- Fuse expert GEMMs (gate+up, down)
- Quantization (FP8, NVFP4)
- Load balancing

---

## When to Use Each Backend

### TensorRT Backend

**Use when**:
- Production deployment with maximum performance
- Model is mature and fully supported
- Build step is acceptable (CI/CD pipeline)
- Static input shapes
- Need absolute lowest latency

**Don't use when**:
- Rapid iteration required
- Model not yet supported
- Dynamic shapes required
- Debugging custom logic

**Command**:
```bash
# 1. Build engine
trtllm-build --checkpoint_dir <hf_model> --output_dir <engine>

# 2. Run inference
python examples/run.py --engine_dir <engine>
```

### PyTorch Backend

**Use when**:
- Development and debugging
- Model not yet in TRT-LLM
- Need flexibility (custom layers, dynamic shapes)
- Rapid experimentation
- **Enable torch.compile** for production-level performance

**Don't use when**:
- Need absolute maximum performance (use TRT backend)
- Don't care about flexibility (use TRT backend)

**Command**:
```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import TorchCompileConfig

# Without torch.compile (debugging)
llm = LLM(model_path, backend="pytorch")

# With torch.compile (production)
llm = LLM(
    model_path,
    backend="pytorch",
    torch_compile_config=TorchCompileConfig(
        enable_fullgraph=True,
        enable_inductor=True,
    ),
)
```

### AutoDeploy Backend

**Use when**:
- Day-0 support for new HuggingFace models
- Don't want to implement model in TRT-LLM
- Need automatic optimizations (sharding, KV cache, fusion)
- Prototyping new architectures
- **Prototype status**: expect breaking changes

**Don't use when**:
- Model already in TRT-LLM (use PyTorch or TRT backend)
- Need production stability (use PyTorch backend)
- Need maximum performance (use TRT backend + torch.compile)

**Command**:
```python
from tensorrt_llm._torch.auto_deploy import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",  # HuggingFace model
    compile_backend="torch-opt",
    world_size=4,
)
```

---

## Documentation Links

### TensorRT Backend

**Docs**:
- Main guide: [docs/source/](../../docs/source/)
- Model implementations: [tensorrt_llm/models/](../models/)
- Layers: [tensorrt_llm/layers/](../layers/)

**Examples**:
- Building engines: `examples/llama/`
- Running inference: `examples/run.py`

**Tests**:
- Model tests: `tests/model/`
- Layer tests: `tests/layers/`

### PyTorch Backend

**Docs**:
- API reference: LLM API docs (see `backend` parameter)
- torch.compile: `TorchCompileConfig` docstrings

**Examples**:
- [examples/llm-api/quickstart_advanced.py](../../examples/llm-api/quickstart_advanced.py)
  ```bash
  python quickstart_advanced.py \
      --model_dir <path> \
      --use_torch_compile \
      --use_piecewise_cuda_graph
  ```

**Tests**:
- [tests/integration/defs/accuracy/test_llm_api_pytorch.py](../../tests/integration/defs/accuracy/test_llm_api_pytorch.py)

**Key Files**:
- Models: [tensorrt_llm/_torch/models/](_torch/models/)
- Modules: [tensorrt_llm/_torch/modules/](_torch/modules/)
- Backends: [tensorrt_llm/_torch/attention_backend/](_torch/attention_backend/)

### AutoDeploy Backend

**Docs**:
- Main guide: [docs/source/torch/auto_deploy/auto-deploy.md](../../docs/source/torch/auto_deploy/auto-deploy.md)
- Advanced: [docs/source/torch/auto_deploy/advanced/](../../docs/source/torch/auto_deploy/advanced/)

**Examples**:
- [examples/auto_deploy/build_and_run_ad.py](../../examples/auto_deploy/build_and_run_ad.py)
  ```bash
  python build_and_run_ad.py \
      --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
      --args.world-size 2
  ```

**Tests**:
- [tests/integration/defs/accuracy/test_llm_api_autodeploy.py](../../tests/integration/defs/accuracy/test_llm_api_autodeploy.py)

**Key Files**:
- Transforms: [tensorrt_llm/_torch/auto_deploy/transform/library/](_torch/auto_deploy/transform/library/)
- Export: [tensorrt_llm/_torch/auto_deploy/export/](_torch/auto_deploy/export/)
- Custom ops: [tensorrt_llm/_torch/auto_deploy/custom_ops/](_torch/auto_deploy/custom_ops/)

---

## Summary

**Three distinct backends**:
1. **TensorRT**: Graph compilation → TensorRT engine (fastest, least flexible)
2. **PyTorch**: Native PyTorch execution (flexible, fast with torch.compile)
3. **AutoDeploy**: Graph transformations on HF models (day-0 support, prototype)

**Key differences**:
- **Model source**: TRT-LLM layers vs PyTorch modules vs HuggingFace models
- **Execution**: Engine vs eager/compiled vs transformed graph
- **Customization**: Plugin-level vs module-level vs graph-level
- **Flexibility**: Low vs high vs medium
- **Performance**: Highest vs high vs medium-high

**When to use**:
- **TensorRT**: Production, maximum performance
- **PyTorch**: Development, flexibility, production with torch.compile
- **AutoDeploy**: New models, rapid prototyping, automatic optimization

**All backends** support:
- Multi-GPU (TP, PP, EP)
- Quantization (FP8, INT8, NVFP4)
- KV cache
- MOE
- Custom attention backends
- Fusion patterns

Choose based on your needs: **performance** (TRT), **flexibility** (PyTorch), or **ease** (AutoDeploy).

---

**Last Updated**: 2025-10-13
**See also**: [ARCHITECTURE.md](ARCHITECTURE.md), [ARCHITECTURE_ADDENDUM.md](ARCHITECTURE_ADDENDUM.md)
