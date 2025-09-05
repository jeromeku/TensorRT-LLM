# ruff: noqa E402
import os
from typing import Optional, Union

import torch
import torch.nn as nn
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternPrettyPrinter, fwd_only, gen_pattern
from torch.fx import GraphModule
FLASHINFER_CACHE_DIR = "./flashinfer_cache"
os.environ["CUDA_HOME"] = "/home/jeromeku/cuda-toolkit"
os.environ["FLASHINFER_JIT_VERBOSE"] = "1"
os.environ["FLASHINFER_WORKSPACE_BASE"] = FLASHINFER_CACHE_DIR
os.makedirs(FLASHINFER_CACHE_DIR, exist_ok=True)

# import tensorrt_llm
# import tensorrt_llm._torch
# import tensorrt_llm._torch.modules
# import tensorrt_llm._torch.modules.rms_norm
from flashinfer.norm import fused_add_rmsnorm, rmsnorm

namespace = "custom"
@torch.library.custom_op(f"{namespace}::flashinfer_rmsnorm", mutates_args=())
def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return rmsnorm(input, weight, eps, enable_pdl=False)


@flashinfer_rmsnorm.register_fake
def _(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(input)


@torch.library.custom_op(f"{namespace}::flashinfer_fused_add_rmsnorm", mutates_args=("input", "residual"))
def flashinfer_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    return fused_add_rmsnorm(input, residual, weight, eps, enable_pdl=False)


@flashinfer_fused_add_rmsnorm.register_fake
def _(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch.empty_like(input)


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        has_weights: bool = True,
        use_flashinfer: bool = False,
    ):
        super().__init__()
        if has_weights:
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        else:
            self.register_buffer(
                "weight", torch.ones(hidden_size, dtype=dtype, device=device), persistent=False
            )
        self.variance_epsilon = eps
        self.use_flashinfer = use_flashinfer

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = ...,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.use_flashinfer:
            if isinstance(residual, torch.Tensor):
                flashinfer_fused_add_rmsnorm(
                    hidden_states, residual, self.weight, self.variance_epsilon
                )
            else:
                hidden_states = flashinfer_rmsnorm(
                    hidden_states, self.weight, self.variance_epsilon
                )
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            if isinstance(residual, torch.Tensor):
                hidden_states = hidden_states + residual.to(torch.float32)
                residual = hidden_states.to(input_dtype)

            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            hidden_states = self.weight * hidden_states.to(input_dtype)

        if residual is ...:
            return hidden_states
        else:
            return hidden_states, residual

    def skip_forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = ...,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if residual is ...:
            return hidden_states
        else:
            return hidden_states, residual


def source_pattern(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    at = auto_functionalized(
        torch.ops.custom.flashinfer_fused_add_rmsnorm.default,
        input=x,
        residual=residual,
        weight=weight,
        eps=eps,
    )
    output_shape = (x.shape[0], 2 * 3)
    output = x.new_empty(output_shape, dtype=x.dtype)
    return at[1], at[2], output * output


p = PatternPrettyPrinter()

M, N = 4, 16
x = torch.randn(M, N).cuda().half()
res = x.clone()
weight = torch.ones((N,)).cuda().half()
eps = 1e-5

out = rmsnorm(x, weight=weight, eps=eps)
ref_norm = RMSNorm(N, eps, dtype=x.dtype, device="cuda")
ref = ref_norm(x)
print((ref - out).abs().max().item())
custom_out = torch.ops.custom.flashinfer_rmsnorm(x, weight, eps)
print((ref - custom_out).abs().max().item())

pattern = gen_pattern(source_pattern, [x, res, weight, eps], fwd_only)
print(pattern)
print(PatternPrettyPrinter.run(pattern))

torch._dynamo.mark_dynamic(x, 0)

def print_aten(gm: GraphModule, _):
    print("asdas", type(gm))
    breakpoint()
    gm.graph.print_tabular()
    return gm


func = torch.compile(source_pattern, backend=print_aten)

func = aot_function(source_pattern, fw_compiler=print_aten)
func(x, res, weight, eps)
