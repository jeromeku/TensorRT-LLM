import torch
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import (
    fp8_block_scale_moe_runner,
)


def main():
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.synchronize()

    # Example shapes
    T, H, E, I = 128, 4096, 64, 1536
    hs = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    hs_scale = torch.ones(T, device="cuda", dtype=torch.float32)
    w1 = torch.randn(I, H, device="cuda", dtype=torch.float8_e4m3fn)
    s1 = torch.ones(I, device="cuda", dtype=torch.float32)
    w2 = torch.randn(H, I, device="cuda", dtype=torch.float8_e4m3fn)
    s2 = torch.ones(H, device="cuda", dtype=torch.float32)
    logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)

    with autotune(cache_path="codex-tensorrt-docs/autotune_cache.json"):
        out = fp8_block_scale_moe_runner(
            logits,
            None,
            hs,
            hs_scale,
            w1,
            s1,
            w2,
            s2,
            num_experts=E,
            top_k=4,
            n_group=None,
            topk_group=None,
            intermediate_size=I,
            local_expert_offset=0,
            local_num_experts=E,
            routed_scaling_factor=None,
            routing_method_type=0,
        )
    print("fp8 moe output shape:", tuple(out.shape))


if __name__ == "__main__":
    main()

