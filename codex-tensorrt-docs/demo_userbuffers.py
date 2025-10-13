import torch
from tensorrt_llm._torch.custom_ops import userbuffers_custom_ops as ub


def main():
    assert torch.cuda.is_available(), "CUDA required"
    a = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(32, 1024, device="cuda", dtype=torch.bfloat16)

    a_ub = ub.copy_to_userbuffers(a)
    c = ub.add_to_ub(a_ub, b)

    w = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)
    z = ub.matmul_to_ub(c, w)

    print("copy_to_userbuffers shape:", tuple(a_ub.shape))
    print("add_to_ub shape:", tuple(c.shape))
    print("matmul_to_ub shape:", tuple(z.shape))


if __name__ == "__main__":
    main()

