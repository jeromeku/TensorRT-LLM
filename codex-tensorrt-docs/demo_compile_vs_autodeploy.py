import torch
import torch.nn as nn


class Toy(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x):
        return self.lin2(torch.nn.functional.relu(self.lin1(x)))


def demo_internal_compiler():
    from tensorrt_llm._torch.compilation import Backend

    model = Toy().cuda().eval()
    backend = Backend(enable_inductor=True, max_num_streams=1)
    compiled = torch.compile(model, backend=backend)
    x = torch.randn(32, 1024, device="cuda")
    y = compiled(x)
    print("internal compiler output shape:", tuple(y.shape))


def demo_autodeploy_torch_opt():
    from tensorrt_llm._torch.auto_deploy.compile.backends.torch_opt import (
        TorchOptCompiler,
    )

    model = Toy().cuda().eval()
    compiler = TorchOptCompiler(
        model,
        args=(),
        kwargs={},
        max_batch_size=128,
        cuda_graph_batch_sizes=[1, 32, 128],
    )
    captured = compiler.compile()
    y = captured(torch.randn(32, 1024, device="cuda"))
    print("autodeploy torch-opt output shape:", tuple(y.shape))


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    demo_internal_compiler()
    demo_autodeploy_torch_opt()

