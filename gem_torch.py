import time
import torch

N=2048
a = torch.randn(N, N, dtype=torch.float32, device="mps")
b = torch.randn(N, N, dtype=torch.float32, device="mps")

def gemm():
    print("Start matrix mul")
    st = time.monotonic()
    c = a @ b
    torch.zeros(1, device="mps").cpu()
    return time.monotonic() - st

flops = N*N*2*N
et = min([gemm() for x in range(10)])
print(f"{flops*1e-9/et} GFLOPS")