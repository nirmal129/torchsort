import sys
from collections import defaultdict
from timeit import timeit
import time

import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import os
import gc

import torchsort

def fix_seed(seed=42):
    # 1. Python's built-in random module
    random.seed(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. Torch (CPU)
    torch.manual_seed(seed)
    
    # 4. Torch (GPU/CUDA)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    
    # 5. CUDNN (The critical part for determinism)
    # This ensures that the same algorithm is chosen for convolutions/ops every time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 6. Python Hash Seed (Good for dictionary/set ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 7. (Optional) Force PyTorch to use deterministic algorithms
    # Note: This might throw an error if an op doesn't have a deterministic version
    # torch.use_deterministic_algorithms(True)

    print(f"✅ Seeds fixed to: {seed}")

# try:
#     import fast_soft_sort.pytorch_ops as fss
# except ImportError:
#     print("install fast_soft_sort:")
#     print("pip install git+https://github.com/google-research/fast-soft-sort")
#     sys.exit()

init_len, final_len, step_len = 500, 20000, 500
N = list(range(init_len, final_len + step_len, step_len))
B = [2 ** i for i in range(9)]
B_CUDA = [2 ** i for i in range(13)]
SAMPLES = 100
CONVERT = 1e-6  # convert seconds to micro-seconds

def time_fn(f):
    return timeit(f, number=SAMPLES) / SAMPLES / CONVERT


def backward(f, x):
    y = f(x)
    torch.autograd.grad(y.sum(), x)

plt.style.use("bmh")
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
def style(name):
    if name == "torch.sort":
        return {"color": palette[0]}
    linestyle = "--" if "backward" in name else "-"
    if "fast_soft_sort" in name:
        return {"color": None, "linestyle": linestyle}
    elif "_parallel" in name:
        return {"color": palette[1], "linestyle": linestyle}
    else:
        return {"color": palette[2], "linestyle": linestyle}


def batch_size(ax):
    data = defaultdict(list)
    for b in B:
        x = torch.randn(b, 100)
        data["torch.sort"].append(time_fn(lambda: torch.sort(x)))
        data["torchsort"].append(time_fn(lambda: torchsort.soft_sort(x)))
        # data["fast_soft_sort"].append(time_fn(lambda: fss.soft_sort(x)))
        x = torch.randn(b, 100, requires_grad=True)
        data["torchsort (with backward)"].append(
            time_fn(lambda: backward(torchsort.soft_sort, x))
        )
        # data["fast_soft_sort (with backward)"].append(
        #     time_fn(lambda: backward(fss.soft_sort, x))
        # )

    for label in data.keys():
        ax.plot(B, data[label], label=label, **style(label))
    ax.set_xlabel("Batch Size")
    ax.set_ylim(0, 5000)
    ax.set_ylabel("Execution Time (μs)")
    ax.legend()


def sequence_length(ax):
    data = defaultdict(list)
    for n in N:
        x = torch.randn(1, n)
        data["torch.sort"].append(time_fn(lambda: torch.sort(x)))
        data["torchsort"].append(time_fn(lambda: torchsort.soft_sort(x)))
        # data["fast_soft_sort"].append(time_fn(lambda: fss.soft_sort(x)))
        x = torch.randn(1, n, requires_grad=True)
        data["torchsort (with backward)"].append(
            time_fn(lambda: backward(torchsort.soft_sort, x))
        )
        # data["fast_soft_sort (with backward)"].append(
        #     time_fn(lambda: backward(fss.soft_sort, x))
        # )

    for label in data.keys():
        ax.plot(N, data[label], label=label, **style(label))
    ax.set_xlabel("Sequence Length")
    ax.set_ylim(0, 1000)
    ax.set_ylabel("Execution Time (μs)")
    ax.legend()


def batch_size_cuda(ax):
    data = defaultdict(list)
    print("Benchmarking batch sizes")
    L = 512
    for b in B_CUDA:
        print(f"Batch size {b} done")
        x = torch.randn(b, L).cuda()
        x_seq = x.clone()
        x_par = x.clone()

        data["torch.sort"].append(time_fn(lambda: torch.sort(x_seq)))
        data["torchsort_parallel"].append(time_fn(lambda: torchsort.soft_sort_parallel(x_par)))
        data["torchsort"].append(time_fn(lambda: torchsort.soft_sort(x_seq)))

        x = torch.randn(b, L, requires_grad=True).cuda()
        x_seq = x.clone().detach().requires_grad_(True)
        x_par = x.clone().detach().requires_grad_(True)

        data["torchsort_parallel (with backward)"].append(
            time_fn(lambda: backward(torchsort.soft_sort_parallel, x_par))
        )
        data["torchsort (with backward)"].append(
            time_fn(lambda: backward(torchsort.soft_sort, x_seq))
        )
    for label in data.keys():
        ax.plot(B_CUDA, data[label], label=label, **style(label))
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Execution Time (μs)")
    ax.legend()


def sequence_length_cuda(ax):
    data = defaultdict(list)
    print("Benchmarking sequence lengths")
    B = 4
    for n in N:
        print(f"Sequence length {n} running")
        x_init = torch.randn(B, n).cuda()
        x = x_init # torch.sort(x_init, descending=True)
        x_seq = x.clone() # x.values.clone()
        x_par = x.clone() # x.values.clone()

        data["torch.sort"].append(time_fn(lambda: torch.sort(x_seq)))
        data["torchsort_parallel"].append(time_fn(lambda: torchsort.soft_sort_parallel(x_par)))
        data["torchsort"].append(time_fn(lambda: torchsort.soft_sort(x_seq)))

        x_init = torch.randn(B, n, requires_grad=True).cuda()
        x = x_init # torch.sort(x_init, descending=True)
        x_seq = x.clone().detach().requires_grad_(True) # x.values.clone().detach().requires_grad_(True)
        x_par = x.clone().detach().requires_grad_(True) # x.values.clone().detach().requires_grad_(True)

        data["torchsort_parallel (with backward)"].append(
            time_fn(lambda: backward(torchsort.soft_sort_parallel, x_par))
        )
        data["torchsort (with backward)"].append(
            time_fn(lambda: backward(torchsort.soft_sort, x_seq))
        )
    for label in data.keys():
        ax.plot(N, data[label], label=label, **style(label))
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Execution Time (μs)")
    ax.legend()


if __name__ == "__main__":
    fix_seed(42)
    
    # jit/warmup
    # x = torch.randn(1, 10, requires_grad=True)
    # backward(torchsort.soft_sort, x)
    # backward(fss.soft_sort, x)

    # fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)
    # sequence_length(ax1)
    # batch_size(ax2)
    # fig.suptitle("Torchsort Benchmark: CPU")
    # fig.tight_layout()
    # plt.savefig("extra/benchmark.png")

    if torch.cuda.is_available():
        # warmup
        x = torch.randn(4, 256, requires_grad=True).cuda()
        x_seq = x.clone().detach().requires_grad_(True)
        x_par = x.clone().detach().requires_grad_(True)
        backward(torchsort.soft_sort_parallel, x_par)
        backward(torchsort.soft_sort, x_seq)
        print("--- Warm up done ---")

        fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)
        sequence_length_cuda(ax1)
        batch_size_cuda(ax2)
        fig.suptitle("Torchsort Benchmark: CUDA")
        fig.tight_layout()
        plt.savefig("extra/benchmark_cuda_nnp.png")
