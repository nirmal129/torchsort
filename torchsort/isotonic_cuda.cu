//  Copyright 2007-2020 The scikit-learn developers.
//  Copyright 2020 Google LLC.
//  Copyright 2021 Teddy Koker.
//  All rights reserved.
// 
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
// 
//    a. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//    b. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//    c. Neither the name of the Scikit-learn Developers  nor the names of
//       its contributors may be used to endorse or promote products
//       derived from this software without specific prior written
//       permission.
// 
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
//  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
//  OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.


#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

//  Copied from fast-soft-sort (https://bit.ly/3r0gOav) with the following modifications:
//  - replace numpy functions with torch equivalents
//  - re-write in CUDA
//  - return solution in place
//  - added backward pass (vector jacobian product)

//  Copied from scikit-learn with the following modifications:
//  - use decreasing constraints by default,
//  - do not return solution in place, rather save in array `sol`,
//  - avoid some needless multiplications.
// namespace {

// Numerically stable log-add-exp
template <typename scalar_t>
__device__ __forceinline__ scalar_t log_add_exp(scalar_t x, scalar_t y) {
    scalar_t larger = max(x, y);
    scalar_t smaller = min(x, y);
    return larger + log1p(exp(smaller - larger));
}

// Returns partition corresponding to solution. Expects sizes to be zeros
template <typename scalar_t>
__device__ void partition(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> solution, 
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sizes, 
    int n, 
    int b) {

    const scalar_t eps = 1.0e-9;
    int tail = 1;

    if (n > 0) {
        sizes[b][0] = 1;
    }

    for (int i = 1; i < n; i++) {
        if (std::abs(solution[b][i] - solution[b][i - 1]) > eps) {
            tail += 1; 
        }
        sizes[b][tail - 1] += 1;
    }
}


template <typename scalar_t>
__global__ void isotonic_l2_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> s,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sums,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> target,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> c,
    int n,
    int batch) {

    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch) {
        // outside the batch
        return;
    }

    // target describes a list of blocks.  at any time, if [i..j] (inclusive) is
    // an active block, then target[i] := j and target[j] := i.
    for (int i = 0; i < n; i++) {
        c[b][i] = 1.0;
        sol[b][i] = s[b][i];
        sums[b][i] = s[b][i];
        target[b][i] = i;
    }

    int i = 0;
    while (i < n) {
        auto k = target[b][i] + 1;
        if (k == n) {
            break;
        }
        if (sol[b][i] > sol[b][k]) {
            i = k;
            continue;
        }
        auto sum_y = sums[b][i];
        auto sum_c = c[b][i];
        while (true) {
            // We are within an increasing subsequence
            auto prev_y = sol[b][k];
            sum_y += sums[b][k];
            sum_c += c[b][k];
            k = target[b][k] + 1;
            if ((k == n) || (prev_y > sol[b][k])) {
                // Non-singleton increasing subsequence is finished,
                // update first entry.
                sol[b][i] = sum_y / sum_c;
                sums[b][i] = sum_y;
                c[b][i] = sum_c;
                target[b][i] = k - 1;
                target[b][k - 1] = i;
                if (i > 0) {
                    // Backtrack if we can.  This makes the algorithm
                    // single-pass and ensures O(n) complexity.
                    i = target[b][i - 1];
                }
                // Otherwise, restart from the same point
                break;
            }
        }
    }
    // Reconstruct the solution
    i = 0;
    while (i < n) {
        auto k = target[b][i] + 1;
        for (int j = i + 1; j < k; j++) {
            sol[b][j] = sol[b][i];
        }
        i = k;
    }
}

template <typename scalar_t>
__global__ void isotonic_kl_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> y,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> w,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> lse_y_,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> lse_w_,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> target,
    int n,
    int batch) {

    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch) {
        // outside the batch
        return;
    }

    // target describes a list of blocks.  At any time, if [i..j] (inclusive) is
    // an active block, then target[i] := j and target[j] := i.
    for (int i = 0; i < n; i++) {
        sol[b][i] = y[b][i] - w[b][i];
        lse_y_[b][i] = y[b][i];
        lse_w_[b][i] = w[b][i];
        target[b][i] = i;
    }

    int i = 0;
    while (i < n) {
        auto k = target[b][i] + 1;
        if (k == n) {
            break;
        }
        if (sol[b][i] > sol[b][k]) {
            i = k;
            continue;
        }
        auto lse_y = lse_y_[b][i];
        auto lse_w = lse_w_[b][i];
        while (true) {
            // We are within an increasing subsequence
            auto prev_y = sol[b][k];
            lse_y = log_add_exp(lse_y, lse_y_[b][k]);
            lse_w = log_add_exp(lse_w, lse_w_[b][k]);
            k = target[b][k] + 1;
            if ((k == n) || (prev_y > sol[b][k])) {
                // Non-singleton increasing subsequence is finished,
                // update first entry.
                sol[b][i] = lse_y - lse_w;
                lse_y_[b][i] = lse_y;
                lse_w_[b][i] = lse_w;
                target[b][i] = k - 1;
                target[b][k - 1] = i;
                if (i > 0) {
                    // Backtrack if we can.  This makes the algorithm
                    // single-pass and ensures O(n) complexity.
                    i = target[b][i - 1];
                }
                // Otherwise, restart from the same point
                break;
            }
        }
    }
    // Reconstruct the solution
    i = 0;
    while (i < n) {
        auto k = target[b][i] + 1;
        for (int j = i + 1; j < k; j++) {
            sol[b][j] = sol[b][i];
        }
        i = k;
    }
}


template <typename scalar_t>
__global__ void isotonic_l2_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> s, // not used
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_input,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> ret,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sizes,
    int n,
    int batch) {

    int end;
    scalar_t sum;
    scalar_t val;

    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch) {
        // outside the batch
        return;
    }

    int start = 0;
    partition(sol, sizes, n, b);
    for (int size = 0; (sizes[b][size] > 0 && size < n); size++) {
        end = start + sizes[b][size];
        sum = 0;
        val = 1.0 / (scalar_t) sizes[b][size];
        
        for (int i = start; i < end; i++) {
            sum += grad_input[b][i];
        }
        for (int i = start; i < end; i++) {
            ret[b][i] = val * sum;
        }
        start = end;
    }
}

template <typename scalar_t>
__global__ void isotonic_kl_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> s,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_input,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> ret,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sizes,
    int n,
    int batch) {

    int end;
    scalar_t sum;
    scalar_t softmax;

    const int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch) {
        // outside the batch
        return;
    }

    int start = 0;
    partition(sol, sizes, n, b);
    for (int size = 0; (sizes[b][size] > 0 && size < n); size++) {
        end = start + sizes[b][size];
        sum = 0;
        softmax = 0;
        
        for (int i = start; i < end; i++) {
            softmax += std::exp(s[b][i]);
            sum += grad_input[b][i];
        }
        for (int i = start; i < end; i++) {
            ret[b][i] = std::exp(s[b][i]) / softmax * sum;
        }
        start = end;
    }
}

template <typename scalar_t>
__global__ void isotonic_l2_parallel_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> s,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sums,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> target,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> c,
    int n,
    int batch) {
    // Each block handles one sequence in the batch.
    // Threads within a block cooperate on that sequence.
    const int b   = blockIdx.x;          // batch index
    const int tid = threadIdx.x;         // lane within the sequence
    const int tpb = blockDim.x;          // threads_per_seq (e.g. 256)

    if (b >= batch) return;

    // ------------------------------------------------------------------
    // Phase 1 (parallel): initialise c, sol, sums, target
    // Each thread handles a strided slice of [0, n).
    // ------------------------------------------------------------------
    for (int i = tid; i < n; i += tpb) {
        c[b][i]      = static_cast<scalar_t>(1);
        sol[b][i]    = s[b][i];
        sums[b][i]   = s[b][i];
        target[b][i] = i;
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2 (parallel): divide-and-conquer PAV
    // ------------------------------------------------------------------

    // Getting previous power-of-2 for binary tree reduction, some threads sit idle
    const int tpb_p2 = 1 << (31 - __clz(tpb));

    // ------------------------------------------------------------------
    // Phase 2a: each thread runs independent PAV on its chunk
    // ------------------------------------------------------------------
    if (tid < tpb_p2) {
        const int remainder   = n % tpb_p2;
        const int chunk_base  = n / tpb_p2;
        const int chunk_size  = chunk_base + (tid < remainder ? 1 : 0);
        const int chunk_start = tid * chunk_base + (tid < remainder ? tid : remainder);
        const int chunk_end   = chunk_start + chunk_size;

        int i = chunk_start;
        while (i < chunk_end) {
            auto k = target[b][i] + 1;
            if (k == chunk_end) break;
            if (sol[b][i] > sol[b][k]) {  // non-increasing: no violation, advance
                i = k;
                continue;
            }
            auto sum_y = sums[b][i];
            auto sum_c = c[b][i];
            while (true) {
                auto prev_y  = sol[b][k];
                sum_y       += sums[b][k];
                sum_c       += c[b][k];
                k            = target[b][k] + 1;
                if ((k == chunk_end) || (prev_y > sol[b][k])) {
                    sol[b][i]        = sum_y / sum_c;
                    sums[b][i]       = sum_y;
                    c[b][i]          = sum_c;
                    target[b][i]     = k - 1;
                    target[b][k - 1] = i;
                    if (i > chunk_start) {
                        i = target[b][i - 1];
                    }
                    break;
                }
            }
        }
    }

    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2b: merge levels
    // ------------------------------------------------------------------
    for (int stride = 1; stride < tpb_p2; stride <<= 1) {
        if (tid < tpb_p2 && (tid % (2 * stride)) == 0) {
            const int remainder     = n % tpb_p2;
            const int chunk_base    = n / tpb_p2;
            const int right_tid     = tid + stride;
            const int right_tid_end = tid + 2 * stride;

            const int left_start  = tid * chunk_base + (tid < remainder ? tid : remainder);
            const int right_start = right_tid * chunk_base + (right_tid < remainder ? right_tid : remainder);
            const int right_end   = (right_tid_end >= tpb_p2)
                                        ? n
                                        : right_tid_end * chunk_base + (right_tid_end < remainder ? right_tid_end : remainder);

            if (right_start < n) {
                int i = left_start;
                while (i < right_end) {
                    int k = target[b][i] + 1;
                    if (k == right_end) break;
                    if (sol[b][i] > sol[b][k]) {  // non-increasing: no violation, advance
                        i = k;
                        continue;
                    }
                    // violation: sol[i] <= sol[k], merge
                    auto sum_y = sums[b][i];
                    auto sum_c = c[b][i];
                    while (true) {
                        auto prev_y  = sol[b][k];
                        sum_y       += sums[b][k];
                        sum_c       += c[b][k];
                        k            = target[b][k] + 1;
                        if ((k == right_end) || (prev_y > sol[b][k])) {
                            sol[b][i]        = sum_y / sum_c;
                            sums[b][i]       = sum_y;
                            c[b][i]          = sum_c;
                            target[b][i]     = k - 1;
                            target[b][k - 1] = i;
                            if (i > left_start) {
                                i = target[b][i - 1];
                            }
                            break;
                        }
                    }
                }
            }
        }

        __syncthreads();
    }
    
    // Sync threads before reconstruction
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 3 (parallel): reconstruct — fill each block's constant value.
    //
    // Strategy: every thread that owns position i (i.e. i % tpb == tid)
    // checks whether it is the START of a block (target[b][i-1]+1 == i
    // or i==0) and if so, writes sol[b][i] to all j in [i+1, target[b][i]).
    // This avoids any write conflicts because blocks are disjoint.
    //
    // Edge cases:
    // * If a block spans fewer elements than tpb, some threads simply
    //   own no start index and do nothing
    // * Singleton blocks (target[b][i] == i) produce an empty inner loop
    // ------------------------------------------------------------------
    for (int i = tid; i < n; i += tpb) {
        // Is i the start of a PAV block?
        bool is_start = (i == 0) || (target[b][i - 1] + 1 == i);
        if (is_start) {
            int end = target[b][i] + 1;   // exclusive end of this block
            scalar_t val = sol[b][i];
            for (int j = i + 1; j < end; j++) {
                sol[b][j] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel prefix-sum (exclusive scan) on a single row of length N,
// operating entirely in global memory.  Called with B blocks, <=1024 threads.
//
// Algorithm: naive O(N log N) work
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void isotonic_l2_backward_parallel_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sol,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_in,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> out,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pid,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> pid_copy,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> psum,  // always float
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> pcnt,  // always float
    int N) {
    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    const int T   = blockDim.x;

    // Phase 1a: boundary flags into both pid and pid_copy
    pid[b][0]      = static_cast<scalar_t>(0);
    pid_copy[b][0] = static_cast<scalar_t>(0);
    for (int i = tid + 1; i < N; i += T) {
        scalar_t flag = (fabsf((float)(sol[b][i] - sol[b][i-1])) > 1e-9f)
                        ? static_cast<scalar_t>(1)
                        : static_cast<scalar_t>(0);
        pid[b][i]      = flag;
        pid_copy[b][i] = flag;
    }
    __syncthreads();

    // Phase 1b: parallel inclusive prefix sum
    // Each stride pass reads from pid_copy, writes to pid, then swaps.
    for (int stride = 1; stride < N; stride <<= 1) {
        for (int i = tid; i < N; i += T) {
            pid[b][i] = (i >= stride)
                        ? pid_copy[b][i] + pid_copy[b][i - stride]
                        : pid_copy[b][i];
        }
        __syncthreads();

        // swap: copy pid back into pid_copy for next stride
        for (int i = tid; i < N; i += T)
            pid_copy[b][i] = pid[b][i];
        __syncthreads();
    }
    // pid[b][i] now holds the pool index for position i

    // Phase 2: zero psum/pcnt
    for (int i = tid; i < N; i += T) {
        psum[b][i] = static_cast<scalar_t>(0);
        pcnt[b][i] = static_cast<scalar_t>(0);
    }
    __syncthreads();

    // Phase 3: accumulate grad into pool buckets
    for (int i = tid; i < N; i += T) {
        int p = (int)pid[b][i];
        atomicAdd(&psum[b][p], (float)grad_in[b][i]);
        atomicAdd(&pcnt[b][p], 1.0f);
    }
    __syncthreads();

    // Phase 4: writeback averaged gradient
    for (int i = tid; i < N; i += T) {
        int p     = (int)pid[b][i];
        out[b][i] = static_cast<scalar_t>(psum[b][p] / (pcnt[b][p] + 1e-12f));
    }
}

// === Host Functions ===================================================================================

// Solves an isotonic regression problem using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.
torch::Tensor isotonic_l2(torch::Tensor y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(y));
    auto batch = y.size(0);
    auto n = y.size(1);
    auto sol = torch::zeros_like(y);
    auto sums = torch::zeros_like(y);
    auto target = torch::zeros_like(y);
    auto c = torch::zeros_like(y);

    const int threads = 1024;
    const int blocks = (batch + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "isotonic_l2", ([&] {
        isotonic_l2_kernel<scalar_t><<<blocks, threads>>>(
            y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sums.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            target.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            c.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            n,
            batch);
    }));
    return sol;
}

// Solves an isotonic regression problem using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.
torch::Tensor isotonic_l2_parallel(torch::Tensor y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(y));

    auto batch = y.size(0);
    auto n     = y.size(1);

    auto sol    = torch::zeros_like(y);
    auto sums   = torch::zeros_like(y);
    auto target = torch::zeros_like(y);
    auto c      = torch::zeros_like(y);

    // One block per batch element; up to threads_per_seq threads per block.
    // Cap at n so we never launch more threads than elements (also satisfies
    // the hardware maximum of 1024 threads/block as long as threads_per_seq
    // <= 1024).
    const int threads_per_seq = 1024;
    const int tpb  = std::min((int64_t)threads_per_seq, n);  // handles n < 1024
    const int blocks = batch;   // one block = one sequence

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "isotonic_l2_parallel", ([&] {
        isotonic_l2_parallel_kernel<scalar_t><<<blocks, tpb>>>(
            y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sums.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            target.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            c.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            n,
            batch);
    }));

    return sol;
}

// Solves isotonic optimization with KL divergence using PAV.
// Formally, it solves argmin_{v_1 >= ... >= v_n} <e^{y-v}, 1> + <e^w, v>.
torch::Tensor isotonic_kl(torch::Tensor y, torch::Tensor w) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(y));
    auto batch = y.size(0);
    auto n = y.size(1);
    auto sol = torch::zeros_like(y);
    auto lse_y_ = torch::zeros_like(y);
    auto lse_w_ = torch::zeros_like(y);
    auto target = torch::zeros_like(y);

    const int threads = 1024;
    const int blocks = (batch + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "isotonic_kl", ([&] {
        isotonic_kl_kernel<scalar_t><<<blocks, threads>>>(
            y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            w.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            lse_y_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            lse_w_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            target.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            n,
            batch);
    }));
    return sol;
}

torch::Tensor isotonic_l2_backward(torch::Tensor s, torch::Tensor sol, torch::Tensor grad_input) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(s));
    auto batch = sol.size(0);
    auto n = sol.size(1);
    auto ret = torch::zeros_like(sol);
    auto sizes = torch::zeros_like(sol);

    const int threads = 1024;
    const int blocks = (batch + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sol.scalar_type(), "isotonic_l2_backward", ([&] {
        isotonic_l2_backward_kernel<scalar_t><<<blocks, threads>>>(
            s.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            ret.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sizes.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            n,
            batch);
    }));
    return ret;
}

torch::Tensor isotonic_l2_backward_parallel(torch::Tensor s, torch::Tensor sol, torch::Tensor grad_input) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(sol));
    const int B = sol.size(0);
    const int N = sol.size(1);

    const int tpb    = std::min((int64_t)1024, (int64_t)N);
    const int blocks = B;

    auto ret          = torch::zeros_like(sol);
    auto pid_buf      = torch::empty_like(sol);
    auto pid_copy_buf = torch::empty_like(sol);

    // float32 for accumulation
    auto float_opts = torch::dtype(torch::kFloat32).device(sol.device());
    auto psum_buf   = torch::zeros({B, N}, float_opts);
    auto pcnt_buf   = torch::zeros({B, N}, float_opts);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sol.scalar_type(), "isotonic_l2_backward_parallel", ([&] {
        isotonic_l2_backward_parallel_kernel<scalar_t><<<blocks, tpb>>>(
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            ret.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            pid_buf.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            pid_copy_buf.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            psum_buf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            pcnt_buf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            N
        );
    }));

    return ret;
}

torch::Tensor isotonic_kl_backward(torch::Tensor s, torch::Tensor sol, torch::Tensor grad_input) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(s));
    auto batch = sol.size(0);
    auto n = sol.size(1);
    auto ret = torch::zeros_like(sol);
    auto sizes = torch::zeros_like(sol);

    const int threads = 1024;
    const int blocks = (batch + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sol.scalar_type(), "isotonic_kl_backward", ([&] {
        isotonic_kl_backward_kernel<scalar_t><<<blocks, threads>>>(
            s.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sol.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            ret.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sizes.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            n,
            batch);
    }));
    return ret;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("isotonic_l2", &isotonic_l2, "Isotonic L2");
  m.def("isotonic_l2_backward", &isotonic_l2_backward, "Isotonic L2 Backward");
  m.def("isotonic_l2_parallel", &isotonic_l2_parallel, "Isotonic L2 Parallel");
  m.def("isotonic_l2_backward_parallel", &isotonic_l2_backward_parallel, "Isotonic L2 Backward Parallel");
  m.def("isotonic_kl", &isotonic_kl, "Isotonic KL");
  m.def("isotonic_kl_backward", &isotonic_kl_backward, "Isotonic KL Backward");
}
