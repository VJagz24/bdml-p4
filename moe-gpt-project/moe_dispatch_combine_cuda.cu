#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
//check to ensure tensors are cuda and contiguous 
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*
dispatch kernel is below
essentially here we implement the dispatch phase of Top-K moe 
so what happens is that each token k selects k experts
each exper gets a copy of token features
tokens are placed into per expert buffers with a cpacity
if capacity is reached extra tokens are dropped 
inputs: x_flat: [S, C],topk_idx: [S, k], topk_vals: [S, k]
expert_inputs: [E * capacity, C], expert_gates:  [E * capacity]
expert_indices:[E * capacity], expert_counts: [E] (
*/

__global__ void moe_dispatch_kernel(
    const float* __restrict__ x_flat,
    const long* __restrict__ topk_idx,
    const float* __restrict__ topk_vals,
    float* __restrict__ expert_inputs,
    float* __restrict__ expert_gates,
    int* __restrict__ expert_indices,
    int* __restrict__ expert_counts,
    int S, int C, int k, int E, int capacity
) {
    //token index and top-k entry index respectively 
    int t = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    //index check
    if (t >= S || j >= k) return;
    //load expert index 
    int64_t e_long = topk_idx[t * k + j];
    if (e_long < 0 || e_long >= E) {
        return;
    }
    int e = (int)e_long;

    // claims slot in buffer for the specific expert and token
    int slot = atomicAdd(&expert_counts[e], 1);
    if (slot >= capacity) {
        return;
    }

    int global_slot = e * capacity + slot;

    //store gate and original token index
    expert_gates[global_slot]  = topk_vals[t * k + j];
    expert_indices[global_slot] = t;

    //copy x_flat[t, :] to expert_inputs[global_slot, :]
    const float* x_row = x_flat + (size_t)t * C;
    float* dst_row = expert_inputs + (size_t)global_slot * C;

    for (int c = 0; c < C; ++c) {
        dst_row[c] = x_row[c];
    }
}


/*
below is the combine kernel which implements the combine phase
essentially each expert produces outputs for its slots
outputs are multiplied by gate vals already
each slot remembers where original token came from
so its accumulated back into order so transformer works
inputs:expert_outputs: [E * capacity, C] , expert_indices: [E * capacity]
y_flat: [S, C]
attomic add is used since multipl experts may attribute to same token
*/
__global__ void moe_combine_kernel(
    const float* __restrict__ expert_outputs,
    const int* __restrict__ expert_indices,
    const int* __restrict__ expert_counts,
    float* __restrict__ y_flat,
    int S, int C, int E, int capacity
) {
    //global slot index and feature dimension index respectively 
    int g = blockIdx.y * blockDim.y + threadIdx.y; 
    int c = blockIdx.x * blockDim.x + threadIdx.x; 

    int total_slots = E * capacity;
    if (g >= total_slots || c >= C) return;
    //get which expert and slot this corresponds to 
    int e = g / capacity;
    int slot = g % capacity;
    int count_e = expert_counts[e];
    //ignore inactive slots
    if (slot >= count_e) {
        return; // inactive slot
    }
    //get tokens original index
    int t = expert_indices[g];
    if (t < 0 || t >= S) {
        return;
    }
    //get gated expert val
    const float* src_row = expert_outputs + (size_t)g * C;
    float val = src_row[c];

    // accumulate into y_flat[t, c]
    atomicAdd(&y_flat[(size_t)t * C + c], val);
}

/*
C++ launch wrappers where these functions
validate tensors, allocate buffers, configure cuda blocks and grid dimensions
launch kernels and check fo cuda errors
*/
//first we have the dispatch launcher
std::vector<at::Tensor> moe_dispatch_cuda(
    at::Tensor x_flat,    // [S, C]
    at::Tensor topk_idx,  // [S, k]
    at::Tensor topk_vals, // [S, k], 
    int num_experts,
    int capacity
) {
    //validates inputs
    CHECK_INPUT(x_flat);
    CHECK_INPUT(topk_idx);
    CHECK_INPUT(topk_vals);

    TORCH_CHECK(x_flat.dim() == 2, "x_flat must be [S, C]");
    TORCH_CHECK(topk_idx.sizes() == topk_vals.sizes(), "topk_idx and topk_vals must have same shape");
    TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must be [S, k]");
    //gets dimensions from inputs
    int64_t S = x_flat.size(0);
    int64_t C = x_flat.size(1);
    int64_t k = topk_idx.size(1);
    int E = num_experts;
    //tensor operations
    auto opts_f = x_flat.options();
    auto opts_i = x_flat.options().dtype(torch::kInt32);
    //allocate output buffers
    at::Tensor expert_inputs  = at::empty({E * capacity, C}, opts_f);
    at::Tensor expert_gates   = at::empty({E * capacity}, opts_f);
    at::Tensor expert_indices = at::empty({E * capacity}, opts_i);
    at::Tensor expert_counts  = at::zeros({E}, opts_i);
    //2d grid where x is over top_k and y is over tokens 
    dim3 block(16, 16);
    dim3 grid(
        (k + block.x - 1) / block.x,
        (S + block.y - 1) / block.y
    );
    //klaunch the kernel
    moe_dispatch_kernel<<<grid, block>>>(
        x_flat.data_ptr<float>(),
        topk_idx.data_ptr<long>(),
        topk_vals.data_ptr<float>(),
        expert_inputs.data_ptr<float>(),
        expert_gates.data_ptr<float>(),
        expert_indices.data_ptr<int>(),
        expert_counts.data_ptr<int>(),
        (int)S, (int)C, (int)k, E, capacity
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "moe_dispatch_kernel failed: ", cudaGetErrorString(err));

    return {expert_inputs, expert_gates, expert_indices, expert_counts};
}
//below is the combine kernel launcher
at::Tensor moe_combine_cuda(
    at::Tensor expert_outputs, // [E*capacity, C]
    at::Tensor expert_indices, // [E*capacity]
    at::Tensor expert_counts,  // [E]
    int S, int C, int E, int capacity
) {
    //again validates the inputs
    CHECK_INPUT(expert_outputs);
    CHECK_INPUT(expert_indices);
    CHECK_INPUT(expert_counts);
    //allocate output tensors
    auto opts_f = expert_outputs.options();
    at::Tensor y_flat = at::zeros({S, C}, opts_f);
    //2d grid x dimension over feature dimension C and y over global expert slot
    dim3 block(16, 16);
    dim3 grid(
        (C + block.x - 1) / block.x,
        (E * capacity + block.y - 1) / block.y
    );
    //launch actual kernel
    moe_combine_kernel<<<grid, block>>>(
        expert_outputs.data_ptr<float>(),
        expert_indices.data_ptr<int>(),
        expert_counts.data_ptr<int>(),
        y_flat.data_ptr<float>(),
        S, C, E, capacity
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "moe_combine_kernel failed: ", cudaGetErrorString(err));

    return y_flat;
}