// moe_dispatch_combine_cpp.cpp
/*
c++ integration between pytorch and cuda
declares cuda implementations, exposes c++ wrappers
and uses pybind 11 to register functions to be used within python
*/
#include <torch/extension.h>
#include <vector>
/*
forward declarations of cuda functions
below is the dispatch kernel that routes token represnetations
to expert buffers from top-k routing results
outputs: expert_inputs, expert_gates, expert_indicies, expert__counts
*/
std::vector<at::Tensor> moe_dispatch_cuda(
    at::Tensor x_flat, //[S, C] flattened token representations
    at::Tensor topk_idx, //[S, k] expert indices per token
    at::Tensor topk_vals,//[S, k] routing weights per token
    int num_experts,
    int capacity
);
/*
below is for the combine kernel which scatters expert outputs into
original token order
outputs: tensor of shape [S, C]
*/
at::Tensor moe_combine_cuda(
    at::Tensor expert_outputs, //[E * capacity, C]
    at::Tensor expert_indices, // mapping from expert slots → original token indices
    at::Tensor expert_counts, //number of tokens routed to each expert
    int S, int C, int E, int capacity
);
/*
c++ wrappers to forward args to cuda implementations 
below we have dispatch and comine wrappers
*/
std::vector<at::Tensor> moe_dispatch(
    at::Tensor x_flat,
    at::Tensor topk_idx,
    at::Tensor topk_vals,
    int num_experts,
    int capacity
) {
    return moe_dispatch_cuda(x_flat, topk_idx, topk_vals, num_experts, capacity);
}

at::Tensor moe_combine(
    at::Tensor expert_outputs,
    at::Tensor expert_indices,
    at::Tensor expert_counts,
    int S, int C, int E, int capacity
) {
    return moe_combine_cuda(expert_outputs, expert_indices, expert_counts, S, C, E, capacity);
}
//pytorch module registering to expose functions to be used in python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch", &moe_dispatch, "MoE dispatch (CUDA)");
    m.def("combine",  &moe_combine,  "MoE combine (CUDA)");
}