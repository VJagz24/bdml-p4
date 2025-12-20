// moe_ffn_cpp.cpp
/*
essentially this is the C++ bridge between python and create cuda kernels
we declare cuda launchers implemented from cuda file, define c++ wrappers to call those
launchers and then register wrappers so it can be used in python 
*/
#include <torch/extension.h>//pytoch C++ extension API 
#include <vector>

/*
forward declaration of cuda launchers
needed so compiler knows the signatures when we bind them into python module 
output: y: final output [N, Dout], h: [N, Dhid](pre activation), a: [N, Dhid] (post activation)
*/
std::vector<at::Tensor> moe_ffn_forward_cuda(
    at::Tensor x, //[N, Din]
    at::Tensor W1, //[Din, Dhid]
    at::Tensor b1, //[Dhid]
    at::Tensor W2, //[Dhid, Dout]
    at::Tensor b2 //[Dout]
);

//below is the backward pass launcher 
//ouputs: {dx, dW1, db1, dW2, db2}
std::vector<at::Tensor> moe_ffn_backward_cuda(
    at::Tensor dy, //upstream grad w.r.t y: [N, Dout]
    at::Tensor x, //[N, Din]
    at::Tensor h, //[N, Dhid]
    at::Tensor a, //  [N, Dhid]
    at::Tensor W1, //[Din, Dhid]
    at::Tensor W2 //[Dhid, Dout]
);

/*
c++ wrapper functions to forward to the cuda implementations
below is the forward and backward wrappers
assume inputs are cuda tensors which is checked in actual fucntions
*/
std::vector<at::Tensor> moe_ffn_forward(
    at::Tensor x,
    at::Tensor W1,
    at::Tensor b1,
    at::Tensor W2,
    at::Tensor b2
) {
    return moe_ffn_forward_cuda(x, W1, b1, W2, b2);
}

std::vector<at::Tensor> moe_ffn_backward(
    at::Tensor dy,
    at::Tensor x,
    at::Tensor h,
    at::Tensor a,
    at::Tensor W1,
    at::Tensor W2
) {
    return moe_ffn_backward_cuda(dy, x, h, a, W1, W2);
}

// PYBIND11 module to expose 2 functions, namely forward and backward, to python so it can be used 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &moe_ffn_forward, "MoE FFN forward (CUDA)");
    m.def("backward", &moe_ffn_backward, "MoE FFN backward (CUDA)");
}