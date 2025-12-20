// moe_ffn_cuda.cu
/*
Cuda Implementation of a 2 layer FFN with GELU
Essentially: 
Forward:  h = x @ W1 + b1
          a = GELU(h)
          y = a @ W2 + b2
Backward: computes gradients for x, W1, b1, W2, b2 given dy.
*/
#include <torch/extension.h>//Pytorch extension API
#include <cuda_runtime.h>//Cuda runtime for lerma launches, errors, etc
#include <vector>
#include <cmath>
//intital checks to make sure tensors re of the right format: cuda and contiguous
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
//validates input tensors in launchers
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//device GELU and derivative operations pretty trivial
//for forward and backward pass
__device__ inline float gelu(float x) {
    const float kAlpha = M_SQRT1_2; // 1/sqrt(2)
    return 0.5f * x * (1.0f + erff(kAlpha * x));
}
//
__device__ inline float gelu_derivative(float x) {
    const float kAlpha =M_SQRT1_2;
    const float rsqrt_2pi= 0.3989422804014327f; // 1/sqrt(2*pi)
    float x_alpha =kAlpha * x;
    float erf_term= erff(x_alpha);
    float exp_term= expf(-0.5f * x * x);
    float grad= 0.5f * (1.0f + erf_term)
               + 0.5f * x * rsqrt_2pi * exp_term;
    return grad;
}

//below are implementations for the kernels of the forward pass


/*
layer 1 forward GEMM + bias + GELU
Inputs: x: [N, Din], W1: [Din, Dhid], b1: [Dhid]
outputs: h: [N, Dhid](pre activation), a: [N, Dhid](post activation)
we use thread mapping as learned in class in this context though
threadIdx.x is the hidden dimension of index j and threadidx.y is token index n
*/
__global__ void moe_ffn_layer1_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    float* __restrict__ h,
    float* __restrict__ a,
    int N, int Din, int Dhid
) {
    //specific indicies per thread 
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int n = blockIdx.y * blockDim.y + threadIdx.y; 
    //index check 
    if (j >= Dhid || n >= N) return;
    //creates pointer to n-th row of x and creates accumulation var starting with bias
    const float* x_row = x + n * Din;
    float sum = b1[j];

    for (int i = 0; i < Din; ++i) {
        sum += x_row[i] * W1[i * Dhid + j];
    }

    float h_val = sum;
    float a_val = gelu(h_val);

    h[n * Dhid + j] = h_val;
    a[n * Dhid + j] = a_val;
}
/*
Layer 2 forward pass includes GEMM and bias
Inputs:a:  [N, Dhid], W2: [Dhid, Dout], b2: [Dout]
Output: y:  [N, Dout]
again uses similar thread mapping but in this context thread
calculates one (n, k ) output element
*/

__global__ void moe_ffn_layer2_forward_kernel(
    const float* __restrict__ a,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    float* __restrict__ y,
    int N, int Dhid, int Dout
) {
    //output and token index based of thread mapping
    int k = blockIdx.x * blockDim.x + threadIdx.x; 
    int n = blockIdx.y * blockDim.y + threadIdx.y; 
    //out of bounds check
    if (k >= Dout || n >= N) return;
    const float* a_row = a + n * Dhid;
    float sum = b2[k];

    for (int j = 0; j < Dhid; ++j) {
        sum += a_row[j] * W2[j * Dout + k];
    }

    y[n * Dout + k] = sum;
}

// below are now the backward kernel implementations 

/*
computes gradients for backpropagation to work correctly
inputs: dy:  [N, Dout], a:   [N, Dhid]
outputs: dW2: [Dhid, Dout],  db2: [Dout]
again similar thread mapping
*/

__global__ void moe_ffn_layer2_grad_wb_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ a,
    float* __restrict__ dW2,
    float* __restrict__ db2,
    int N, int Dhid, int Dout
) {
    //output and hidden index respectively 
    int k = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 

    if (k >= Dout || j >= Dhid) return;

    float grad_w = 0.0f;
    for (int n = 0; n < N; ++n) {
        float dy_val = dy[n * Dout + k];
        float a_val  = a[n * Dhid + j];
        grad_w += a_val * dy_val;
    }
    dW2[j * Dout + k] = grad_w;

    // db2: only once per k we arbitrarily chose when j=0
    if (j == 0) {
        float grad_b = 0.0f;
        for (int n = 0; n < N; ++n) {
            grad_b += dy[n * Dout + k];
        }
        db2[k] = grad_b;
    }
}

/*
calculates gradients w.r.t activation
inputs: dy: [N, Dout], W2: [Dhid, Dout]
output: dA: [N, Dhid]
thread context here is that each thread computs dA[n,j]
*/

__global__ void moe_ffn_layer2_grad_input_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ W2,
    float* __restrict__ dA,
    int N, int Dhid, int Dout
) {
    //hiddent and token index respectively
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    int n = blockIdx.y * blockDim.y + threadIdx.y; 

    if (j >= Dhid || n >= N) return;

    float sum = 0.0f;
    const float* dy_row = dy + n * Dout;
    const float* w_row  = W2 + j * Dout;

    for (int k = 0; k < Dout; ++k) {
        sum += dy_row[k] * w_row[k];
    }

    dA[n * Dhid + j] = sum;
}

/*
apply GELU to backprop 
inputs:  dA: [N, Dhid], h:  [N, Dhid], dG: [N, Dhid]
*/
__global__ void moe_ffn_layer1_grad_dG_kernel(
    const float* __restrict__ dA,
    const float* __restrict__ h,
    float* __restrict__ dG,
    int N, int Dhid
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= Dhid || n >= N) return;

    float da_val = dA[n * Dhid + j];
    float h_val  = h[n * Dhid + j];
    float gprime = gelu_derivative(h_val);

    dG[n * Dhid + j] = da_val * gprime;
}

/*
computes gradients for W1, b1
inputs: x:   [N, Din], dG:  [N, Dhid]
outputs: dW1: [Din, Dhid], db1: [Dhid]
*/

__global__ void moe_ffn_layer1_grad_wb_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dG,
    float* __restrict__ dW1,
    float* __restrict__ db1,
    int N, int Din, int Dhid
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // hidden idx
    int i = blockIdx.y * blockDim.y + threadIdx.y; // input idx

    if (j >= Dhid || i >= Din) return;

    float grad_w = 0.0f;
    for (int n = 0; n < N; ++n) {
        float x_val  = x[n * Din + i];
        float dG_val = dG[n * Dhid + j];
        grad_w += x_val * dG_val;
    }
    dW1[i * Dhid + j] = grad_w;

    // db1[j]: only once per j (i == 0)
    if (i == 0) {
        float grad_b = 0.0f;
        for (int n = 0; n < N; ++n) {
            grad_b += dG[n * Dhid + j];
        }
        db1[j] = grad_b;
    }
}

/*
compute gradient w.r.t x
inputs: dG: [N, Dhid], W1: [Din, Dhid]
outputs: dx: [N, Din] 
here each thread calculates dx[n, i]
*/

__global__ void moe_ffn_layer1_grad_input_kernel(
    const float* __restrict__ dG,
    const float* __restrict__ W1,
    float* __restrict__ dx,
    int N, int Din, int Dhid
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // input idx
    int n = blockIdx.y * blockDim.y + threadIdx.y; // token idx

    if (i >= Din || n >= N) return;

    float sum = 0.0f;
    const float* dG_row = dG + n * Dhid;
    const float* w_row  = W1 + i * Dhid;

    for (int j = 0; j < Dhid; ++j) {
        sum += dG_row[j] * w_row[j];
    }

    dx[n * Din + i] = sum;
}

/*
Launchers for C++ API 
essentially includes functions exposed to C++/Pytorch 
where functions validate, inputs, allocate tensors, configure, grids
and blocks, launch kernels and check for cuda errors
*/
std::vector<at::Tensor> moe_ffn_forward_cuda(
    at::Tensor x,   // [N, Din]
    at::Tensor W1,  // [Din, Dhid]
    at::Tensor b1,  // [Dhid]
    at::Tensor W2,  // [Dhid, Dout]
    at::Tensor b2   // [Dout]
) {
    //validate tensors on GPU
    CHECK_INPUT(x);
    CHECK_INPUT(W1);
    CHECK_INPUT(b1);
    CHECK_INPUT(W2);
    CHECK_INPUT(b2);
    //shape match checking to avoid future errors
    TORCH_CHECK(x.dim() == 2, "x must be [N, Din]");
    TORCH_CHECK(W1.dim() == 2, "W1 must be [Din, Dhid]");
    TORCH_CHECK(b1.dim() == 1, "b1 must be [Dhid]");
    TORCH_CHECK(W2.dim() == 2, "W2 must be [Dhid, Dout]");
    TORCH_CHECK(b2.dim() == 1, "b2 must be [Dout]");
    //dimensions from inputs
    const auto N    = x.size(0);
    const auto Din  = x.size(1);
    const auto Din2 = W1.size(0);
    const auto Dhid = W1.size(1);
    const auto Dhid2 = W2.size(0);
    const auto Dout  = W2.size(1);
    //check dimensions for matrix multiplication
    TORCH_CHECK(Din == Din2, "x.size(1) must equal W1.size(0)");
    TORCH_CHECK(Dhid == b1.size(0), "W1.size(1) must equal b1.size(0)");
    TORCH_CHECK(Dhid == Dhid2, "W1.size(1) must equal W2.size(0)");
    TORCH_CHECK(Dout == b2.size(0), "W2.size(1) must equal b2.size(0)");
    //allocate tensors for outputs
    auto options = x.options();
    at::Tensor h = at::empty({N, Dhid}, options);
    at::Tensor a = at::empty({N, Dhid}, options);
    at::Tensor y = at::empty({N, Dout}, options);
    //configure the kernel launches including grids and blocks
    dim3 block1(16, 16);
    dim3 grid1((Dhid + block1.x - 1) / block1.x,
               (N    + block1.y - 1) / block1.y);
    //layer 1 forward pass launch
    moe_ffn_layer1_forward_kernel<<<grid1, block1>>>(
        x.data_ptr<float>(),
        W1.data_ptr<float>(),
        b1.data_ptr<float>(),
        h.data_ptr<float>(),
        a.data_ptr<float>(),
        N, Din, Dhid
    );
    //layer 2 forward pass launch
    dim3 block2(16, 16);
    dim3 grid2((Dout + block2.x - 1) / block2.x,
               (N    + block2.y - 1) / block2.y);

    moe_ffn_layer2_forward_kernel<<<grid2, block2>>>(
        a.data_ptr<float>(),
        W2.data_ptr<float>(),
        b2.data_ptr<float>(),
        y.data_ptr<float>(),
        N, Dhid, Dout
    );
    //check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "moe_ffn_forward_cuda failed with error: ", cudaGetErrorString(err));

    return {y, h, a};
}

std::vector<at::Tensor> moe_ffn_backward_cuda(
    at::Tensor dy,  // [N, Dout]
    at::Tensor x,   // [N, Din]
    at::Tensor h,   // [N, Dhid]
    at::Tensor a,   // [N, Dhid]
    at::Tensor W1,  // [Din, Dhid]
    at::Tensor W2   // [Dhid, Dout]
) {
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    CHECK_INPUT(h);
    CHECK_INPUT(a);
    CHECK_INPUT(W1);
    CHECK_INPUT(W2);

    const auto N    = x.size(0);
    const auto Din  = x.size(1);
    const auto Dhid = h.size(1);
    const auto Dout = dy.size(1);

    auto options = x.options();

    at::Tensor dW1 = at::zeros_like(W1);
    at::Tensor db1 = at::zeros({Dhid}, options);
    at::Tensor dW2 = at::zeros_like(W2);
    at::Tensor db2 = at::zeros({Dout}, options);
    at::Tensor dx  = at::zeros_like(x);
    at::Tensor dA  = at::zeros_like(a);
    at::Tensor dG  = at::zeros_like(h);

    dim3 block_w2(16, 16);
    dim3 grid_w2((Dout + block_w2.x - 1) / block_w2.x,
                 (Dhid + block_w2.y - 1) / block_w2.y);

    moe_ffn_layer2_grad_wb_kernel<<<grid_w2, block_w2>>>(
        dy.data_ptr<float>(),
        a.data_ptr<float>(),
        dW2.data_ptr<float>(),
        db2.data_ptr<float>(),
        N, Dhid, Dout
    );

    dim3 block_da(16, 16);
    dim3 grid_da((Dhid + block_da.x - 1) / block_da.x,
                 (N    + block_da.y - 1) / block_da.y);

    moe_ffn_layer2_grad_input_kernel<<<grid_da, block_da>>>(
        dy.data_ptr<float>(),
        W2.data_ptr<float>(),
        dA.data_ptr<float>(),
        N, Dhid, Dout
    );

    dim3 block_dg(16, 16);
    dim3 grid_dg((Dhid + block_dg.x - 1) / block_dg.x,
                 (N    + block_dg.y - 1) / block_dg.y);

    moe_ffn_layer1_grad_dG_kernel<<<grid_dg, block_dg>>>(
        dA.data_ptr<float>(),
        h.data_ptr<float>(),
        dG.data_ptr<float>(),
        N, Dhid
    );

    dim3 block_w1(16, 16);
    dim3 grid_w1((Dhid + block_w1.x - 1) / block_w1.x,
                 (Din  + block_w1.y - 1) / block_w1.y);

    moe_ffn_layer1_grad_wb_kernel<<<grid_w1, block_w1>>>(
        x.data_ptr<float>(),
        dG.data_ptr<float>(),
        dW1.data_ptr<float>(),
        db1.data_ptr<float>(),
        N, Din, Dhid
    );

    dim3 block_dx(16, 16);
    dim3 grid_dx((Din + block_dx.x - 1) / block_dx.x,
                 (N   + block_dx.y - 1) / block_dx.y);

    moe_ffn_layer1_grad_input_kernel<<<grid_dx, block_dx>>>(
        dG.data_ptr<float>(),
        W1.data_ptr<float>(),
        dx.data_ptr<float>(),
        N, Din, Dhid
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "moe_ffn_backward_cuda failed with error: ", cudaGetErrorString(err));

    return {dx, dW1, db1, dW2, db2};
}
