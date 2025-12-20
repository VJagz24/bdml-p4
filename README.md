# Modded-nanoGPT with MoE

Comparing standard GPT vs. MoE architectures at 5M, 40M, 124M, and 350M parameter scales on WikiText-103.

## 1) MegaBlocks MoE (`megablocks-moe/`)

This implementation uses **MegaBlocks** for efficient MoE with 8 experts and top-2 routing.

A complete Colab notebook (`Baseline_vs_Megablocks_MoE_Comparison.ipynb`) is included with all commands and configurations.

## Installation

```bash
pip install torch transformers datasets megablocks
pip install git+https://github.com/tgale96/grouped_gemm@main
```

## How to Run

### 1. Clone Repository
```bash
git clone https://github.com/VJagz24/bdml-p4.git
cd bdml-p4/megablocks-moe
```

### 2. Configure Model Size

Use sed commands to quickly configure for different model sizes:

**5M Model:**
```bash
sed -i 's/n_layer = .*/n_layer = 4/g' train_baseline.py train_moe.py
sed -i 's/n_head = .*/n_head = 4/g' train_baseline.py train_moe.py
sed -i 's/n_embd = .*/n_embd = 256/g' train_baseline.py train_moe.py
sed -i 's/block_size = .*/block_size = 256/g' train_baseline.py train_moe.py
sed -i 's/max_seq_length = .*/max_seq_length = 256/g' train_baseline.py train_moe.py
sed -i 's/max_iters = .*/max_iters = 5000/g' train_baseline.py train_moe.py
sed -i 's/warmup_iters = .*/warmup_iters = 500/g' train_baseline.py train_moe.py
```

**40M Model:**
```bash
sed -i 's/n_layer = .*/n_layer = 8/g' train_baseline.py train_moe.py
sed -i 's/n_head = .*/n_head = 8/g' train_baseline.py train_moe.py
sed -i 's/n_embd = .*/n_embd = 512/g' train_baseline.py train_moe.py
sed -i 's/block_size = .*/block_size = 512/g' train_baseline.py train_moe.py
sed -i 's/max_seq_length = .*/max_seq_length = 512/g' train_baseline.py train_moe.py
sed -i 's/max_iters = .*/max_iters = 5000/g' train_baseline.py train_moe.py
sed -i 's/warmup_iters = .*/warmup_iters = 500/g' train_baseline.py train_moe.py
```

**124M Model:**
```bash
sed -i 's/n_layer = .*/n_layer = 12/g' train_baseline.py train_moe.py
sed -i 's/n_head = .*/n_head = 12/g' train_baseline.py train_moe.py
sed -i 's/n_embd = .*/n_embd = 768/g' train_baseline.py train_moe.py
sed -i 's/block_size = .*/block_size = 1024/g' train_baseline.py train_moe.py
sed -i 's/max_seq_length = .*/max_seq_length = 1024/g' train_baseline.py train_moe.py
sed -i 's/max_iters = .*/max_iters = 5000/g' train_baseline.py train_moe.py
sed -i 's/warmup_iters = .*/warmup_iters = 500/g' train_baseline.py train_moe.py
```

**350M Model:**
```bash
sed -i 's/n_layer = .*/n_layer = 24/g' train_baseline.py train_moe.py
sed -i 's/n_head = .*/n_head = 16/g' train_baseline.py train_moe.py
sed -i 's/n_embd = .*/n_embd = 1024/g' train_baseline.py train_moe.py
sed -i 's/block_size = .*/block_size = 1024/g' train_baseline.py train_moe.py
sed -i 's/max_seq_length = .*/max_seq_length = 1024/g' train_baseline.py train_moe.py
sed -i 's/max_iters = .*/max_iters = 5000/g' train_baseline.py train_moe.py
sed -i 's/warmup_iters = .*/warmup_iters = 500/g' train_baseline.py train_moe.py
```

**Configure MoE-specific settings:**
```bash
# Set top-k routing to 2 (only for train_moe.py)
sed -i 's/top_k = .*/top_k = 2/g' train_moe.py

# Reduce MoE batch size for memory (only for train_moe.py)
sed -i 's/batch_size = .*/batch_size = 4/g' train_moe.py
sed -i 's/gradient_accumulation_steps = .*/gradient_accumulation_steps = 2/g' train_moe.py

# Fix cache loading
sed -i 's/torch.load(train_cache_file)/torch.load(train_cache_file, weights_only=False)/g' train_baseline.py train_moe.py
sed -i 's/torch.load(val_cache_file)/torch.load(val_cache_file, weights_only=False)/g' train_baseline.py train_moe.py
```

### 3. Run Training

```bash
# Train baseline model
python train_baseline.py

# Train MoE model
python train_moe.py
```
## 2) Cuda Kernels MoE (`moe-gpt-project/`)

This directory includes the 2 CUDA kernel implementations. The layout is as such:
-moe-gpt-project:
      -main.ipynb: python notebook to run the kernel implementations
      -model.py : model implementation for CUDA Kernel 1: End-to-End Scatter–Compute–Gather Kernel
      -model_kernel.py: model implementation for CUDA Kernel 2: End-to-End Scat- ter–Compute–Gather Kernel
      CUDA Kernel 1 implementation:
          -model_combine.py
          -moe_dispatch_combine_cpp.cpp
          -moe_dispatch_combine_cuda.cu
          -setit.py
      CUDA Kernel 2 implementation: 
          -moe_ffn.py
          -moe_ffn_cpp.cpp
          -moe_ffn_cuda.cu
          -setup.py
      -train_moe.py: training script
There are 2 ways to reproduce the CUDA Kernel 1 results:
Method 1: 
    -Clone the repository in a collab
    -open up the project directory 
    -open the main.ipynb file
    -connect to A100 GPU and run all cells 
Method 2: 
    -Download moe-gpt-project directly
    -Put it in google drive
    -Open main.ipynb in collab
    -Connect to A100  GPU and run all cells
As mentioned CUDA Kernel 2 was not used for results since when initially tested it suffered much more degrading performance so the primary focus became CUDA Kernel 1.
If you want to produce results for CUDA kernel 2 you follow the same steps as in Method 1 or Method 2. But you change the name of model_kernel.py in the moe-gpt-project folder to model.py and vice versa. 



Results will be saved in `out-baseline/` and `out-moe/` directories.
