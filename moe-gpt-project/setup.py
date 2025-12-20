from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="moe_ffn_ext",
    ext_modules=[
        CUDAExtension(
            name="moe_ffn_ext",
            sources=[
                "moe_ffn_cpp.cpp",
                "moe_ffn_cuda.cu",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)