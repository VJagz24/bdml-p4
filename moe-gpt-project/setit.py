'''
allows us build and install custom pytorch cuda extension where setuptools compiles
c++ and cuda into python usable package
'''
from setuptools import setup
#pytorch helpers to compile extensions to integrate easily within pytorch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    #package name for the extension
    name="moe_dc_ext",
    ext_modules=[
        CUDAExtension(
            #name of module to import into python
            "moe_dc_ext",
            #what to compile 
            sources=[
                "moe_dispatch_combine_cpp.cpp",
                "moe_dispatch_combine_cuda.cu",
            ],
        )
    ],
    #overrides setuptools with pytorch for compatibility
    cmdclass={"build_ext": BuildExtension},
)