from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup, find_packages
import os
import torch
from torch.utils.cpp_extension import include_paths

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')

setup(
    name='renderutils_plugin',
    ext_modules=[
        CUDAExtension(
            'renderutils_plugin',
            sources=['c_src/torch_bindings.cpp', 'c_src/bsdf.cu', 'c_src/cubemap.cu', 'c_src/loss.cu', 'c_src/mesh.cu', 'c_src/normal.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            },
            library_dirs=[torch_lib_dir],
            extra_link_args=[f'-Wl,-rpath,{torch_lib_dir}']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)