from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quantile_pooling_cuda',
    version='0.2',
    ext_modules=[
        CUDAExtension('quantile_pooling_cuda', [
            'quantile_pooling_cuda/quantile_pooling.cpp',
            'quantile_pooling_cuda/quantile_pooling_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=['quantile_pooling'],
)
