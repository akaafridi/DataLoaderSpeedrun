from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Setup the extension
setup(
    name='fastfetch',
    ext_modules=[
        CppExtension(
            'fastfetch',
            ['fastfetch.cpp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
