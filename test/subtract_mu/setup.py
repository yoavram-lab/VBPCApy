# setup.py

import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path."""
    def __init__(self, user=False):
        self.user = user
    def __str__(self):
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'subtract_mu_from_sparse',          # Name of the Python module
        ['subtract_mu_from_sparse.cpp'],                # Source file
        include_dirs=[
            get_pybind_include(),            # Path to pybind11 headers
            get_pybind_include(user=True)
        ],
        language='c++',
        extra_compile_args=['-O3', '-std=c++11'],  # Optimization and C++ standard
    ),
]

setup(
    name='subtract_mu_from_sparse',
    version='0.0.1',
    author='Your Name',
    author_email='your.email@example.com',
    description='Subtract Mu from a sparse matrix X using C++ and pybind11',
    ext_modules=ext_modules,
    install_requires=['pybind11', 'numpy', 'scipy'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
