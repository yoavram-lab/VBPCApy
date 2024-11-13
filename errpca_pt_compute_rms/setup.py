# setup.py

from setuptools import setup, Extension
import pybind11
from setuptools.command.build_ext import build_ext

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path.

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'errpca_pt',
        ['errpca_pt.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            '/opt/homebrew/opt/eigen/include/eigen3'  # Update this path based on your Eigen installation
        ],
        language='c++',
        extra_compile_args=['-O3', '-std=c++14'],
        extra_link_args=[]
    ),
]

setup(
    name='errpca_pt',
    version='0.1',
    author='Your Name',
    author_email='your.email@example.com',
    description='ERRPCA_PT: Compute sparse matrix of reconstruction errors',
    ext_modules=ext_modules,
    install_requires=['pybind11', 'numpy', 'scipy'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
