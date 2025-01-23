import os
from setuptools import setup, Extension, find_packages
import pybind11

# Helper function to get the include paths for pybind11
def get_pybind_include():
    return pybind11.get_include()

# Define the C++ extensions
ext_modules = [
    Extension(
        'vbpca_py.errpca_pt',  # Module name
        ['src/vbpca_py/errpca_pt.cpp'],  # Source file
        include_dirs=[
            get_pybind_include(),  # pybind11 include path
            get_pybind_include(),  # User-specific pybind11 include path
            os.path.abspath('src/vbpca_py/eigen/eigen-3.4.0')
        ],
        language='c++',
        extra_compile_args=['-O3', '-std=c++14'],  # Optimization and C++ standard
    ),
    Extension(
        'vbpca_py.subtract_mu_from_sparse',  # Module name
        ['src/vbpca_py/subtract_mu_from_sparse.cpp'],  # Source file
        include_dirs=[
            get_pybind_include(),  # pybind11 include path
            get_pybind_include()  # User-specific pybind11 include path
        ],
        language='c++',
        extra_compile_args=['-O3', '-std=c++11'],  # Optimization and C++ standard
    )
]

# Setup configuration
setup(
    name='vbpca_py',
    version='0.1.0',
    description='A Python package with C++ extensions for PCA',
    author='Shany',
    author_email='shany215.sn@gmail.com',
    packages=find_packages(where="src"),  # Locate Python packages under "src"
    package_dir={"": "src"},  # Set the root directory for the packages
    ext_modules=ext_modules,  # Add the C++ extensions
    install_requires=['pybind11', 'numpy', 'scipy'],  # Dependencies
    zip_safe=False,  # Avoid setuptools packaging the project into .egg files
)
