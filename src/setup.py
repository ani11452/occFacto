from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os

# # Get include directory
# numpy_include_dir = numpy.get_include()

# # Define the extension modeuls for pykdtree (kd tree)
# pykdtree = Extension(
#     'kdtree',
#     sources=[
#         'external/libkdtree/pykdtree/kdtree.c',
#         'external/libkdtree/pykdtree/_kdtree_core.c'
#     ],
#     language='c',
#     extra_compile_args=['-O3', '-fopenmp'],
#     extra_link_args=['-lgomp'],
# )

# # Define the extension module for mcubes
# mcubes_module = Extension(
#     'libmcubes.mcubes',
#     sources=[
#         'external/libmcubes/mcubes.pyx',
#         'external/libmcubes/pywrapper.cpp',
#         'external/libmcubes/marchingcubes.cpp'
#     ],
#     language='c++',
#     extra_compile_args=['-std=c++11'],
#     include_dirs=[numpy_include_dir]
# )

# # Define the extension module for libmesh
# triangle_hash_module = Extension(
#     'libmesh.triangle_hash',
#     sources=[
#         'external/libmesh/triangle_hash.pyx'
#     ],
#     libraries=['m']  # Unix-like specific
# )

# ext_modules = [
#     pykdtree,
#     mcubes_module,
#     triangle_hash_module,
# ]

# Define the path
path = os.path.join(os.path.dirname(__file__), "python")
print(path)

# Setup configuration
setup(
    name="occFacto",
    packages=find_packages(path),
    package_dir={'': "python"},
    # ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
