from setuptools import setup, Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension
import numpy

# Uncomment these lines
numpy_include_dir = numpy.get_include()
print("NumPy include directory:", numpy_include_dir)

triangle_hash_module = Extension(
    'external.libmesh.triangle_hash',
    sources=[
        'external/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]  # Add this line
)

ext_modules = [
    triangle_hash_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)


# try:
#     from setuptools import setup
# except ImportError:
#     from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
# from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
# import numpy


# # Get the numpy include directory.
# # numpy_include_dir = numpy.get_include()
# # print(numpy.get_include())

# # Extensions
# # pykdtree (kd tree)
# # pykdtree = Extension(
# #     'external.libkdtree.pykdtree.kdtree',
# #     sources=[
# #         'external/libkdtree/pykdtree/kdtree.c',
# #         'external/libkdtree/pykdtree/_kdtree_core.c'
# #     ],
# #     language='c',
# #     extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
# #     extra_link_args=['-lgomp'],
# # )

# # mcubes (marching cubes algorithm)
# # mcubes_module = Extension(
# #     'external.libmcubes.mcubes',
# #     sources=[
# #         'external/libmcubes/mcubes.pyx',
# #         'external/libmcubes/pywrapper.cpp',
# #         'external/libmcubes/marchingcubes.cpp'
# #     ],
# #     language='c++',
# #     extra_compile_args=['-std=c++11'],
# #     include_dirs=[numpy_include_dir]
# # )

# # triangle hash (efficient mesh intersection)
# triangle_hash_module = Extension(
#     'external.libmesh.triangle_hash',
#     sources=[
#         'external/libmesh/triangle_hash.pyx'
#     ],
#     libraries=['m']  # Unix-like specific
# )

# # Gather all extension modules
# ext_modules = [
#     # pykdtree,
#     # mcubes_module,
#     triangle_hash_module,
# ]

# setup(
#     ext_modules=cythonize(ext_modules),
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )