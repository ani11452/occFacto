from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy

numpy_include_dir = numpy.get_include()

# Extensions setup
triangle_hash_module = Extension(
    'libmesh.triangle_hash',
    sources=['libmesh/triangle_hash.pyx'],
    libraries=['m'], # Unix-like specific
    include_dirs=[numpy_include_dir]  
)

ext_modules = [
    triangle_hash_module,
]

setup(
    name='libmesh',  # This is the name you'll use for pip install
    version='1.0.0',  # Update the version number for new releases
    description='A brief description of libmesh',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules, language_level="3"),
    cmdclass={'build_ext': BuildExtension},
)

