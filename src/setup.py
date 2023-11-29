from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os

# Get include directory
numpy_include_dir = numpy.get_include()

# Define the extension module for mcubes
mcubes_module = Extension(
    'im2mesh.utils.libmcubes.mcubes',
    sources=[
        'python/occFacto/utils/libmcubes/mcubes.pyx',
        'python/occFacto/utils/libmcubes/pywrapper.cpp',
        'python/occFacto/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# Define the path
path = os.path.join(os.path.dirname(__file__), "python")
print(path)

# Setup configuration
setup(
    name="occFacto",
    packages=find_packages(path),
    package_dir={'': "python"},
    ext_modules=cythonize(mcubes_module),
    cmdclass={
        'build_ext': BuildExtension
    }
)
