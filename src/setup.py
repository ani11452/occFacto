from setuptools import setup, find_packages
import os
path = os.path.join(os.path.dirname(__file__),"occFacto")

setup(
    name="occFacto",
    packages=find_packages(path),
    package_dir={'': "occFacto"},
)