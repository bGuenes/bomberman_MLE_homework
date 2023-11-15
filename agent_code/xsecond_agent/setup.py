from distutils.core import setup
from Cython.Build import cythonize

setup(name="state_to_features", ext_modules=cythonize("state_to_features.py"))