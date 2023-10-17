import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

os.environ["CC"] = "gcc"

include_path = [np.get_include()]

# Specify the include path for NumPy headers
include_path = [np.get_include()]

extensions = [
    Extension("knn_cython", ["knn_cython.pyx"], include_dirs=include_path),
]

setup(
    ext_modules=cythonize(extensions,annotate=True, language_level="3"),
)