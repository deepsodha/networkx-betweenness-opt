from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "betweenness_core",
    sources=["betweenness_core.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-march=native"],
)

setup(
    name="betweenness_core",
    ext_modules=cythonize([ext], compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
    }),
)
