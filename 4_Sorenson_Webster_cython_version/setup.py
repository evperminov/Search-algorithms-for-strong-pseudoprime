from setuptools import Extension, setup
from Cython.Build import cythonize
from glob import glob
import sys

extensions = []
extra_compile_args = ['-std=c++11']
extra_link_args = ['-lprimesieve']
include_dirs=["lib/primesieve/include", *sys.path]

extensions.append(Extension(
    "Sorenson_Webster_cython",
    ["Sorenson_Webster_cython_multiprocess.pyx"] + #  ["Sorenson_Webster_cython.pyx"] 
    glob("lib/primesieve/src/*.cpp") +
    glob("lib/primesieve/src/primesieve/*.cpp"),
    include_dirs=include_dirs,
    libraries=['gmp'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
    ))

ext_modules = cythonize(extensions, include_path=include_dirs, compiler_directives={'embedsignature': True,}, annotate=True)

setup(
    ext_modules=ext_modules
)


# python setup.py build_ext --inplace