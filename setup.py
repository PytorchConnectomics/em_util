import os, sys
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName), np.get_include()]

def getExtension():
    extensions = []
    extensions += [Extension(
            'imu.seg.region_graph',
            sources=['imu/seg/region_graph.pyx', 'imu/seg/cpp/region_graph.cpp'],
            language='c++',
            extra_link_args=['-std=c++11'],
            extra_compile_args=['-std=c++11'])]
    return extensions

def setup_package(no_cython=True):
    __version__ = '0.1'
    url = 'https://github.com/donglaiw/ImUtil'

    exts = [] 
    if not no_cython:
        exts = cythonize(getExtension())
    import pdb; pdb.set_trace()

    setup(name='imu',
        description='Utility Functions for Image Analysis',
        version=__version__,
        url=url,
        license='MIT',
        author='Donglai Wei',
        install_requires=['cython','scipy','numpy','networkx','h5py','imageio'],
        include_dirs=getInclude(), 
        packages=find_packages(),
        package_data={
            '': [
                'imu/seg/cpp/*.h',
                'imu/seg/cpp/*.cpp',
                'imu/seg/*.pyx',
            ]
        },
        ext_modules=exts
    )

if __name__=='__main__':
    # pip install --editable .
    # python setup.py build_ext --inplace
    # python setup.py build_ext --inplace --cython
    no_cython = True
    if '--cython' in sys.argv:
        no_cython = False
        sys.argv.remove("--cython")
    setup_package(no_cython)
