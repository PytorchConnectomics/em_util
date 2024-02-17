import os, sys
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages
import numpy as np

def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName), np.get_include()]

def getExtension():
    extensions = []
    extensions += [Extension(
            'emu.seg.region_graph',
            sources=['emu/seg/region_graph.pyx', 'emu/seg/cpp/region_graph.cpp'],
            language='c++',
            extra_link_args=['-std=c++11'],
            extra_compile_args=['-std=c++11'])]
    return extensions

def setup_package(no_cython=True):
    __version__ = '0.1'
    url = 'https://github.com/donglaiw/emutil'

    exts = [] 
    package_data = {}
    if not no_cython:
        from Cython.Build import cythonize
        exts = cythonize(getExtension())
        package_data = {'': [
                'emu/seg/cpp/*.h',
                'emu/seg/cpp/*.cpp',
                'emu/seg/*.pyx',
            ]}

    setup(name='emu',
        description='Utility Functions for Image Analysis',
        version=__version__,
        url=url,
        license='MIT',
        author='Donglai Wei',
        install_requires=['scipy','numpy','networkx','h5py','imageio', 'skimage','tqdm'],
        include_dirs=getInclude(), 
        packages=find_packages(),
        package_data=package_data,
        ext_modules=exts
    )

if __name__=='__main__':
    # install main python functions
    # pip install --editable .
    # compile extra cython functions
    # python setup.py build_ext --inplace
    no_cython = True
    if 'build_ext' in sys.argv:
        no_cython = False
    setup_package(no_cython)
