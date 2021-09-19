import os, sys
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName)]

def getExtension():
    extensions = []
    extensions += [Extension(
            'emu.seg.region_graph',
            sources=['emu/seg/region_graph.pyx', 'emu/seg/cpp/region_graph.cpp'],
            language='c++',
            extra_link_args=['-std=c++11'],
            extra_compile_args=['-std=c++11'])]
    return extensions

def setup_package():
    __version__ = '0.1'
    url = 'https://github.com/donglaiw/EM-util'

    setup(name='emu',
        description='Utility Functions for EM Connectomics',
        version=__version__,
        url=url,
        license='MIT',
        author='Donglai Wei',
        install_requires=['cython','scipy','numpy','networkx','h5py','imageio'],
        include_dirs=getInclude(), 
        packages=find_packages(),
        package_data={
            '': [
                'emu/seg/cpp/*.h',
                'emu/seg/cpp/*.cpp',
                'emu/seg/*.pyx',
            ]
        },
        ext_modules=cythonize(getExtension())
    )

if __name__=='__main__':
    # pip install --editable .
    # python setup.py build_ext --inplace
    setup_package()
