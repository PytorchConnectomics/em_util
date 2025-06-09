import os
from distutils.sysconfig import get_python_inc
from setuptools import setup, find_packages

def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName)]

def setup_package():
    __version__ = '0.1'
    url = 'https://github.com/PytorchConnectomics/em_util'

    setup(name='em_util',
        description='Utility Functions for EM Image Analysis',
        version=__version__,
        url=url,
        license='MIT',
        author='Donglai Wei',
        install_requires=['scipy','numpy','networkx','h5py','imageio', 'scikit-image','tqdm', 'connected-components-3d','pyyaml'],
        include_dirs=getInclude(),
        packages=find_packages(),
    )

if __name__=='__main__':
    # install main python functions
    # python -m pip install --editable .
    setup_package()
