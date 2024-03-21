from setuptools import setup, find_packages
def setup_package():
    __version__ = '0.1'
    url = 'https://github.com/PytorchConnectomics/EM-util'

    setup(name='em_util',
        description='Utility Functions for EM Image Analysis',
        version=__version__,
        url=url,
        license='MIT',
        author='Donglai Wei',
        install_requires=['scipy','numpy','networkx','h5py','imageio', 'scikit-image','tqdm', 'connected-components-3d'],
        packages=find_packages(),
    )

if __name__=='__main__':
    # install main python functions
    # pip install --editable .
    setup_package()
