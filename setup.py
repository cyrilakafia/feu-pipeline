from setuptools import setup, find_packages

setup(
    name='feu',
    version='0.01.12.2025',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'openpyxl',
        'h5py',
        'matplotlib',

    ],
)