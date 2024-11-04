from setuptools import setup, find_packages

setup(
    name='feu',
    version='0.0.12.2024.11.4',
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