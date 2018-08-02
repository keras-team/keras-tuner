"""Setup script."""
from setuptools import find_packages
from setuptools import setup

setup(
    name="Kerastuner",
    version="0.6",
    description="Hypertuner for Keras",
    author='Elie Bursztein',
    author_email='elieb@google.com',
    url='https://fixme',
    license='Apache License 2.0',
    install_requires=[
        "attrs",
        "cprint",
        "numpy",
        "tabulate",
        "termcolor",
        "tqdm",
        "xxhash",
        "psutil",
    ],
    packages=find_packages(),
)
