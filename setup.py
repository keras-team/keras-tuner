"""Setup script."""
from setuptools import find_packages
from setuptools import setup
import kerastuner
import time

version = kerastuner.__version__ + str(int(time.time()))

setup(
    name="Kerastuner",
    version=version,
    description="Hypertuner for Keras",
    author='Elie Bursztein',
    author_email='elieb@google.com',
    url='https://fixme',
    license='Apache License 2.0',
    install_requires=[
        "attrs",
        "numpy",
        "pathlib",
        "tabulate",
        "termcolor",
        "tqdm"
    ],
    packages=find_packages()
)
