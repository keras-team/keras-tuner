"""Setup script."""
from setuptools import find_packages
from setuptools import setup

setup(
    name="Kerastuner",
    version="0.5",
    description="Hypertuner for Keras",
    author='Elie Bursztein',
    author_email='elieb@google.com',
    url='https://fixme',
    license='Apache License 2.0',
    install_requires=open("requirements.txt").read().splitlines(),
    packages=find_packages(),
)
