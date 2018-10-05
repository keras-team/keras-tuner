"""Setup script."""

# pip release : python setup.py sdist upload -r google
# pip install -i "https://$1:$2@pypi-dot-protect-research.appspot.com/pypi kerastuner"


from setuptools import find_packages
from setuptools import setup

setup(
    name="Kerastuner",
    version="0.6.15",
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
        "pyfarmhash",
    ],
    packages=find_packages(),
)
