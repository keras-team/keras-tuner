rm -rf dist/
python setup.py sdist bdist_wheel
twine upload --verbose --repository-url https://pypi-dot-protect-research.appspot.com dist/*