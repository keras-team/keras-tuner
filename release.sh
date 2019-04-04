rm -rf dist/
python setup.py sdist bdist_wheel
twine upload --skip-existing --repository-url https://pypi-dot-protect-research.appspot.com dist/*