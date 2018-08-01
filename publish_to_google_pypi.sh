#!/bin/bash

PYPIRC_TEMPLATE=<<EOS
[distutils]
index-servers =
  pypi
  pypitest
  google

[pypi]  # Optional
repository: https://pypi.python.org/pypi
username: your_pypi_user
password: your_pypi_password

[pypitest]  # Also optional
repository: https://testpypi.python.org/pypi
username: your_pypi_user
password: your_pypi_password

[google]
username: your_google_pypi_user
password: your_google_pypi_password
repository=https://pypy-dot-keras-tuner.appspot.com
EOS

if [ ! -f "$HOME/.pypirc" ]; then
  echo "Please paste the following into ~/.pypirc, and fill in the "
  echo "missing data."
  echo "Then, run this command again"
  echo "$PYPIRC_TEMPLATE"
  exit 1
fi

python setup.py sdist upload -r google
