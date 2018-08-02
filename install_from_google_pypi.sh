#!/bin/bash
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <username> <password>"
  exit 1
fi
pip install -i "https://$1:$2@pypi-dot-protect-research.appspot.com/pypi kerastuner"
