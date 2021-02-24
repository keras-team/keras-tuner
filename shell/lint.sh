isort --sl -c .
if ! [ $? -eq 0 ]
then
    exit 1
fi
flake8
if ! [ $? -eq 0 ]
then
    exit 1
fi
black --check --line-length 85 --exclude kerastuner/protos .
if ! [ $? -eq 0 ]
then
    exit 1
fi
