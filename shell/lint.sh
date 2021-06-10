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
black --check --line-length 85 --exclude keras_tuner/protos .
if ! [ $? -eq 0 ]
then
    exit 1
fi
