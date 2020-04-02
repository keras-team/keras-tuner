ARG PY_VERSION=3.6
FROM python:${PY_VERSION}

RUN pip install tensorflow-cpu \
                future \
                numpy \
                tabulate \
                terminaltables \
                colorama \
                tqdm \
                requests \
                scipy \
                scikit-learn \
                pytest \
                flake8 \
                mock \
                portpicker \
                pytest-xdist \
                pytest-cov

COPY ./ /keras-tuner
WORKDIR /keras-tuner
RUN pip install -e .[tests] --progress-bar off
