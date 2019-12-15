ARG PY_VERSION=3.6
FROM python:${PY_VERSION}
ARG PY_VERSION=3.6

RUN if [ "${PY_VERSION}" = "3.6" ]; then pip install mkdocs ; fi

RUN pip install tensorflow>=2.0.0-beta1 \
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
