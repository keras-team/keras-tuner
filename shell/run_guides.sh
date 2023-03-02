#!/bin/bash
guides=(
    https://raw.githubusercontent.com/keras-team/keras-io/master/guides/keras_tuner/getting_started.py
    https://raw.githubusercontent.com/keras-team/keras-io/master/guides/keras_tuner/distributed_tuning.py
    https://raw.githubusercontent.com/keras-team/keras-io/master/guides/keras_tuner/custom_tuner.py
    https://raw.githubusercontent.com/keras-team/keras-io/master/guides/keras_tuner/visualize_tuning.py
    https://raw.githubusercontent.com/keras-team/keras-io/master/guides/keras_tuner/tailor_the_search_space.py
)

for guide in ${guides[@]}; do
    wget $guide -O /tmp/a.py
    if ! python /tmp/a.py; then
        echo "error occured!"
        exit 1
    fi
done