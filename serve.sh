#!/usr/bin/env bash

# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving

TESTDATA="$(pwd)/serve_models/"

# Start TensorFlow Serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/viz_mnist:/models/viz_mnist" \
    -e MODEL_NAME=viz_mnist \
    tensorflow/serving
