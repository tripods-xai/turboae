FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
RUN apt-get update && pip install scikit-commpy

