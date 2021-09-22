FROM tensorflow/tensorflow:latest-gpu-py3-jupyter
RUN apt-get update && \
    apt-get install -y vim
COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt

