FROM pytorch/pytorch:latest
RUN apt-get update && \
    apt-get install -y vim
COPY docker_requirements.txt /opt/app/requirements.txt
RUN pip install --upgrade pip && \ 
    pip install -r /opt/app/requirements.txt
