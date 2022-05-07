sudo docker build . -t turboae:latest
sudo docker run --gpus all -v $(pwd):/code/turboae -u $(id -u):$(id -g) -w /code/turboae -it turboae:latest bash
