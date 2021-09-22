sudo docker build . -t turbo:latest
sudo docker run --gpus all -v $(pwd):/code/turboae -u $(id -u):$(id -g) -it turbo:latest bash
