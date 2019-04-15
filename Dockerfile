FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3-pip

RUN pip3 install -U pip wheel
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl && \
    rm -r ~/.cache/pip

WORKDIR /transformer-lm

COPY requirements.txt .
RUN pip3 install awscli && \
    pip3 install -r requirements.txt && \
    rm -r ~/.cache/pip

COPY . .
RUN pip3 install -e .
