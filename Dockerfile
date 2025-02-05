# Need devel version to get the nvcc and install flash-attn
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG PY_VERSION=3.12

ARG PIP_DISABLE_PIP_VERSION_CHECK=1
ARG PIP_NO_CACHE_DIR=1
# Avoid using interactive Debian frontend
ARG DEBIAN_FRONTEND=noninteractive

ENV PATH="$PATH:/root/.local/bin"
ENV HOME=/root

RUN apt update \
&& apt install -y software-properties-common wget curl git \
&& add-apt-repository ppa:deadsnakes/ppa \
&& apt-get update \
&& apt-get install -y --no-install-recommends python${PY_VERSION} python${PY_VERSION}-dev python${PY_VERSION}-venv \
&& update-alternatives --install /usr/bin/python python /usr/bin/python${PY_VERSION} 1 \
&& update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PY_VERSION} 1 \
&& sudo apt install -y \
    build-essential \
    cmake \
    libgtk-3-dev \
    libcanberra-gtk3-module \
    libjpeg8-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libopenexr-dev \
    libatlas-base-dev \
    gfortran \
    libgl1 \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/* # remove package lists to save space

RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \
    && pip install --upgrade pip \
    && pip install --upgrade setuptools packaging

WORKDIR /root


RUN pip install --no-cache-dir "poetry>=2.0.0,<3.0.0"
COPY pyproject.toml .
COPY poetry.lock .
RUN poetry config virtualenvs.create false && \
    poetry config virtualenvs.in-project false && \
    poetry install -n


# Set up entrypoint and user for running
ENTRYPOINT ["/bin/bash"]