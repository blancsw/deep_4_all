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
&& update-alternatives --config python \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/* # remove package lists to save space

# Install pip and poetry. Use ensurepip for robustness and install pipx
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \
    && pip install --upgrade pip \
    && pip install --upgrade setuptools packaging

RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip install ninja psutil && pip install flash-attn --no-build-isolation

WORKDIR /root

# Set up entrypoint and user for running
ENTRYPOINT ["/bin/bash"]