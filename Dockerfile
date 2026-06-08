FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV WORKSPACE=/workspace
ENV HF_HOME=${WORKSPACE}/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=${WORKSPACE}

ARG BLENDER_VERSION=4.4.3
ARG BLENDER_MAJOR=4.4

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    xz-utils \
    git \
    git-lfs \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libdbus-1-3 \
    libx11-6 \
    libxext6 \
    libxi6 \
    libxrender1 \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    libxfixes3 \
    libxkbcommon0 \
    libxxf86vm1 \
    libsm6 \
    libice6 \
    libfontconfig1 \
    libfreetype6 \
    libgl1 \
    libegl1 \
    libwayland-client0 \
    libwayland-cursor0 \
    libwayland-egl1 \
    libdecor-0-0 \
    libjemalloc2 \
    libtbb12 \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

RUN mkdir -p /opt/blender && \
    cd /tmp && \
    wget -q --show-progress \
      "https://download.blender.org/release/Blender${BLENDER_MAJOR}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" && \
    tar -xf "blender-${BLENDER_VERSION}-linux-x64.tar.xz" && \
    mv "blender-${BLENDER_VERSION}-linux-x64" "/opt/blender/blender-${BLENDER_VERSION}" && \
    ln -s "/opt/blender/blender-${BLENDER_VERSION}/blender" /usr/local/bin/blender && \
    ln -s "/opt/blender/blender-${BLENDER_VERSION}/blender" /usr/local/bin/blender44 && \
    rm -f "/tmp/blender-${BLENDER_VERSION}-linux-x64.tar.xz"

WORKDIR ${WORKSPACE}

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /tmp/requirements.txt

CMD ["/bin/bash"]
