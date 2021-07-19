FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update && DEBIAN_FRONTEND="noninteractive"\
    apt-get install -y \
    git \
    openssh-server ssh cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd &&\
    mkdir /root/.ssh && chmod 700 /root/.ssh &&\
    touch /root/.ssh/authorized_keys


RUN pip install --upgrade pip git+https://github.com/huggingface/transformers.git ninja \
                            gpustat beir\
                            scipy \
                            sklearn datasets sentencepiece

RUN mkdir -p /beir-evaluation
WORKDIR /beir-evaluation

COPY src/ ./src
