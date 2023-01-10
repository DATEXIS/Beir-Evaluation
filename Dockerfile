FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

RUN apt-get update && DEBIAN_FRONTEND="noninteractive"\
    apt-get install -y \
    git \
    openssh-server ssh cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd &&\
    mkdir /root/.ssh && chmod 700 /root/.ssh &&\
    touch /root/.ssh/authorized_keys


RUN pip install --upgrade pip git+https://github.com/huggingface/transformers.git ninja \
                            gpustat beir typing \
                            scipy tqdm numpy \
                            sklearn datasets sentencepiece

WORKDIR DABERTX-Transformers
COPY DABERTX-transformers .
RUN pip install .


RUN mkdir -p /beir-evaluation
WORKDIR /beir-evaluation


COPY src/ ./src
