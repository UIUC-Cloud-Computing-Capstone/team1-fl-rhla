FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

RUN rm -f /etc/apt/apt.conf.d/docker-clean && apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p $CONDA_DIR
RUN rm /tmp/miniconda.sh
RUN conda clean -afy

RUN conda init bash

WORKDIR /workspace

# Clone repo
RUN git clone https://github.com/UIUC-Cloud-Computing-Capstone/team1-fl-rhla.git
WORKDIR /workspace/team1-fl-rhla

# Create conda environment
RUN conda env create --name env.fl --file=setup/environment-final.yml

# Create HuggingFace accelerate config directory
RUN mkdir -p /root/.cache/huggingface/accelerate

# Execute
CMD ["conda", "run", "--no-capture-output", "-n", "env.fl", "bash", "-c", \
     "nvidia-smi || true && \
      cp setup/accelerate_default_config.yaml /root/.cache/huggingface/accelerate/default_config.yaml && \
      bash scripts/experiments/run-cifar100-smoke-test.sh"]