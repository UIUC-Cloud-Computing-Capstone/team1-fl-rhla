# Docker Smoke Test Setup

This repository provides a **Docker container** to run the `team1-fl-rhla` training script with GPU support and Anaconda environment. Outputs are automatically synced to a host folder using a bind mount.

---

## Prerequisites

1. **Docker** installed
2. **NVIDIA GPU** with drivers
3. **NVIDIA Container Toolkit** for GPU support

Check GPU access:

```bash
docker run --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

---

## Build the Docker Image

Clone this repository and build the Docker image:

```bash
git clone https://github.com/UIUC-Cloud-Computing-Capstone/team1-fl-rhla.git
cd team1-fl-rhla/docker

docker build -t hralora-smoke-test .
```

---

## Run the Training Script

Create an output folder on the host:

```bash
mkdir -p out
```

Run the container with GPU access and live output bind mount:

```bash
docker run --gpus all \
  -v $(pwd)/out:/workspace/team1-fl-rhla/log \
  hralora-smoke-test
```