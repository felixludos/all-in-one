FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04@sha256:85fb7ac694079fff1061a0140fd5b5a641997880e12112d92589c3bbb1e8b7ca

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_VERSION=2.0.0+cu118
ARG NATTEN_WHEEL=https://github.com/SHI-Labs/NATTEN/releases/download/v0.14.6/natten-0.14.6%2Btorch200cu118-cp310-cp310-linux_x86_64.whl#sha256=afb05fc4857dbf623fd026f2f8e1db42e3531ba6bff30cf4741339418f299efa

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      ffmpeg \
      libsndfile1 \
      python3.10 \
      python3.10-dev \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.docker.txt ./
COPY constraints.docker.txt ./

RUN python3 -m pip install --no-cache-dir \
      pip==24.0 setuptools==69.5.1 wheel==0.43.0 \
    && python3 -m pip install --no-cache-dir --no-deps \
      "torch==${PYTORCH_VERSION}" \
      --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install --no-cache-dir --no-deps \
      "torchaudio==2.0.1+cu118" \
      --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install --no-cache-dir --no-deps "${NATTEN_WHEEL}"
RUN python3 -m pip install --no-cache-dir \
      Cython==0.29.37 numpy==1.23.5 scipy==1.10.1
RUN python3 -m pip install --no-cache-dir --no-build-isolation --constraint constraints.docker.txt -r requirements.docker.txt

# CUDA runtime omits the unversioned libnvrtc linker name that cuDNN loads for
# Demucs GPU inference.
RUN ln -s libnvrtc.so.11.2 /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin allin1 \
    && mkdir -p /cache \
    && chown -R allin1:allin1 /cache

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["python3", "-m", "uvicorn", "allin1.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--no-server-header"]
