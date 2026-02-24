FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Cài curl để healthcheck hoạt động
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean

WORKDIR /workspace