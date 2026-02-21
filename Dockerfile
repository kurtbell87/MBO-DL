# =============================================================================
# MBO-DL Docker image — multi-stage build
# Stage 1 (builder): Compile C++ tools with all dependencies
# Stage 2 (runtime): Slim Python image with compiled binaries + ML stack
#
# Build:  docker buildx build --platform linux/amd64 -t mbo-dl:latest .
# Run:    docker run --rm -v /data:/data:ro mbo-dl:latest bar_feature_export --help
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — compile C++ binaries
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# System build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    ca-certificates \
    curl \
    wget \
    libssl-dev \
    libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

# Apache Arrow + Parquet from official APT repo (NOT FetchContent)
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    gnupg \
    lsb-release \
    && wget -qO- https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb -O /tmp/arrow.deb \
    && apt-get install -y /tmp/arrow.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libarrow-dev \
    libparquet-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/arrow.deb

# Download libtorch CPU (linux x86_64) — early layer for caching
ARG LIBTORCH_URL=https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-LATEST.zip
RUN curl -fsSL "${LIBTORCH_URL}" -o /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip
ENV Torch_DIR=/opt/libtorch/share/cmake/Torch

# Download ONNX Runtime CPU (linux x86_64)
ARG ONNXRT_VERSION=1.17.1
RUN curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRT_VERSION}/onnxruntime-linux-x64-${ONNXRT_VERSION}.tgz" \
    -o /tmp/onnxrt.tgz \
    && tar xzf /tmp/onnxrt.tgz -C /opt \
    && mv /opt/onnxruntime-linux-x64-${ONNXRT_VERSION} /opt/onnxruntime \
    && rm /tmp/onnxrt.tgz
ENV onnxruntime_DIR=/opt/onnxruntime
ENV CMAKE_PREFIX_PATH="/opt/onnxruntime:${CMAKE_PREFIX_PATH}"

# Copy source tree
WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake/ cmake/
COPY src/ src/
COPY tools/ tools/
COPY tests/ tests/

# Build with Release — FetchContent handles GTest, databento-cpp, XGBoost
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/opt/libtorch;/opt/onnxruntime" \
    && cmake --build build --parallel "$(nproc)" \
    && echo "Build complete"

# ---------------------------------------------------------------------------
# Stage 2: Runtime — slim Python image with compiled binaries
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies for compiled binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libzstd1 \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI for S3 results upload
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscli.zip /tmp/aws

# Copy compiled tool binaries from builder
COPY --from=builder /src/build/bar_feature_export /usr/local/bin/
COPY --from=builder /src/build/oracle_expectancy /usr/local/bin/
COPY --from=builder /src/build/subordination_test /usr/local/bin/
COPY --from=builder /src/build/info_decomposition_export /usr/local/bin/

# Copy shared libraries that binaries link against
COPY --from=builder /opt/libtorch/lib/*.so* /usr/local/lib/
COPY --from=builder /opt/onnxruntime/lib/*.so* /usr/local/lib/

# Arrow/Parquet shared libs from builder
COPY --from=builder /usr/lib/x86_64-linux-gnu/libarrow*.so* /usr/local/lib/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libparquet*.so* /usr/local/lib/

# databento + xgboost shared libs (built via FetchContent)
COPY --from=builder /src/build/_deps/databento-build/lib*.so* /usr/local/lib/
COPY --from=builder /src/build/_deps/xgboost-build/lib*.so* /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
RUN ldconfig

# Python ML stack
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && rm /tmp/requirements.txt

# Copy project Python scripts
COPY scripts/ /work/scripts/

WORKDIR /work

# No ENTRYPOINT — command specified at docker run
CMD ["bash"]
