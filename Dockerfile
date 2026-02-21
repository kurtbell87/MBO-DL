# =============================================================================
# MBO-DL Docker image — multi-stage build
# Stage 1 (builder): Compile C++ tool binaries with all dependencies
# Stage 2 (runtime): Slim Python image with compiled binaries + ML stack
#
# Build:  docker buildx build --platform linux/amd64 -t mbo-dl:test .
# Run:    docker run --rm mbo-dl:test bar_feature_export --help
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — compile C++ binaries
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# System build tools (unzip needed for libtorch zip)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    pkg-config \
    ca-certificates \
    curl \
    wget \
    unzip \
    gpg \
    libssl-dev \
    libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

# CMake >= 3.24 required by databento-cpp (Ubuntu 22.04 ships 3.22)
RUN wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /etc/apt/trusted.gpg.d/kitware.gpg \
    && echo 'deb https://apt.kitware.com/ubuntu/ jammy main' > /etc/apt/sources.list.d/kitware.list \
    && apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/* \
    && cmake --version

# Apache Arrow + Parquet from official APT repo
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    gnupg \
    lsb-release \
    && wget -q "https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb" \
       -O /tmp/arrow.deb \
    && apt-get install -y -V /tmp/arrow.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       libarrow-dev \
       libparquet-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/arrow.deb

# Download libtorch CPU (linux x86_64) — required at cmake configure time
# even though tool binaries don't link against it
ARG LIBTORCH_VERSION=2.5.1
RUN curl -fsSL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip" \
    -o /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip
ENV Torch_DIR=/opt/libtorch/share/cmake/Torch

# Download ONNX Runtime CPU (linux x86_64) — required at cmake configure time
ARG ONNXRT_VERSION=1.17.1
RUN curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRT_VERSION}/onnxruntime-linux-x64-${ONNXRT_VERSION}.tgz" \
    -o /tmp/onnxrt.tgz \
    && tar xzf /tmp/onnxrt.tgz -C /opt \
    && mv /opt/onnxruntime-linux-x64-${ONNXRT_VERSION} /opt/onnxruntime \
    && rm /tmp/onnxrt.tgz

# Copy source tree
WORKDIR /src
COPY CMakeLists.txt ./
COPY cmake/ cmake/
COPY src/ src/
COPY tools/ tools/
COPY tests/ tests/

# CMake configure — FetchContent downloads GTest, databento-cpp, XGBoost here
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/opt/libtorch;/opt/onnxruntime"

# Build ONLY the 4 tool binaries (skip all test targets for speed)
RUN cmake --build build --parallel "$(nproc)" \
    --target bar_feature_export oracle_expectancy subordination_test info_decomposition_export

# Collect shared library dependencies needed at runtime.
# Exclude libs that the python:3.11-slim base already provides to avoid
# conflicts (especially OpenSSL — copying Ubuntu's libssl breaks Python SSL).
RUN mkdir -p /runtime-libs && \
    for bin in build/bar_feature_export build/oracle_expectancy \
               build/subordination_test build/info_decomposition_export; do \
        ldd "$bin" 2>/dev/null | grep '=> /' | awk '{print $3}' | \
        grep -v -E '(ld-linux|libc\.|libm\.|libdl\.|librt\.|libpthread\.|libssl\.|libcrypto\.|libgcc_s\.|libstdc\+\+\.|libz\.|libresolv\.)' | \
        while read -r lib; do \
            [ -f "$lib" ] && cp -nL "$lib" /runtime-libs/ 2>/dev/null || true; \
        done; \
    done && \
    echo "=== Runtime libs collected ===" && ls -lh /runtime-libs/

# ---------------------------------------------------------------------------
# Stage 2: Runtime — slim Python image with compiled binaries
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libzstd1 \
    libgomp1 \
    libstdc++6 \
    ca-certificates \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI for S3 results upload
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscli.zip /tmp/aws

# Python ML stack (install BEFORE C++ libs to avoid SSL conflicts)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && rm /tmp/requirements.txt

# Copy compiled tool binaries
COPY --from=builder /src/build/bar_feature_export /usr/local/bin/
COPY --from=builder /src/build/oracle_expectancy /usr/local/bin/
COPY --from=builder /src/build/subordination_test /usr/local/bin/
COPY --from=builder /src/build/info_decomposition_export /usr/local/bin/

# Copy shared library dependencies to /opt/app-libs/ (NOT /usr/local/lib/)
# to avoid conflicts with system libs used by Python
COPY --from=builder /runtime-libs/ /opt/app-libs/

# C++ binaries find their deps via LD_LIBRARY_PATH; Python uses system libs
ENV LD_LIBRARY_PATH=/opt/app-libs/

# Copy project Python scripts
COPY scripts/ /work/scripts/

WORKDIR /work

# No ENTRYPOINT — command specified at docker run
CMD ["bash"]
