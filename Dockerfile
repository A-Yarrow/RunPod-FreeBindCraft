# 1 Base image optimized for JAX on CUDA 12.1 (works on RTX6000, L40, L40S, A40)
FROM nvcr.io/nvidia/jax:23.08-py3
# For A100 with cuda 11.8 use:
# FROM nvcr.io/nvidia/jax:23.08-cuda11.8-py3

LABEL org.opencontainers.image.source="https://github.com/A-Yarrow/bindcraft-runpod.git"
LABEL org.opencontainers.image.description="BindCraft GPU (RunPod UI, no PyRosetta)"
LABEL maintainer="Yarrow Madrona <yarrowmadrona@gmail.com>"

# 2 OS dependencies + OpenCL tools for GPU registration
RUN apt-get update && apt-get install -y \
    wget \
    vim \
    rsync \
    git \
    libgfortran5 \
    ca-certificates \
    clinfo \
    ocl-icd-opencl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3 Install Miniconda (lighter than Miniforge3)
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh && \
    bash miniforge.sh -b -p $CONDA_DIR && \
    rm miniforge.sh

# 4 Set up Conda and Mamba
RUN conda install -y -n base -c conda-forge mamba && \
    conda clean -afy

# 5 Clone BindCraft RunPod repo (always no PyRosetta)
RUN git clone --branch dev --single-branch https://github.com/A-Yarrow/bindcraft-runpod.git /app/bindcraft

WORKDIR /app/bindcraft

# 6 Install BindCraft (install_bindcraft.sh should exclude PyRosetta logic)
RUN chmod +x install_bindcraft.sh && \
    bash install_bindcraft.sh --no-pyrosetta || true

# 7 Set permissions on startup script and notebook
RUN chmod 755 /app/bindcraft/start.sh && \
    chmod 644 /app/bindcraft/bindcraft-runpod-start.ipynb

# 8 JAX / CUDA runtime settings
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
ENV XLA_FLAGS="--xla_gpu_enable_command_buffer=false"

# 9 Jupyter port
EXPOSE 8888

# 10 Default startup
CMD ["/app/bindcraft/start.sh"]