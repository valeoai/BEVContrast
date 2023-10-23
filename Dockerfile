FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

# --------------------------------------------------------------------

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server sudo

# -------------------------------------------------------------------

##############################################
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --install-option="--blas=openblas" --install-option="--force_cuda" -v --no-deps

RUN apt-get install libgl1-mesa-glx -y
RUN pip install spconv-cu113

# Torchsparse
RUN apt-get update
RUN apt-get install libsparsehash-dev
RUN FORCE_CUDA=1 pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# Torch Geometric
RUN pip install torch_scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
RUN pip install torch-geometric hydra-core
RUN pip install tqdm easydict tensorboardX

RUN pip install pytorch-lightning==1.6.5

RUN pip install nuscenes-devkit==1.1.9
