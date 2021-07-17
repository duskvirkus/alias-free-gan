FROM google/cloud-sdk:slim

# Build args.
ARG GITHUB_REF=refs/heads/main

ENV PYTHON_VERSION=3.9

ENV HOME /root

WORKDIR $HOME

ARG APT_INSTALL="apt-get install -y --no-install-recommends"
ARG PIP_INSTALL="python -m pip --no-cache-dir install --upgrade"
ARG GIT_CLONE="git clone --depth 10"

RUN apt-get update

RUN $APT_INSTALL \
    build-essential \
    software-properties-common \
    ca-certificates \
    wget \
    git \
    zlib1g-dev \
    nasm \
    cmake \
    curl \
    apt-utils

# RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh

ENV PATH=/root/miniconda3/bin:$PATH

RUN conda create -y --name container python=$PYTHON_VERSION

# Run the rest of commands within the new conda env.
# Use absolute path to appease Codefactor.
SHELL ["/root/miniconda3/bin/conda", "run", "-n", "container", "/bin/bash", "-c"]
RUN conda install -y python=$PYTHON_VERSION mkl

ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Pascal;Volta;Turing"

RUN $APT_INSTALL libsm6 libxext6 libxrender1

# Clone your repo. Optionally checkout pending code.
RUN git clone https://github.com/duskvirkus/alias-free-gan-pytorch-lightning.git && \
    cd alias-free-gan-pytorch-lightning && \
    git fetch origin $GITHUB_REF:CI && \
    git checkout CI

RUN conda init bash

RUN $PIP_INSTALL pytorch-lightning \
    pytorch-lightning-bolts \
    wandb \
    ninja \
    arrayfire \
    pytest \
    numpy \
    scipy \
    nltk \
    lmdb \
    cython \
    pydantic \
    pyhocon \
    torch==1.9 \
    torchvision==0.10 \
    opencv-python-headless \
    setuptools

RUN python -m pip uninstall -y pillow pil jpeg libtiff libjpeg-turbo

RUN $GIT_CLONE https://github.com/libjpeg-turbo/libjpeg-turbo.git
WORKDIR libjpeg-turbo
RUN mkdir build
WORKDIR build
RUN cmake -G"Unix Makefiles" -DCMAKE_INSTALL_PREFIX=libjpeg-turbo -DWITH_JPEG8=1 ..
RUN make
RUN make install
WORKDIR libjpeg-turbo
RUN mv include/jerror.h include/jmorecfg.h include/jpeglib.h include/turbojpeg.h /usr/include
RUN mv include/jconfig.h /usr/include/x86_64-linux-gnu
RUN mv lib/*.* /usr/lib/x86_64-linux-gnu
RUN mv lib/pkgconfig/* /usr/lib/x86_64-linux-gnu/pkgconfig
RUN ldconfig

RUN CFLAGS="${CFLAGS} -mavx2" $PIP_INSTALL --force-reinstall --no-binary :all: --compile pillow-simd

WORKDIR $HOME

RUN python -c "import pytorch_lighting; print(pytorch_lighting.__version__)"
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]



# RUN ldconfig
# RUN apt-get clean
# RUN apt-get autoremove
# RUN rm -rf /var/lib/apt/lists/* /tmp/* ~/*
