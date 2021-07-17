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
    curl

# RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update

RUN $APT_INSTALL python3.9 python3.9-dev

RUN wget -O $HOME/get-pip.py https://bootstrap.pypa.io/get-pip.py

RUN python3.9 $HOME/get-pip.py

RUN ln -s /usr/bin/python3.9 /usr/local/bin/python3
RUN ln -s /usr/bin/python3.9 /usr/local/bin/python

ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Pascal;Volta;Turing"

RUN $APT_INSTALL libsm6 libxext6 libxrender1

RUN $PIP_INSTALL -r requirements.txt

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

RUN ldconfig
RUN apt-get clean
RUN apt-get autoremove
RUN rm -rf /var/lib/apt/lists/* /tmp/* ~/*
