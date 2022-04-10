FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
# 如果是3090以下 建议用10.2； 3090只能11.3
# FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# Just in case we need it
ENV DEBIAN_FRONTEND noninteractive

RUN apt update && apt install -y --no-install-recommends \
         git \
         curl \
         vim rsync ssh\
         libpng16-16  libjpeg8 libtiff5 libglib2.0-0 libsm6 libxext6 libxrender-dev htop\
         python-pip \
         python3-pip && \
         pip install --upgrade "pip < 21.0" && \
         pip3 install --upgrade "pip < 21.0"

RUN curl -o ~/miniconda.sh -LO https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda clean -ya && \
     /opt/conda/bin/conda create -n python37 python=3.7

# install zsh
RUN apt install -y wget git zsh tmux vim g++
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p git \
    -p ssh-agent \
    -p https://github.com/agkozak/zsh-z \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting

# change to conda env
ENV PATH /opt/conda/envs/python37/bin:/opt/conda/envs/bin:$PATH
RUN mkdir -p /home/kin/mmfn/data -p /home/kin/carla
WORKDIR /home/kin/mmfn

# training need
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN echo "export PYTHONWARNINGS="ignore"" >> ~/.zshrc
RUN echo "export PYTHONPATH="/home/kin/mmfn/":${PYTHONPATH}" >> ~/.zshrc

# 如果是3090以下 建议用10.2； 3090只能11.3
# RUN pip3 install torch torchvision torchaudio
