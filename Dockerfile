
#FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         wget \
         ca-certificates \
         libboost-all-dev \
         python-qt4 \
         libjpeg-dev \
         libpng-dev \
         re2c \
         python3-setuptools \
         libgl1-mesa-glx libsm6 libxrender1 libxext-dev \
         libgl1 python3-pip && \
     rm -rf /var/lib/apt/lists/*

WORKDIR /home

RUN cd /home && \
    git clone https://github.com/mchong6/RetrieveInStyle.git

RUN pip install tqdm gdown scikit-learn==0.22 scipy lpips dlib opencv-python

COPY models/ /home/RetrieveInStyle/

ENV CUDA_HOME=/opt/conda/
RUN mkdir -p /home/.cache/torch/hub/checkpoints/ && \
    mv /home/RetrieveInStyle/vgg16-397923af.pth /home/.cache/torch/hub/checkpoints/

RUN pip install pandas matplotlib 
WORKDIR /home/RetrieveInStyle

####
ARG  USER=docker
ARG  GROUP=docker
ARG  UID
ARG  GID
## must use ; here to ignore user exist status code
RUN  [ ${GID} -gt 0 ] && groupadd -f -g ${GID} ${GROUP}; \
     [ ${UID} -gt 0 ] && useradd -d /home -M -g ${GID} -K UID_MAX=${UID} -K UID_MIN=${UID} ${USER}; \
     chown -R ${UID}:${GID} /home && \
     touch /var/run/nginx.pid && \
     mkdir -p /var/log/nginx /var/lib/nginx && \
     chown ${UID}:${GID} $(find /home -maxdepth 2 -type d -print) /var/run/nginx.pid && \
     chown -R ${UID}:${GID} /var/log/nginx /var/lib/nginx
USER ${UID}
####

