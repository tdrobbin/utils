# For a reliable and production-ready build, pin specific versions for dependencies, rather than using 'latest'
FROM continuumio/anaconda3:latest

# Set work dir in the container
WORKDIR /app

# Update conda and install mamba
RUN conda update -n base conda
RUN conda install mamba -n base -c conda-forge

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    wget \
    sudo \
    curl \
    build-essential \
    clang \
    gcc \
    g++ \
    cmake \
    libssl-dev \
    libssh-dev \
    htop \
    libffi-dev \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxss1 \ 
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 

# APT-GET installations  
RUN apt-get update && apt-get install -y \
    postgresql postgresql-contrib libpq-dev \
    curl 

# MongoDB Installation
RUN apt-get install -y mongodb

# Redis Installation
RUN apt-get install -y redis-server

# Node.js Installation
RUN curl -sL https://deb.nodesource.com/setup_15.x | bash -
RUN apt-get install -y nodejs

# Docker Installation
RUN apt-get remove docker docker-engine docker.io containerd runc
RUN apt-get update && apt-get install -y \
     apt-transport-https \
     ca-certificates \
     curl \
     gnupg

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

RUN add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

RUN apt-get update && apt-get install docker-ce docker-ce-cli containerd.io

# Anaconda updates
RUN /opt/conda/bin/conda update conda 
RUN /opt/conda/bin/conda install anaconda-navigator 

# Copy local code to the container image.
ENV APP_HOME /app
COPY . ./app

ENTRYPOINT ["/bin/bash"]
