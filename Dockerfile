FROM nvidia/cuda:9.0-cudnn7-devel
LABEL maintainer="Álvaro Barbero Jiménez"

# System requirements
RUN apt-get update && \
    apt-get install -y build-essential cmake curl zlib1g-dev

# Install python miniconda3
ENV MINICONDA_HOME="/opt/miniconda"
ENV PATH="${MINICONDA_HOME}/bin:${PATH}"
RUN curl -o Miniconda3-latest-Linux-x86_64.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && chmod +x Miniconda3-latest-Linux-x86_64.sh \
  && ./Miniconda3-latest-Linux-x86_64.sh -b -p "${MINICONDA_HOME}" \
&& rm Miniconda3-latest-Linux-x86_64.sh

# Python requirements
ADD environment.yml environment.yml
RUN conda env update -n root -f environment.yml \
    && pip install gym[atari]
    && conda clean -y --all \
    && rm environment.yml

# App files
WORKDIR /deerl-pong
COPY *.py ./

ENTRYPOINT bash
