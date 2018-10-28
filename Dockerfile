FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -qy git vim wget bzip2 gcc g++ \
        cmake libopenmpi-dev python3-dev zlib1g-dev \
	&& apt-get purge

ENV CONDA_DIR /opt/conda

ENV PATH $CONDA_DIR/bin:$PATH

RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh -qO /tmp/miniconda3.sh \
    && bash /tmp/miniconda3.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda3.sh

RUN conda install -y pytorch-cpu torchvision-cpu -c pytorch

RUN conda install -y jupyter matplotlib

RUN pip install pybullet tqdm stable-baselines papermill nbdime

COPY . /example

WORKDIR /example

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

CMD ["papermill", "simple_training_example.ipynb", "out_PPO.ipynb"]