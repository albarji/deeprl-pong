# Deep Reinforcement Learning Pong

A repository for methods to learn to play Pong, using Deep Reinforcement Learning methods.

## Install

### Docker

[Install Docker](https://www.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), then build 
the image as

    docker build -t deeprl-pong .
    
After the build you can open a terminal into the docker container as

    nvidia-docker run --rm -it deeprl-pong

### Local install

After getting a working [Conda distribution](https://anaconda.org/anaconda/python), install the environment as

    conda env create -f environment.yml
    
Then log into the environment

    source activate deeprl-pong
    
And install the addons for Atari games

    pip install gym[atari]

## Implemented methods

### Policy gradients

**policygradientpytorch.py**

A Pytorch implentation of the Policy Gradients method, maily based on [Andrej Karpathy's blogpost on the topic](http://karpathy.github.io/2016/05/31/rl/).

### Policy gradients (Keras version)

**policygradientkeras.py**

Keras version of the Policy Gradients method. Way less efficient, as Keras does not allow to tinker with the gradients as easily as Pytorch.

