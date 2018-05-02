# Deep Reinforcement Learning Pong

A repository for methods to learn to play Pong, using Deep Reinforcement Learning methods.

## Implemented methods

### Policy gradients

**policygradientpytorch.py**

A Pytorch implentation of the Policy Gradients method, maily based on [Andrej Karpathy's blogpost on the topic](http://karpathy.github.io/2016/05/31/rl/).

### Policy gradients (Keras version)

**policygradientkeras.py**

Keras version of the Policy Gradients method. Way less efficient, as Keras does not allow to tinker with the gradients as easily as Pytorch.

