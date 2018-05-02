# Agent that learns how to play pong by using a simple Policy Gradient method

import gym
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
import keras.backend as K
import numpy as np


# Environment definitions
pong_inputdim = (80, 80, 1)
epsilon = 1e-6


def prepro(image):
    """ prepro 210x160x3 uint8 frame into 80x80 2D image

    Source: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
    """
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return np.reshape(image, pong_inputdim)


def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward

    Source: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r) + epsilon
    return discounted_r


def pgloss(y_true, y_pred):
    """Policy Gradients loss. Maximizes log(output) · reward"""
    return - K.mean(K.log(y_pred) * y_true)


def buildpolicynetwork():
    model = Sequential()
    model.add(Conv2D(32, 5, activation='relu', input_shape=pong_inputdim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(2))
    model.add((Flatten()))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss=pgloss)
    return model


def runepisode(env, model, steps=2000, render=False):
    observation = env.reset()
    curx = prepro(observation)
    prevx = None
    observations = []
    rewards = []

    for _ in range(steps):
        if render:
            env.render()
        x = curx - prevx if prevx is not None else np.zeros(pong_inputdim)
        aprob = model.predict(np.expand_dims(x, axis=0))
        action = 2 if np.random.uniform() < aprob else 3
        observation, reward, done, info = env.step(action)
        prevx = curx
        curx = prepro(observation)
        observations.append(x)
        rewards.append(reward)
        if done:
            break

    return rewards, observations


def train(render=False):
    env = gym.make("Pong-v0")
    model = buildpolicynetwork()
    model.summary()

    episode = 0
    while True:
        # Gather samples
        rewards, observations = runepisode(env, model, render=render)
        print("Total reward for episode {}: {}".format(episode, np.sum(rewards)))
        drewards = discount_rewards(rewards)
        # Update policy network
        model.fit(np.array(observations), drewards, epochs=1, batch_size=len(observations))
        episode += 1


if __name__ == "__main__":
    train(render=True)
