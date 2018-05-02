# Agent that learns how to play pong by using a simple Policy Gradient method

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import moviepy.editor as mpy


# Environment definitions
pong_inputdim = (1, 80, 80)
pong_actions = 6
eps = np.finfo(np.float32).eps.item()

# Initialize device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    discounted_r /= np.std(discounted_r) + eps
    #import matplotlib.pyplot as plt; plt.plot(discounted_r); plt.show()
    return discounted_r


class Policy(nn.Module):
    """Pytorch CNN implementing a Policy"""
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(1568, pong_actions)

        self.saved_log_probs = []

    def forward(self, x):
        x = F.relu(self.bn1((self.conv1(x))))
        x = F.relu(self.bn2((self.conv2(x))))
        x = F.relu(self.bn3((self.conv3(x))))
        return F.softmax(self.head(x.view(x.size(0), -1)), dim=1)

    def select_action(self, state):
        state = state.float().unsqueeze(0)
        probs = self(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()


def runepisode(env, policy, steps=2000, render=False):
    observation = env.reset()
    curx = prepro(observation)
    prevx = None
    observations = []
    rewards = []
    rawframes = []

    for _ in range(steps):
        if render:
            env.render()
        x = curx - prevx if prevx is not None else np.zeros(pong_inputdim)
        x = torch.tensor(x).to(device)
        action = policy.select_action(x)
        observation, reward, done, info = env.step(action)
        prevx = curx
        curx = prepro(observation)
        observations.append(x)
        rewards.append(reward)
        rawframes.append(observation)
        if done:
            break

    return rewards, observations, rawframes


def saveanimation(rawframes, filename):
    """Saves a sequence of frames as an animation

    The filename must include an appropriate video extension
    """
    clip = mpy.ImageSequenceClip(rawframes, fps=60)
    clip.write_videofile(filename)


def train(render=False, checkpoint='policygradient.pt', saveanimations=False):
    env = gym.make("Pong-v0")
    try:
        policy = torch.load(checkpoint)
        print("Resumed checkpoint {}".format(checkpoint))
    except:
        policy = Policy()
        print("Created policy network from scratch")
    print(policy)
    policy.to(device)
    print("device: {}".format(device))
    optimizer = optim.RMSprop(policy.parameters(), lr=1e-4)

    episode = 0
    while True:
        # Gather samples
        rewards, observations, rawframes = runepisode(env, policy, render=render)
        print("Total reward for episode {}: {}".format(episode, np.sum(rewards)))
        drewards = discount_rewards(rewards)
        # Update policy network
        policy_loss = [-log_prob * reward for log_prob, reward in zip(policy.saved_log_probs, drewards)]
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del policy.saved_log_probs[:]

        episode += 1
        # Save policy network from time to time
        if not episode % 100:
            torch.save(policy, checkpoint)
        # Save animation (if requested)
        if saveanimations:
            saveanimation(rawframes, "{}_episode{}.mp4".format(checkpoint, episode))

if __name__ == "__main__":
    train(render=False, saveanimations=False)
