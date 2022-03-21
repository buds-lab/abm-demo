# https://bytepawn.com/solving-the-cartpole-reinforcement-learning-problem-with-pytorch.html
# https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# simple actions
def select_action_simple(obs):
    # using the x velocity and angular velocity
    # if it goes to the right, compesante to the left and vice-versa
    if obs[2] + obs[3] < 0:
        return 0
    else:
        return 1


# a simple NN
class PolicyNN(nn.Module):
    def __init__(self):
        super(PolicyNN, self).__init__()
        self.fc_1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.fc_2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []


    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return F.softmax(x, dim=1)


def select_action(model, obs):
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    probs = model(obs)
    m = Categorical(probs)
    action = m.sample()
    model.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(model):
    R = 0
    policy_loss = []
    returns = []
    for r in model.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(model.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_log_probs[:]


#############################
# Begining of the environment!
# load up any environment! But let's go with the cartpole just for shits and giggles
env = gym.make('CartPole-v0')
env.reset()

# some simple NN
model = PolicyNN()
optimizer = optim.Adam(model.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()
gamma = 0.99

# a new concept arrived! an "episode" one full run of the simulationself.
# this changes depending on the problem and can be defined however we want
num_steps = 10000
episodes = 10000
log_interval = 10
running_reward = 10
simple_actions = False # TODO: change!

for i_episode in range(episodes):
    # every episode is a new start, or a great start
    observation, episode_reward = env.reset(), 0
    probs = []

    # here we define how long the episode will run, say "num_steps"
    for t in range(num_steps + 1):
        env.render()  # we are all visual people, let's see the actual cartpole
        # no more random actions!
        if simple_actions:
            action = select_action_simple(observation)
        else:
            action = select_action(model, observation)

        # remember that `step` is one of the very important functions!
        # when you are doing your custom gym environment, this is the most
        # important function to pay attention to.

        # observation: states of the environment after performing `action`
        #   For the cartpoke problem is a 4-tuple:
        #   x position of cart,
        #   x velocity of cart,
        #   angular position of pole,
        #   angular velocity of pole.
        # reward: whether the action was good or bad, useful for the agent to learn
        #   in this environment, every time-step the pole remains straight up,
        #   rewards is +1, -1 other wise.
        #   The entire problem is "solved" when averaged reward is >= 195 over
        #   100 consecutive episodes.
        # done: has the episode finished?
        # info: useful for debinning
        observation, reward, done, info = env.step(action)
        model.rewards.append(reward)
        episode_reward += reward

        if done:
            break

    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
    if not simple_actions:
        finish_episode(model)

    if i_episode % log_interval == 0:
        print("Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
            i_episode, episode_reward, running_reward))

    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
            "the last episode runs to {} time steps!".format(running_reward, t))
        break

env.close()
