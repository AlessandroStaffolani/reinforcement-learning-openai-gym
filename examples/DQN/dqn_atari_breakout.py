import gym
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt

import torch
import torch.optim as optim

from models.dqn import AtariDQN
from agents.atari import BreakoutAgent
from agents.memory.experience import Experience
from utils.gym_wrapper import make_env

ENV_NAME = 'Breakout-v0'

N_EPISODES = 2
D = (1, 80, 80)  # Dimension of the frame after preprocessing BCHW

FIG_SIZE = (15, 8)  # Usefull for stats plotting
USE_CUDA = False
RENDER_ENV = True

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

GAMMA = 0.999
BATCH_SIZE = 128
MEMORY_LIMIT = 10 ** 4
TARGET_UPDATE = 10  # Every TARGET_UPDATE episode update the state_value_net weights copying from the policy_net

Stats = namedtuple('Stats', ('episode_lengths', 'episode_rewards'))

device = torch.device("cuda" if USE_CUDA else "cpu")

env = gym.make(ENV_NAME)

policy_net = AtariDQN(D, env.action_space.n).to(device)
state_value_net = AtariDQN(D, env.action_space.n).to(device)

state_value_net.load_state_dict(policy_net.state_dict())
state_value_net.eval()

# Init the RMSprop optimizer to spped-up the mini-batch learning
optimizer = optim.RMSprop(policy_net.parameters())

print(policy_net)
print(state_value_net)

agent = BreakoutAgent(env,
                      state_shape=D,
                      render_env=RENDER_ENV,
                      epsilon_start=EPS_START,
                      epsilon_end=EPS_END,
                      epsilon_decay=EPS_DECAY,
                      gamma=GAMMA,
                      batch_size=BATCH_SIZE,
                      experience=Experience,
                      memory_limit=MEMORY_LIMIT,
                      device=device
                      )

episode_length = 0
total_reward = np.zeros(N_EPISODES)

stats = Stats(
    episode_lengths=np.zeros(N_EPISODES),
    episode_rewards=np.zeros(N_EPISODES)
)

last_state = None

for i_episode in range(N_EPISODES):
    done = False

    # Reset the env after each episode termination
    current_state = agent.reset()
    state = current_state

    while not done:
        agent.render()
        action = agent.choose_action(state=state, policy_net=policy_net, steps_done=episode_length)
        current_state, reward, done, info = agent.act(action)

        # Compute state difference between previous and current state
        if not done:
            next_state = current_state - last_state if last_state is not None else torch.zeros((1, *D))
        else:
            next_state = None
        last_state = current_state

        # Necessary for the sake of compatibility with torch when using it in the optimize_model method
        action = torch.tensor([[action]], dtype=torch.long, device=device)
        reward = torch.tensor([reward], device=device)
        # Push an experience tuple in the memory
        agent.memory_push(current_state, action, next_state, reward)

        total_reward[i_episode] += reward

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        agent.optimize_model(
            policy_net=policy_net,
            state_value_net=state_value_net,
            optimizer=optimizer
        )

        # print('Episode: {} | Reward: {} | Total episode reward: {} | Episodes total reward mean {} | Info: {}'
        #      .format(i_episode + 1, reward, total_reward[i_episode], total_reward.mean(), info))
        episode_length += 1

    # Update the state_value_net network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        state_value_net.load_state_dict(policy_net.state_dict())

    print('Episode terminated after {} frames'.format(episode_length + 1))
    stats.episode_lengths[i_episode] = episode_length
    stats.episode_rewards[i_episode] = total_reward[i_episode]
    episode_length = 0
    last_state = None

agent.close()
print(stats)

fig1 = plt.figure(figsize=FIG_SIZE)
plt.plot(stats.episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.title("Episode Length over Time")
plt.show(fig1)

fig2 = plt.figure(figsize=FIG_SIZE)
smoothing_window = 1
goal_value = None
rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
plt.plot(rewards_smoothed)
plt.xlabel("Episode")
plt.ylabel("Episode Reward (Smoothed)")
title = "Episode Reward over Time (Smoothed over window size {})".format(smoothing_window)

if goal_value is not None:
    plt.axhline(goal_value, color='g', linestyle='dashed')
    title = "Episode Reward over Time (Smoothed over window size" \
            " " + str(smoothing_window) + ", goal value " + str(goal_value) + ")"

plt.title(title)
plt.show(fig2)