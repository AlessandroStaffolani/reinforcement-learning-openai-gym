
import gym
import numpy as np
from collections import namedtuple

import torch

from preprocessing.atari import preprocess_breakout, imshow
from models.dqn import AtariDQN
from agents.atari import BreakoutAgent

ENV_NAME = 'Breakout-v0'

N_EPISODES = 100
D = (1, 80, 80)  # Dimension of the frame after preprocessing BCHW

FIG_SIZE = (15, 8)  # Usefull for stats plotting
USE_CUDA = False
RENDER_ENV = False

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

Stats = namedtuple('Stats', ('episode_lengths', 'episode_rewards'))

device = torch.device("cuda" if USE_CUDA else "cpu")

env = gym.make(ENV_NAME)

policy_net = AtariDQN(D, env.action_space.n).to(device)
state_value_net = AtariDQN(D, env.action_space.n).to(device)


print(policy_net)
print(state_value_net)

agent = BreakoutAgent(env,
                      state_shape=D,
                      render_env=RENDER_ENV,
                      epsilon_start=EPS_START,
                      epsilon_end=EPS_END,
                      epsilon_decay=EPS_DECAY
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
        state = current_state - last_state if last_state is not None else torch.zeros((1, *D))
        last_state = current_state

        total_reward[i_episode] += reward

        # print('Episode: {} | Reward: {} | Total episode reward: {} | Episodes total reward mean {} | Info: {}'
        #      .format(i_episode + 1, reward, total_reward[i_episode], total_reward.mean(), info))
        episode_length += 1

    print('Episode terminated after {} frames'.format(episode_length + 1))
    stats.episode_lengths[i_episode] = episode_length
    stats.episode_rewards[i_episode] = total_reward[i_episode]
    episode_length = 0
    last_state = None

agent.close()
print(stats)
