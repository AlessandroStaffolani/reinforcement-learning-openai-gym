import math
import random
import numpy as np

import torch
from torch import nn

from agents.experience_replay import AtariExperience
from preprocessing.atari import preprocess_breakout


class BreakoutAgent:

    def __init__(self,
                 env,
                 state_shape=(1, 80, 80),
                 render_env=False,
                 epsilon_start=0.9,
                 epsilon_end=0.05,
                 epsilon_decay=200
                 ):
        self.env = env
        self.state_shape = state_shape
        self.render_env = render_env
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start

    def _update_espilon(self, steps_done):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * steps_done / self.epsilon_decay)

    def reset(self):
        return preprocess_breakout(self.env.reset())

    def render(self):
        if self.render_env:
            return self.env.render()
        else:
            return None

    def choose_action(self, state, policy_net, steps_done):
        # Update epsilon
        self._update_espilon(steps_done)
        sample = random.random()

        if sample < self.epsilon:
            # Perform a random action selection
            return random.randrange(self.env.action_space.n)
        else:
            # Choose an action accordingly to the policy parameterization
            # -> choose the action with the highest probability accordingly the policy
            with torch.no_grad():
                # Feed the policy network with the state -> obtain the probability for each action
                action_prob = policy_net(state)
                # t.max(1) compute the max and return a tuple (max_val, max_index)
                # We need the action (so the index)
                return action_prob.max(1)[1].item()

    def act(self, action):
        state, reward, done, info = self.env.step(action)
        return preprocess_breakout(state), reward, done, info

    def close(self):
        return self.env.close()


class AtariAgent:

    def __init__(self, env, replay_memory):
        self.env = env
        self.replay_memory = replay_memory
        self._reset()
        self.last_action = 0

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        """
        Select action
        Execute action and step environment
        Add state/action/reward to experience replay
        """
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state

        exp = AtariExperience(self.state, action, reward, is_done, new_state)
        self.replay_memory.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def calculate_loss(self, batch, net, target_net, device="cpu", gamma=0.99):
        """
        Calculate MSE between actual state action values,
        and expected state action values from DQN
        """
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)

        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)

        done = torch.ByteTensor(dones).to(device)

        state_action_values = net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)
        next_state_values = target_net(next_states_v).max(1)[0]
        next_state_values[done] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * gamma + rewards_v
        return nn.MSELoss()(state_action_values, expected_state_action_values)
