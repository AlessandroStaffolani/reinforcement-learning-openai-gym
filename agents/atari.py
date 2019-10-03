import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.experience_replay import AtariExperience
from preprocessing.atari import preprocess_breakout
from agents.memory.experience import ExperienceReplay, Experience


class BreakoutAgent:

    def __init__(self,
                 env,
                 state_shape=(1, 80, 80),
                 render_env=False,
                 epsilon_start=0.9,
                 epsilon_end=0.05,
                 epsilon_decay=200,
                 gamma=0.999,
                 batch_size=128,
                 experience=Experience,
                 memory_limit=10 ** 4,
                 device='cpu'
                 ):
        self.env = env
        self.state_shape = state_shape
        self.render_env = render_env
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ExperienceReplay(memory_limit, experience)
        self.device = device

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

    def act(self, action):
        state, reward, done, info = self.env.step(action)
        return preprocess_breakout(state), reward, done, info

    def close(self):
        return self.env.close()

    def memory_push(self, *args):
        self.memory.push(*args)

    def memory_sample(self, as_tuple=False):
        return self.memory.sample(self.batch_size, as_tuple=as_tuple)

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

    def optimize_model(self, policy_net, state_value_net, optimizer):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory_sample(as_tuple=True)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = state_value_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


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
