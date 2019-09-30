import numpy as np
import sys
import math

from utils.plotting import EpisodeStats
from algorithms.discretization_functions import default_discretization


class QLearning:

    def __init__(self,
                 env,
                 buckets=(1, ),
                 discount_factor=1.0,
                 min_alpha=0.1,
                 min_epsilon=0.1,
                 discretize_fn=default_discretization,
                 ada_divisor=25,
                 env_render=False,
                 env_wrapper=False
                 ):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        :param env: OpenAI environment.
        :param buckets: Tuple containing the bins for discretize the continuous features. Default None.
        :param discount_factor: Gamma discount factor. Float between [0, 1].
        :param alpha: Temporal-Difference (TD) learning rate. Float between (0, 1].
        :param discretize_fn: Function used to discretize the state when necessary. Default no trasformation done.
        :param epsilon: Chance to sample a random action. Float between (0, 1].
        :param env_render: Wheter render the env or not. Boolean default False.
        """
        self.env = env
        self.buckets = buckets
        self.discount_factor = discount_factor
        self.min_alpha = min_alpha
        self.min_epsilon = min_epsilon
        self.discretize_fn = discretize_fn
        self.ada_divisor = ada_divisor
        self.env_render = env_render
        self.env_wrapper = env_wrapper
        # TODO check the value of the parameters

    def __init_action_value_function(self):
        """
        Init the action-value function Q
        """
        return np.zeros(self.buckets + (self.env.action_space.n, ))

    def process_state(self, state):
        """
        Method to process the state when necessary using the discretize_fn provided to the agent.

        :param state: State to be processed.
        :return: The state processed.
        """
        return self.discretize_fn(state, self.env, self.buckets)

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def get_alpha(self, t):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))

    def learn(self, num_episodes):
        """
        Q-learning function approximation: Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s',a) - Q(s,a))

        :param num_episodes:
        :return: A tuple (Q, stats).
        Q is the optimal action-value function founded, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """

        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        Q = self.__init_action_value_function()

        # Keeps track of useful statistics
        stats = EpisodeStats(episode_lengths=np.zeros(num_episodes),
                             episode_rewards=np.zeros(num_episodes),
                             episode_epsilon=np.zeros(num_episodes),
                             episode_alpha=np.zeros(num_episodes))

        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, self.env.action_space.n)

        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()

            # Get the first random state
            state = self.process_state(self.env.reset())

            # Time step t
            t = 0

            # Compute the new epsilon and alpha
            epsilon = self.get_epsilon(i_episode)
            alpha = self.get_alpha(i_episode)

            while True:

                # Render the env if requested
                if self.env_render:
                    self.env.render()

                # Take a step
                action_probs = policy(state, epsilon)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.process_state(next_state)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t
                stats.episode_epsilon[i_episode] = epsilon
                stats.episode_alpha[i_episode] = alpha

                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + self.discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                if done:
                    break

                state = next_state
                t += 1

        return Q, stats


def make_epsilon_greedy_policy(Q, num_actions):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    :param Q: A dictionary that maps from state -> action-values.
              Each value is a numpy array of length num_actions (see below)
    :param num_actions: Number of actions in the environment.
    :return: A function that takes the state as an argument and returns
             the probabilities for each action in the form of a numpy array of length num_actions.
    """

    def policy_fn(state, epsilon):
        A = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn



