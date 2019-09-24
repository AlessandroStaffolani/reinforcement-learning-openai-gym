import numpy as np
import sys
from collections import defaultdict

from utils.plotting import EpisodeStats


class QLearning:

    def __init__(self,
                 env,
                 discount_factor=1.0,
                 alpha=0.5,
                 epsilon=0.1,
                 env_render=False,
                 env_wrapper=False
                 ):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        :param env: OpenAI environment.
        :param discount_factor: Gamma discount factor. Float between [0, 1].
        :param alpha: Temporal-Difference (TD) learning rate. Float between (0, 1].
        :param epsilon: Chance to sample a random action. Float between (0, 1].
        :param env_render: Wheter render the env or not. Boolean default False.
        """
        self.env = env
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self.env_render = env_render
        self.env_wrapper = env_wrapper
        # TODO check the value of the parameters

    def __init_action_value_function(self):
        """
        Init the action-value function Q
        """
        return defaultdict(lambda: np.zeros(self.env.action_space.n))

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
                             episode_rewards=np.zeros(num_episodes))

        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, self.epsilon, self.env.action_space.n)

        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()

            # Get the first random state
            state = self.env.reset()

            # Time step t
            t = 0

            while True:

                # Render the env if requested
                if self.env_render:
                    self.env.render()

                # Take a step
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = self.env.step(action)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + self.discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += self.alpha * td_delta

                if done:
                    break

                state = next_state
                t += 1

        return Q, stats


def make_epsilon_greedy_policy(Q, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    :param Q: A dictionary that maps from state -> action-values.
              Each value is a numpy array of length num_actions (see below)
    :param epsilon: The probability to select a random action. Float between (0, 1].
    :param num_actions: Number of actions in the environment.
    :return: A function that takes the state as an argument and returns
             the probabilities for each action in the form of a numpy array of length num_actions.
    """

    def policy_fn(state):
        A = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn



