import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import namedtuple

import torch

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "episode_epsilon", "episode_alpha"])


def plot_episode_stats(stats, n_episodes, smoothing_window=10, noshow=False, goal_value=None, fig_size=(15, 8), ada_divisor=25, show_params=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=fig_size)
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=fig_size)
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
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=fig_size)
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    if show_params:
        # Plot Epsilon over episode
        fig4 = plt.figure(figsize=(15, 8))
        plt.plot(np.arange(n_episodes), stats.episode_epsilon)
        plt.xlabel("Episode t")
        plt.ylabel("Epsilon")
        plt.title("Epsilon over episode using ada_divisor of {}".format(ada_divisor))
        if noshow:
            plt.close(fig4)
        else:
            plt.show(fig4)

        # Plot Epsilon over episode
        fig5 = plt.figure(figsize=(15, 8))
        plt.plot(np.arange(n_episodes), stats.episode_alpha)
        plt.xlabel("Episode t")
        plt.ylabel("Alpha")
        plt.title("Alpha over episode using ada_divisor of {}".format(ada_divisor))
        if noshow:
            plt.close(fig5)
        else:
            plt.show(fig5)

        return fig1, fig2, fig3, fig4, fig5


def plot_durations(episode_durations, is_ipython):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        from IPython import display
        display.clear_output(wait=True)
        display.display(plt.gcf())