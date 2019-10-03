import argparse
from os import path
import time
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from definitions import ROOT_DIR
from utils.gym_wrapper import make_env
from models.dqn import AtariDQN
from agents.experience_replay import ExperienceReplay
from agents.atari import AtariAgent
from utils.system_utils import get_path


ENV_NAME = "Breakout-v0"
MEAN_REWARD_BOUND = 100

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10 ** 4 * 1  # Maximum number of experiences stored in replay memory
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 1000  # How many frames in between syncing target DQN with behaviour DQN
LEARNING_STARTS = 50000  # Number of experiences to add to replay memory before training network

MODEL_SAVE_FOLDER = path.join(ROOT_DIR, 'pretrained-models', ENV_NAME)
MODEL = path.join(ROOT_DIR, 'pretrained-models/Breakout-v0/reward-2.08.pth')
RENDER_ENV = True
LOAD_MODEL = False

USE_CUDA = False


def main():
    EPSILON_DECAY = 10 ** 5
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.02

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=USE_CUDA, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=ENV_NAME,
                        help="Name of the environment, default=" + ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward to stop training, default={}".format(round(MEAN_REWARD_BOUND, 2)))
    parser.add_argument("-m", "--model", default=MODEL, help="Model file to load")
    parser.add_argument("-r", "--render", type=bool, default=RENDER_ENV,
                        help="Render the environment, default=" + str(RENDER_ENV))
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    print("ReplayMemory will require {}gb of GPU RAM".format(round(REPLAY_SIZE * 32 * 84 * 84 / 1e+9, 2)))

    env = make_env(args.env, True)
    policy_net = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = AtariDQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(log_dir=get_path(path.join(ROOT_DIR, 'logs/tensorboard')), comment="-" + args.env)

    print(policy_net)

    replay_memory = ExperienceReplay(REPLAY_SIZE)
    agent = AtariAgent(env, replay_memory)
    epsilon = EPSILON_START

    if LOAD_MODEL:
        policy_net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
        target_net.load_state_dict(policy_net.state_dict())
        print("Models loaded from disk!")
        # Lower exploration rate IMPORTANT
        EPSILON_START = EPSILON_FINAL

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    best_mean_reward = None
    frame_idx = 0
    timestep_frame = 0
    timestep = time.time()

    while True:
        if args.render:
            env.render()

        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY)

        reward = agent.play_step(policy_net, epsilon, device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - timestep_frame) / (time.time() - timestep)
            timestep_frame = frame_idx
            timestep = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("{} frames: done {} games, mean reward {}, eps {}, speed {} f/s".format(
                frame_idx, len(total_rewards), round(mean_reward, 3), round(epsilon, 2), round(speed, 2)))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward or len(total_rewards) % 25 == 0:
                # Save model if it has performed better than all the previous ones
                save_path = get_path(MODEL_SAVE_FOLDER)
                torch.save(policy_net.state_dict(), path.join(save_path, 'reward-' + str(mean_reward) + '.pth'))
                if best_mean_reward is not None:
                    print("New best mean reward {} -> {}, model saved".format(round(best_mean_reward, 3),
                                                                              round(mean_reward, 3)))
                best_mean_reward = mean_reward
            if mean_reward > args.reward and len(total_rewards) > 10:
                print("Game solved in {} frames! Average score of {}".format(frame_idx, mean_reward))
                break

        if len(replay_memory) < LEARNING_STARTS:
            continue

        if frame_idx % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        optimizer.zero_grad()
        batch = replay_memory.sample(BATCH_SIZE)
        loss_t = agent.calculate_loss(batch, policy_net, target_net, device=device)
        loss_t.backward()
        optimizer.step()

    env.close()
    writer.close()


if __name__ == '__main__':
    main()
