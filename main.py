import argparse
import gym

from algorithms.q_learning import QLearning
from utils.gym_wrapper import wrap_env


def main():
    parser = argparse.ArgumentParser(description='Reinforcement Learning playground using Gym Open AI')
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run '
                                                                         '(default: "CartPole-v0")')
    args = parser.parse_args()

    env = gym.make(args.env_id)

    env = wrap_env(env, args.env_id)

    agent = QLearning(env, env_render=True)

    Q, stats = agent.learn(10)

    print(Q)
    print(stats)

    env.close()


if __name__ == "__main__":
    main()