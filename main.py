import argparse
import gym

from algorithms.q_learning import QLearning
from utils.gym_wrapper import wrap_env
from algorithms.discretization_functions import discretize_cartpole_v0
from preprocessing.atari import imshow, preprocess_breakout


def main():
    parser = argparse.ArgumentParser(description='Reinforcement Learning playground using Gym Open AI')
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run '
                                                                         '(default: "CartPole-v0")')
    args = parser.parse_args()

    env = gym.make(args.env_id)

    env = wrap_env(env, args.env_id)

    buckets = (1, 1, 6, 12,)

    agent = QLearning(env, buckets=buckets, discretize_fn=discretize_cartpole_v0)

    Q, stats = agent.learn(500)

    print(Q)
    print(stats)

    env.close()


def test():
    env_name = "Breakout-v0"

    env = gym.make(env_name)
    frame = env.reset()
    imshow(frame)
    imshow(preprocess_breakout(frame))
    for i_episode in range(1):
        frame = env.reset()
        for t in range(10):
            env.render()
            imshow(preprocess_breakout(frame))
            action = env.action_space.sample()
            frame, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


if __name__ == "__main__":
    test()
    #main()
