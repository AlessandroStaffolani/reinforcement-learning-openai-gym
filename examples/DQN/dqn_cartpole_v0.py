import gym
from os import path
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.optim as optim

from preprocessing.cartpole import get_screen
from algorithms.dqn import DQN, select_action, optimize_model
from algorithms.replay_memory import ReplayMemory
from utils.plotting import plot_durations
from utils.gym_wrapper import wrap_env
from definitions import ROOT_DIR

resume_model = False
model_path = path.join(ROOT_DIR, 'models/CartPole-v0/policy_net.pt')

# init the env
env = gym.make('CartPole-v0').unwrapped

env = wrap_env(env, 'CartPole-v0')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Transition Tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Test the screen
env.reset()
plt.figure()
plt.imshow(get_screen(env, device).cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen(env, device)
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

# Init the policy and the target net
policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
if resume_model:
    policy_net.load_state_dict(torch.load(model_path))

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Init the optimizer
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000, Transition)

steps_done = 0

episode_durations = []

num_episodes = 200

for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen(env, device)
    current_screen = get_screen(env, device)
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action, steps_done = select_action(
            state=state,
            policy_net=policy_net,
            n_actions=n_actions,
            device=device,
            steps_done=steps_done,
            eps_start=EPS_START,
            eps_end=EPS_END,
            eps_decay=EPS_DECAY
        )

        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env, device)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            memory=memory,
            Transition=Transition,
            device=device,
            batch_size=BATCH_SIZE,
            gamma=GAMMA
        )
        if done:
            episode_durations.append(t + 1)
            plot_durations(episode_durations, is_ipython)
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


# Print policy_net's state_dict
print("PolicyNet's state_dict:")
for param_tensor in policy_net.state_dict():
    print(param_tensor, "\t", policy_net.state_dict()[param_tensor].size())

torch.save(policy_net.state_dict(), path.join(ROOT_DIR, model_path))

env.render()
env.close()
plt.ioff()
plt.show()

