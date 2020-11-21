import collections

import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from wimblepong.dqn_net import DQN
import torch
from torch.utils.tensorboard import SummaryWriter
from wimblepong.agent import ExperienceReplay, Agent
import torch.optim as optim
import torch.nn as nn


import cv2

def _rgb2gray(rgb):
    res = cv2.resize(rgb[...,:3], dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
    res = np.matmul(res[...,:3], np.array([0.2989, 0.5870, 0.1140]))
    res = np.reshape(res, (84, 84, 1))
    return res

def preprocess(frame):
    if frame.size == 200 * 200 * 3:
        img = np.reshape(frame, [200, 200, 3]).astype(np.float32)
    else:
        assert False, "Unknown resolution."
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true",
                    help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
# env = gym.make("WimblepongVisualMultiplayer-v0")
DEFAULT_ENV_NAME = "WimblepongVisualSimpleAI-v0"
env = gym.make(DEFAULT_ENV_NAME)
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Number of episodes/games to play
episodes = 1000
D = 200 * 200
# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
# policy = Policy(D, 3)
# player = Agent(policy)
# player = Agent(env, player_id)

# Housekeeping
states = []
win1 = 0
win_hist = []
device = torch.device("cpu")
net = DQN((4,84,84), env.action_space.n).to(device)
target_net = DQN((4,84,84), env.action_space.n).to(device)
writer = SummaryWriter(comment="-" + DEFAULT_ENV_NAME)
replay_size = 10000
# replay_size = 500

buffer_g = ExperienceReplay(replay_size)
agent = Agent(env, buffer_g)
eps_start=1.0
epsilon = eps_start
learning_rate = 1e-4
eps_decay=.999985
eps_min=0.02
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
total_rewards = []
frame_idx = 0
best_mean_reward = None
MEAN_REWARD_BOUND = 19.0
replay_start_size = 10000
# replay_start_size = 500

batch_size = 32
gamma = 0.99
sync_target_frames = 1000
done = False
while True:
    epsilon = max(epsilon * eps_decay, eps_min)

    frame_idx += 1
    reward, done = agent.play_step(net, epsilon, device=device)
    print(reward, done)
    if reward is not None:
        total_rewards.append(reward)

        mean_reward = np.mean(total_rewards[-100:])

        print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (
            frame_idx, len(total_rewards), mean_reward, epsilon))

        writer.add_scalar("epsilon", epsilon, frame_idx)
        writer.add_scalar("reward_100", mean_reward, frame_idx)
        writer.add_scalar("reward", reward, frame_idx)

        if best_mean_reward is None or best_mean_reward < mean_reward:

            torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best.dat")
            best_mean_reward = mean_reward
            if best_mean_reward is not None:
                print("Best mean reward updated %.3f" % (best_mean_reward))

        if mean_reward > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % frame_idx)
            break



    if len(buffer_g) < replay_start_size:
            print('here')
            continue



    batch = buffer_g.sample(batch_size)
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    print('state_action_values.shape', state_action_values.shape)
    next_state_values = target_net(next_states_v).max(1)[0]

    next_state_values[done_mask] = 0.0

    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v

    loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss_t.backward()
    optimizer.step()
    if frame_idx % sync_target_frames == 0:
        target_net.load_state_dict(net.state_dict())


writer.close()

