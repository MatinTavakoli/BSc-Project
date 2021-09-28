# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import os
import numpy as np
import random
import time
import cv2
import math
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Activation, AveragePooling2D, Flatten
from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras.callbacks import TensorBoard
from keras import backend
import tensorflow as tf

from threading import Thread
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from car_env import CarEnv

# ==============================================================================
# -- constants -----------------------------------------------------------------
# ==============================================================================

REPLAY_MEMORY_SIZE = 10_000
MIN_REPLAY_MEMORY_SIZE = 2_000
MINIBATCH_SIZE = 32
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_NETWORK_EVERY = 10
MODEL_NAME = "simple_nn"

max_frames = 40000
max_steps = 500
frame_idx = 0
rewards = []


# ==============================================================================
# -- sac agent class -----------------------------------------------------------
# ==============================================================================

class SACAgent:
    pass


# ==============================================================================
# -- replay buffer class --------------------------------------------------------
# ==============================================================================

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, minibatch_size):
        batch = random.sample(self.buffer, minibatch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# -- network classes TODO: make it convolutional!-------------------------------
# ==============================================================================

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        # self.linear1 = nn.Linear(state_dim, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, 1)
        #
        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)

        # pytorch (conv2d)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5, 5), padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=0)
        # self.conv2x_fully = nn.Linear(1123584, 1)  # TODO: don't hardcode this shit!
        # self.conv3x_stride2_fully = nn.Linear(247616, 1)  # TODO: don't hardcode this shit!
        self.conv3x_stride4_fully = nn.Linear(2560, 1)  # TODO: don't hardcode this shit!

    def forward(self, state):
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # x = self.linear3(x)

        # CNN
        avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(4, 4), padding=0)

        x = F.relu(self.conv1(state))
        x = avgpool(x)

        x = F.relu(self.conv2(x))
        x = avgpool(x)

        x = F.relu(self.conv3(x))
        x = avgpool(x)

        x = torch.flatten(x, 1)

        x = self.conv3x_stride4_fully(x)

        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_states, num_actions, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        # pytorch (linear)
        # self.linear1 = nn.Linear(num_states, hidden_size)  # TODO: why is this addition?!
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, num_actions)
        #
        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)

        # pytorch (conv2d)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5, 5), padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=0)
        # self.conv2x_fully = nn.Linear(1123584, num_actions)  # TODO: don't hardcode this shit!
        # self.conv3x_stride2_fully = nn.Linear(247616, num_actions)  # TODO: don't hardcode this shit!
        self.conv3x_stride4_fully = nn.Linear(2560, num_actions)  # TODO: don't hardcode this shit!

    def forward(self, state, action):
        # original
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # x = self.linear3(x)

        # CNN
        avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(4, 4), padding=0)

        x = F.relu(self.conv1(state))
        x = avgpool(x)

        x = F.relu(self.conv2(x))
        x = avgpool(x)

        x = F.relu(self.conv3(x))
        x = avgpool(x)

        x = torch.flatten(x, 1)

        x = self.conv3x_stride4_fully(x)

        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # self.linear1 = nn.Linear(num_inputs, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        #
        # self.mean_linear = nn.Linear(hidden_size, num_actions)
        # self.mean_linear.weight.data.uniform_(-init_w, init_w)
        # self.mean_linear.bias.data.uniform_(-init_w, init_w)
        #
        # self.log_std_linear = nn.Linear(hidden_size, num_actions)
        # self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        # self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        # pytorch (conv2d)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5, 5), padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=0)

        # self.conv2x_mean_fully = nn.Linear(1123584, num_actions)  # TODO: don't hardcode this shit!
        # self.conv2x_log_std_fully = nn.Linear(1123584, num_actions)  # TODO: don't hardcode this shit!

        # self.conv3x_stride2_mean_fully = nn.Linear(247616, num_actions)  # TODO: don't hardcode this shit!
        # self.conv3x_stride2_log_std_fully = nn.Linear(247616, num_actions)  # TODO: don't hardcode this shit!

        self.conv3x_stride4_mean_fully = nn.Linear(2560, num_actions)  # TODO: don't hardcode this shit!
        self.conv3x_stride4_log_std_fully = nn.Linear(2560, num_actions)  # TODO: don't hardcode this shit!

    def forward(self, state):
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        #
        # mean = self.mean_linear(x)
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # CNN
        avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(4, 4), padding=0)

        x = F.relu(self.conv1(state))
        x = avgpool(x)

        x = F.relu(self.conv2(x))
        x = avgpool(x)

        x = F.relu(self.conv3(x))
        x = avgpool(x)

        x = torch.flatten(x, 1)

        mean = self.conv3x_stride4_mean_fully(x)
        log_std = self.conv3x_stride4_log_std_fully(x)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        return action[0]


def update(minibatch_size, gamma=0.99, soft_tau=1e-2, ):
    state, action, reward, next_state, done = replay_memory.sample(minibatch_size)

    state = torch.FloatTensor(state).to(device).permute(0, 3, 1, 2)  # permutation needed for conv2d
    next_state = torch.FloatTensor(next_state).to(device).permute(0, 3, 1, 2)  # permutation needed for conv2d
    action = torch.IntTensor(action).to(device).type(torch.int64)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value1 = predicted_q_value1.gather(1, action.view(-1, 1)).view(-1)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_q_value2 = predicted_q_value2.gather(1, action.view(-1, 1)).view(-1)

    predicted_value = value_net(state).reshape(MINIBATCH_SIZE)

    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)
    new_sample_actions = torch.multinomial(new_action, 1, replacement=True)  # sampling from float action probabilities
    log_prob = log_prob.gather(1, new_sample_actions.view(-1, 1)).view(-1)

    # Training Q Function
    target_value = target_value_net(next_state).reshape(MINIBATCH_SIZE)
    target_q_value = reward + (1 - done) * gamma * target_value

    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())

    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()

    # Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action)).gather(1, new_sample_actions.view(-1, 1)).view(-1)
    target_value_func = (predicted_new_q_value - log_prob)

    print(predicted_value.shape)
    print(predicted_new_q_value.shape)
    print(log_prob.shape)
    print(target_value_func.shape)

    value_loss = value_criterion(predicted_value, target_value_func.detach())

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    # Training Policy Function
    policy_loss = (log_prob - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


state_dim = CarEnv.im_height * CarEnv.im_width * 3
action_dim = 3  # TODO: make it 9!

value_net = ValueNetwork(state_dim).to(device)
target_value_net = ValueNetwork(state_dim).to(device)

soft_q_net1 = SoftQNetwork(state_dim, action_dim).to(device)
soft_q_net2 = SoftQNetwork(state_dim, action_dim).to(device)

policy_net = PolicyNetwork(state_dim, action_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

value_criterion = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

value_lr = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
