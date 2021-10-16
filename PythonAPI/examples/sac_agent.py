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
from ModifiedTensorBoard import ModifiedTensorBoard

from threading import Thread
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from car_env import CarEnv, PATH

# ==============================================================================
# -- constants -----------------------------------------------------------------
# ==============================================================================

REPLAY_MEMORY_SIZE = 10_000
MIN_REPLAY_MEMORY_SIZE = 2_000
# PREDICTION_BATCH_SIZE = 1
# TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_NETWORK_EVERY = 5

NUM_OF_EPISODES = 10
MINIBATCH_SIZE = 32
MIN_REWARD = -100
DISCOUNT = 0.99


# ==============================================================================
# -- sac agent class -----------------------------------------------------------
# ==============================================================================

class SACAgent:

    def __init__(self, rrt_mode=False):
        self.num_of_episodes = NUM_OF_EPISODES
        self.minibatch_size = MINIBATCH_SIZE
        self.min_reward = MIN_REWARD

        state_dim = CarEnv.im_height * CarEnv.im_width * 3
        action_dim = 3  # TODO: make it 9!

        if not rrt_mode:
            MODEL_NAME = "SAC"  # used for simple SAC
        else:
            MODEL_NAME = "SAC_RRT"  # used for SAC + RRT

        self.model_name = MODEL_NAME

        self.value_net = ValueNetwork(state_dim, rrt_mode=rrt_mode).to(device)
        self.target_value_net = ValueNetwork(state_dim, rrt_mode=rrt_mode).to(device)

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, rrt_mode=rrt_mode).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, rrt_mode=rrt_mode).to(device)

        self.policy_net = PolicyNetwork(state_dim, action_dim, rrt_mode=rrt_mode).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        value_lr = 3e-4
        soft_q_lr = 3e-4
        policy_lr = 3e-4

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(
            log_dir='logs/{}/{}-{}-{}-NEGATIVE{}'.format(self.__class__.__name__, MODEL_NAME, int(time.time()),
                                                         self.num_of_episodes,
                                                         abs(MIN_REWARD)))

    def update(self, minibatch_size, rrt_mode=False, gamma=0.99, soft_tau=1e-2, ):
        if not rrt_mode:
            state, action, reward, next_state, done = self.replay_memory.sample(minibatch_size)
        else:
            state, action, reward, next_state, done, x, y, next_x, next_y = self.replay_memory.sample(minibatch_size, rrt_mode=True)

        state = torch.FloatTensor(state).to(device).permute(0, 3, 1, 2)  # permutation needed for conv2d
        next_state = torch.FloatTensor(next_state).to(device).permute(0, 3, 1, 2)  # permutation needed for conv2d
        action = torch.IntTensor(action).to(device).type(torch.int64)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        if rrt_mode:
            x = torch.FloatTensor(x).unsqueeze(1).to(device)
            y = torch.FloatTensor(y).unsqueeze(1).to(device)
            next_x = torch.FloatTensor(next_x).unsqueeze(1).to(device)
            next_y = torch.FloatTensor(next_y).unsqueeze(1).to(device)

        if not rrt_mode:
            predicted_q_value1 = self.soft_q_net1(state, action)
            predicted_q_value2 = self.soft_q_net2(state, action)
            predicted_value = self.value_net(state).reshape(self.minibatch_size)
            new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state)
            target_value = self.target_value_net(next_state)
        else:
            predicted_q_value1 = self.soft_q_net1(state, action, rrt_mode=True, x_loc=x, y_loc=y)
            predicted_q_value2 = self.soft_q_net2(state, action, rrt_mode=True, x_loc=x, y_loc=y)
            predicted_value = self.value_net(state, rrt_mode=True, x_loc=x, y_loc=y).reshape(self.minibatch_size)
            new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state, rrt_mode=True, x_loc=x, y_loc=y)
            target_value = self.target_value_net(next_state, rrt_mode=True, x_loc=next_x, y_loc=next_y)

        predicted_q_value1 = predicted_q_value1.gather(1, action.view(-1, 1)).view(-1)
        predicted_q_value2 = predicted_q_value2.gather(1, action.view(-1, 1)).view(-1)

        new_action = (new_action + 1) / 2  # due to tanh activation, need to bring it in the (0, 1) interval (TODO: is this a bunch of bull?!)
        new_sample_actions = torch.multinomial(new_action, 1, replacement=True)  # sampling from float action probabilities
        log_prob = log_prob.gather(1, new_sample_actions.view(-1, 1)).view(-1)

        # Training Q Function
        target_q_value = (reward + (1 - done) * gamma * target_value).reshape(self.minibatch_size)

        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Value Function
        if not rrt_mode:
            predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action)).gather(1, new_sample_actions.view(-1, 1)).view(-1)
        else:
            predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action, rrt_mode=True, x_loc=x, y_loc=y), self.soft_q_net2(state, new_action, rrt_mode=True, x_loc=x, y_loc=y)).gather(1, new_sample_actions.view(-1, 1)).view(-1)
        target_value_func = (predicted_new_q_value - log_prob)

        value_loss = self.value_criterion(predicted_value, target_value_func.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        # Training Policy Function
        policy_loss = (log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

# ==============================================================================
# -- network classes -----------------------------------------------------------
# ==============================================================================

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, rrt_mode=False, init_w=3e-3):
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

        if not rrt_mode:
            self.conv3x_stride4_fully = nn.Linear(2560, 1)  # TODO: don't hardcode this shit!
        else:
            self.conv3x_stride5_fully1 = nn.Linear(512, 5)  # TODO: don't hardcode this shit!
            self.conv3x_stride5_fully2 = nn.Linear(8, 1)  # TODO: don't hardcode this shit!

    def forward(self, state, rrt_mode=False, x_loc=None, y_loc=None):
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # x = self.linear3(x)

        # CNN
        if not rrt_mode:
            avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(4, 4), padding=0)
        else:
            avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(5, 5), padding=0)

        x = F.relu(self.conv1(state))
        x = avgpool(x)

        x = F.relu(self.conv2(x))
        x = avgpool(x)

        x = F.relu(self.conv3(x))
        x = avgpool(x)

        x = torch.flatten(x, 1)

        if not rrt_mode:
            x = self.conv3x_stride4_fully(x)

        else:
            x = self.conv3x_stride5_fully1(x)
            global_planner_one_hot = calculate_one_hot_batch(x_loc, y_loc, MINIBATCH_SIZE)
            x = torch.cat([x, global_planner_one_hot], 1)
            x = self.conv3x_stride5_fully2(x)

        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_states, num_actions, rrt_mode=False, init_w=3e-3):
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

        if not rrt_mode:
            self.conv3x_stride4_fully = nn.Linear(2560, num_actions)  # TODO: don't hardcode this shit!
        else:
            self.conv3x_stride5_fully1 = nn.Linear(512, 5)  # TODO: don't hardcode this shit!
            self.conv3x_stride5_fully2 = nn.Linear(8, num_actions)  # TODO: don't hardcode this shit!

    def forward(self, state, action, rrt_mode=False, x_loc=None, y_loc=None):
        # original
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        # x = self.linear3(x)

        # CNN
        print(rrt_mode)
        if not rrt_mode:
            avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(4, 4), padding=0)
        else:
            avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(5, 5), padding=0)

        x = F.relu(self.conv1(state))
        x = avgpool(x)

        x = F.relu(self.conv2(x))
        x = avgpool(x)

        x = F.relu(self.conv3(x))
        x = avgpool(x)

        x = torch.flatten(x, 1)

        if not rrt_mode:
            x = self.conv3x_stride4_fully(x)

        else:
            x = self.conv3x_stride5_fully1(x)
            global_planner_one_hot = calculate_one_hot_batch(x_loc, y_loc, MINIBATCH_SIZE)
            x = torch.cat([x, global_planner_one_hot], 1)
            x = self.conv3x_stride5_fully2(x)

        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, rrt_mode=False, init_w=3e-3, log_std_min=-20, log_std_max=2):
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

        if not rrt_mode:
            # Simple SAC
            self.conv3x_stride4_mean_fully = nn.Linear(2560, num_actions)  # TODO: don't hardcode this shit!
            self.conv3x_stride4_log_std_fully = nn.Linear(2560, num_actions)  # TODO: don't hardcode this shit!

        else:
            # SAC + RRT (old)
            # self.conv3x_stride4_mean_fully1 = nn.Linear(2560, 10)  # TODO: don't hardcode this shit!
            # self.conv3x_stride4_mean_fully2 = nn.Linear(13, num_actions)  # TODO: don't hardcode this shit!
            # self.conv3x_stride4_log_std_fully1 = nn.Linear(2560, 10)  # TODO: don't hardcode this shit!
            # self.conv3x_stride4_log_std_fully2 = nn.Linear(13, num_actions)  # TODO: don't hardcode this shit!

            # SAC + RRT (new)
            self.conv3x_stride5_mean_fully1 = nn.Linear(512, 5)  # TODO: don't hardcode this shit!
            self.conv3x_stride5_mean_fully2 = nn.Linear(8, num_actions)  # TODO: don't hardcode this shit!
            self.conv3x_stride5_log_std_fully1 = nn.Linear(512, 5)  # TODO: don't hardcode this shit!
            self.conv3x_stride5_log_std_fully2 = nn.Linear(8, num_actions)  # TODO: don't hardcode this shit!

    def forward(self, state, rrt_mode=False, x_loc=None, y_loc=None):
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        #
        # mean = self.mean_linear(x)
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # CNN
        if not rrt_mode:
            avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(4, 4), padding=0)
        else:
            avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=(5, 5), padding=0)

        x = F.relu(self.conv1(state))
        x = avgpool(x)

        x = F.relu(self.conv2(x))
        x = avgpool(x)

        x = F.relu(self.conv3(x))
        x = avgpool(x)

        x = torch.flatten(x, 1)

        if not rrt_mode:
            mean = self.conv3x_stride4_mean_fully(x)
            log_std = self.conv3x_stride4_log_std_fully(x)

        else:
            global_planner_one_hot = calculate_one_hot_batch(x_loc, y_loc, state.shape[0])

            mean = self.conv3x_stride5_mean_fully1(x)
            mean = torch.cat([mean, global_planner_one_hot], 1)
            mean = self.conv3x_stride5_mean_fully2(mean)

            log_std = self.conv3x_stride5_log_std_fully1(x)
            log_std = torch.cat([log_std, global_planner_one_hot], 1)
            log_std = self.conv3x_stride5_log_std_fully2(log_std)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, rrt_mode=False, x_loc=None, y_loc=None, epsilon=1e-6):
        if not rrt_mode:
            mean, log_std = self.forward(state)
        else:
            mean, log_std = self.forward(state, rrt_mode=rrt_mode, x_loc=x_loc, y_loc=y_loc)

        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, rrt_mode=False, x_loc=None, y_loc=None):
        state = torch.FloatTensor(state).unsqueeze(0).to(device).permute(0, 3, 1, 2)  # permutation needed for conv2d
        if not rrt_mode:
            mean, log_std = self.forward(state)
        else:
            mean, log_std = self.forward(state, rrt_mode=rrt_mode, x_loc=x_loc, y_loc=y_loc)

        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        return action[0]


# ==============================================================================
# -- replay buffer class --------------------------------------------------------
# ==============================================================================

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, rrt_mode=False, x=None, y=None, next_x=None, next_y=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        if not rrt_mode:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done, x, y, next_x, next_y)

        self.position = (self.position + 1) % self.capacity

    def sample(self, minibatch_size, rrt_mode=False):
        batch = random.sample(self.buffer, minibatch_size)
        if not rrt_mode:
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done
        else:
            state, action, reward, next_state, done, x, y, next_x, next_y = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done, x, y, next_x, next_y

    def __len__(self):
        return len(self.buffer)


def calculate_one_hot_batch(x_loc, y_loc, minibatch_size):
    immediate_goal = PATH[0]
    im_goal_x = torch.tensor([immediate_goal[0]])
    im_goal_y = torch.tensor([immediate_goal[1]])

    right_x_one_hot = torch.where(x_loc > im_goal_x, 1, 0)
    right_y_one_hot = torch.where(y_loc < im_goal_y, 1, 0)
    right_one_hot = torch.logical_and(right_x_one_hot, right_y_one_hot)

    left_x_one_hot = torch.where(x_loc < im_goal_x, 1, 0)
    left_y_one_hot = torch.where(y_loc > im_goal_y, 1, 0)
    left_one_hot = torch.logical_and(left_x_one_hot, left_y_one_hot)

    check1 = torch.where(right_one_hot, 0, 1)
    check2 = torch.where(left_one_hot, 0, 1)

    straight_one_hot = torch.logical_and(check1, check2)

    one_hot = torch.cat([right_one_hot.reshape(minibatch_size, 1), straight_one_hot.reshape(minibatch_size, 1), left_one_hot.reshape(minibatch_size, 1)], 1)

    return one_hot