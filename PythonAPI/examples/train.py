# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import glob
import os
import numpy as np
import sys
import random
import time
import cv2
import math

from keras import backend
import tensorflow as tf

from threading import Thread
from tqdm import tqdm

# ==============================================================================
# -- importing agents ----------------------------------------------------------
# ==============================================================================
from PythonAPI.examples.car_env import CarEnv
from PythonAPI.examples.dqn_agent import DQNAgent
from PythonAPI.examples.sac_agent import *

# ==============================================================================
# -- importing carla -----------------------------------------------------------
# ==============================================================================

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# ==============================================================================
# -- constants -----------------------------------------------------------------
# ==============================================================================

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480

EPISODE_LENGTH = 12
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_NETWORK_EVERY = 5

MEMORY_FRACTION = 0.6
MIN_REWARD = -100

NUM_OF_EPISODES = 75
DISCOUNT = 0.99
EPSILON = 1
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.001
AGGREGATE_STATE_EVERY = 10


if __name__ == "__main__":
    FPS = 20
    ep_rewards = [-100]

    random.seed(10)  # TODO: FIGURE OUT A BETTER SEED!
    np.random.seed(10)
    tf.set_random_seed(10)

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    if not os.path.isdir("models"):
        os.makedirs("models")

    env = CarEnv()

    mode = 2

    if mode == 1:

        agent = DQNAgent()

        trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
        trainer_thread.start()

        while not agent.training_initialized:
            time.sleep(0.01)

        agent.get_qs(np.ones((env.im_height, env.im_width, 3)))
        # # reloading (for debugging purposes)
        # MODEL_PATH = 'models/64x2(reloaded)- -30.00max- -38.10avg- -48.00min-1632311799'
        # # Load the model
        # model = load_model(MODEL_PATH)

        for episode in tqdm(range(1, NUM_OF_EPISODES + 1), ascii=True, unit="episodes"):
            env.collision_hist = []
            agent.tensorboard.step = episode
            episode_reward = 0
            step = 1
            current_state = env.reset()
            done = False
            episode_start = time.time()

            while True:
                if np.random.random() > EPSILON:
                    action = np.argmax(agent.get_qs(current_state))

                else:
                    action = np.random.randint(0, 3)
                    time.sleep(1 / FPS)

                print('action {} was selected'.format(action))

                new_state, reward, done, _ = env.step(action)
                episode_reward += reward

                agent.update_replay_memory((current_state, action, reward, new_state, done))

                step += 1

                if done:
                    break

            for actor in env.actor_list:
                actor.destroy()

            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STATE_EVERY or episode == 1:
                average_reward = sum(ep_rewards[-AGGREGATE_STATE_EVERY:]) / len(ep_rewards[-AGGREGATE_STATE_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATE_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATE_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                               epsilon=EPSILON)

                # saving good models
                if min_reward >= MIN_REWARD:
                    agent.model.save(
                        f'models/{agent.MODEL_NAME}-{max_reward:7.2f}max-{average_reward:7.2f}avg-{min_reward:7.2f}min-{int(time.time())}')

            if EPSILON > MIN_EPSILON:
                EPSILON *= EPSILON_DECAY
                EPSILON = max(MIN_EPSILON, EPSILON)

        agent.terminate = True
        trainer_thread.join()
        agent.model.save(
            f'models/{agent.MODEL_NAME}-{max_reward:7.2f}max-{average_reward:7.2f}avg-{min_reward:7.2f}min-{int(time.time())}')

    elif mode == 2:
        while frame_idx < max_frames:
            state = CarEnv.reset()
            episode_reward = 0

            for step in range(max_steps):
                if frame_idx > 1000:
                    action = policy_net.get_action(state).detach()
                    next_state, reward, done, _ = CarEnv.step(action)
                else:
                    action = np.random.randint(0, 3)
                    next_state, reward, done, _ = CarEnv.step(action)

                replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                frame_idx += 1

                if len(replay_buffer) > batch_size:
                    update(batch_size)

                if done:
                    break

            rewards.append(episode_reward)