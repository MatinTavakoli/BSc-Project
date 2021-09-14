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
from collections import deque

from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import backend
import tensorflow as tf

from threading import Thread
from tqdm import tqdm

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

EPISODE_LENGTH = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_NETWORK_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.6
MIN_REWARD = -100

NUM_OF_EPISODES = 100
DISCOUNT = 0.99
EPSILON = 1
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.001
AGGREGATE_STATE_EVERY = 10

# ==============================================================================
# -- customizing tensorboard ---------------------------------------------------
# ==============================================================================

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

# ==============================================================================
# -- env class -----------------------------------------------------------------
# ==============================================================================

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)  # seconds

        self.world = self.client.get_world()  # world connection
        self.blueprint_library = self.world.get_blueprint_library()

        self.model = self.blueprint_library.filter("bmw")[0]  # fetch the first model of bmw

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        # spawn vehicle
        self.vehicle_spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model, self.vehicle_spawn_point)
        self.vehicle.set_autopilot(False)  # making sure its not in autopilot!
        self.actor_list.append(self.vehicle)

        # camera sensory data
        self.camera = self.blueprint_library.find("sensor.camera.rgb")
        self.camera.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.camera.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.camera.set_attribute("fov", "110")

        # spawn camera
        self.camera_spawn_point = carla.Transform(carla.Location(x=5, z=2))  # TODO: fine-tune these values!
        self.camera_sensor = self.world.spawn_actor(self.camera, self.camera_spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.camera_sensor)
        self.camera_sensor.listen(lambda data: self.process_camera_sensory_data(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))
        time.sleep(3)

        # collision sensor
        collison_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collison_sensor, self.camera_spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.process_collision_sensory_data(event))

        while self.front_camera == None:
            time.sleep(0.01)  # wait a little

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

        return self.front_camera

    def process_camera_sensory_data(self, data):
        data_arr = np.array(data.raw_data)
        data_pic = data_arr.reshape((self.im_height, self.im_width, 4))[:, :, :3]  # we only want rgb!
        if self.SHOW_CAM:
            cv2.imshow("", data_pic)
            cv2.waitKey(1)
        self.front_camera = data_pic

    def process_collision_sensory_data(self, event):
        self.collision_hist.append(event)  # add the accident to the list

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))

        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0 * self.STEER_AMT))

        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        # TODO: change the conditions!

        # if we had a crash
        if len(self.collision_hist) != 0:
            done = True
            reward = -100

        # the car is moving too slow
        elif kmh < 50:
            done = False
            reward = -1

        else:
            done = False
            reward = 1  # encourage the vehicle's performance!

        #  terminating the episode (no reward)
        if self.episode_start + EPISODE_LENGTH < time.time():
            done = True

        return self.front_camera, reward, done, None


# ==============================================================================
# -- agent class ---------------------------------------------------------------
# ==============================================================================

class DQNAgent:

    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir='logs/{}-{}'.format(MODEL_NAME, int(time.time())))
        self.target_update_counter = 0
        self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:  # i.e., we're not ready to train yet!
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # main model
        current_states = np.array([transition[0] for transition in minibatch])
        current_states /= 255  # normalizing for the neural network
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        # target model
        new_current_states = np.array([transition[3] for transition in minibatch])
        new_current_states /= 255  # normalizing for the neural network
        with self.graph.as_default():
            future_qs_list = self.model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        for index, (current_state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(np.array(X) / 255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_NETWORK_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X, y, verbose=0, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

if __name__ == "__main__":
    FPS = 20
    ep_rewards = [-100]

    random.seed(1)
    np.random.seed(1)
    tf.set_random_seed(1)

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    if not os.path.isdir("models"):
        os.makedirs("models")

    agent = DQNAgent()
    env = CarEnv()

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()

    while not agent.training_initialized:
        time.sleep(0.01)

    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

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

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward

            agent.update_replay_memory((current_state, action, new_state, reward, done))

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
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=EPSILON)

            # saving good models
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}-{max_reward:7.2f}max-{average_reward:7.2f}avg-{min_reward:7.2f}min-{int(time.time())}')

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)

    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}-{max_reward:7.2f}max-{average_reward:7.2f}avg-{min_reward:7.2f}min-{int(time.time())}')