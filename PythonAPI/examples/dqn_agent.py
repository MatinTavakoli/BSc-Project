# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import numpy as np
import random
import time
from collections import deque

from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Activation, AveragePooling2D, Flatten
from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
import tensorflow as tf
from ModifiedTensorBoard import ModifiedTensorBoard

# ==============================================================================
# -- constants -----------------------------------------------------------------
# ==============================================================================

IM_WIDTH = 640
IM_HEIGHT = 480

EPISODE_LENGTH = 12
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_NETWORK_EVERY = 5
MODEL_NAME = "64x2(noturnv2)"

MIN_REWARD = -100

NUM_OF_EPISODES = 75
DISCOUNT = 0.99


# ==============================================================================
# -- dqn agent class -----------------------------------------------------------
# ==============================================================================

class DQNAgent:

    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir='logs/{}-{}-{}-NEGATIVE{}'.format(MODEL_NAME, int(time.time()), NUM_OF_EPISODES, abs(MIN_REWARD)))
        self.target_update_counter = 0
        self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        # base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))
        #
        # x = base_model.output
        # x = GlobalAveragePooling2D()(x)
        #
        # predictions = Dense(3, activation="linear")(x)
        # model = Model(inputs=base_model.input, outputs=predictions)
        # model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])

        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=(IM_HEIGHT, IM_WIDTH, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        # model.add(Conv2D(64, (3, 3), padding='same'))
        # model.add(Activation('relu'))
        # model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        model.add(Flatten())

        x = model.output
        predictions = Dense(3, activation='sigmoid')(x)
        model = Model(input=model.input, output=predictions)
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
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        # target model
        new_current_states = np.array([transition[3] for transition in minibatch])
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

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
            self.model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
                           callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_NETWORK_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

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