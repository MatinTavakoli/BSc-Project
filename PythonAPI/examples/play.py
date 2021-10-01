import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from car_env import CarEnv
from sac_agent import *
from train import MEMORY_FRACTION


# MODEL_PATH = 'models/64x2(noturn)- -52.00max- -82.00avg--106.00min-1632316132'
MODEL_PATH = 'models/SACAgent/conv_nn_(simple_reward)- -10.00max- -10.00avg- -10.00min-1633080230'

if __name__ == '__main__':

    # Memory fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create environment
    env = CarEnv()

    # For agent speed measurements - keeps last 20 frametimes
    fps_counter = deque(maxlen=20)

    mode = 2

    # Load the model
    if mode == 1:
        model = load_model(MODEL_PATH)
    elif mode == 2:
        agent = SACAgent()
        # agent.policy_net.load_state_dict(torch.load(MODEL_PATH))

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    # model.predict(np.ones((1, env.im_height, env.im_width, 3)))  # commented for debugging purposes

    # Loop over episodes
    while True:

        print('Restarting episode')

        # Reset environment and get initial state
        current_state = env.reset()
        env.collision_hist = []

        done = False

        # Loop over steps
        while True:

            # Show current frame
            cv2.imshow(f'Agent - preview', current_state)
            cv2.waitKey(1)

            # For FPS counter
            step_start = time.time()

            if mode == 1:
                # Predict an action based on current observation space
                print(current_state)
                qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape))[0]
                print(qs)
                # action = np.argmax(qs)
                action = 1

            elif mode == 2:
                action_dist = agent.policy_net.get_action(current_state).detach()
                action = np.argmax(action_dist)

            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, _ = env.step(action)

            # Set current step for next loop iteration
            current_state = new_state

            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{action_dist[0]:>5.2f}, {action_dist[1]:>5.2f}, {action_dist[2]:>5.2f}] {action}')

        # Destroy an actor at end of episode
        for actor in env.actor_list:
            actor.destroy()