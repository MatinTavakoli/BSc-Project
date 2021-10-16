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


# MODEL_PATH = 'models/DQNAgent/64x2- -34.00max- -98.60avg--141.00min-1632304191'
MODEL_PATH = 'models/SACAgent/conv_nn_(simple_reward)_stacked-policy_net- -10.00max- -10.00avg- -10.00min-1633114880'
# MODEL_PATH = 'models/SACAgent/conv_nn_global(simple_reward)-policy_net--200.00max--200.00avg--200.00min-1633459279'

if __name__ == '__main__':

    # Memory fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create environment
    env = CarEnv()

    # For agent speed measurements - keeps last 20 frametimes
    fps_counter = deque(maxlen=20)

    mode = 2  # mode = 1 for DQN, mode = 2 for SAC
    rrt_mode = False  # rrt_mode = False for SAC, rrt_mode = True for combined SAC and RRT

    # Load the model
    if mode == 1:
        model = load_model(MODEL_PATH)
    elif mode == 2:
        agent = SACAgent(rrt_mode=rrt_mode)
        agent.policy_net.load_state_dict(torch.load(MODEL_PATH))

    # Initialize predictions - first prediction takes longer as of initialization that has to be done
    # It's better to do a first prediction then before we start iterating over episode steps
    # model.predict(np.ones((1, env.im_height, env.im_width, 3)))  # commented for debugging purposes

    # Loop over episodes
    while True:

        print('Restarting episode')

        # Reset environment and get initial state
        if mode == 1:
            current_state = env.reset()
        elif mode == 2:
            if not rrt_mode:
                current_state = env.reset()
            else:
                current_state, current_x, current_y = env.reset(rrt_mode=True)
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
                qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape))[0]
                action = np.argmax(qs)

            elif mode == 2:
                if not rrt_mode:
                    action_dist = agent.policy_net.get_action(current_state).detach()
                else:
                    action_dist = agent.policy_net.get_action(current_state, rrt_mode=True, x_loc=current_x, y_loc=current_y).detach()
                action = np.argmax(action_dist)

            # Step environment (additional flag informs environment to not break an episode by time limit)
            if not rrt_mode:
                new_state, reward, done, _ = env.step(action)
            else:
                new_state, reward, done, new_x, new_y = env.step(action, rrt_mode=True)

            # Set current step for next loop iteration
            current_state = new_state
            if rrt_mode:
                current_x, current_y = new_x, new_y

            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)

            if mode == 1:
                action_dist = qs

            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{action_dist[0]:>5.2f}, {action_dist[1]:>5.2f}, {action_dist[2]:>5.2f}] {action}')

        # Destroy an actor at end of episode
        for actor in env.actor_list:
            actor.destroy()