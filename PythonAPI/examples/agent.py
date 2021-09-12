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

# ==============================================================================
# -- import carla --------------------------------------------------------------
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

IM_WIDTH = 640
IM_HEIGHT = 480


# ==============================================================================
# -- functions -----------------------------------------------------------------
# ==============================================================================

def process_sensory_data(data):
    data_arr = np.array(data.raw_data)
    data_pic = data_arr.reshape((IM_HEIGHT, IM_WIDTH, 4))[:, :, :3]  # we only want rgb!
    cv2.imshow("", data_pic)
    cv2.waitKey(1)
    print('reached here')
    return data_pic / 255.0  # normalizing sensory data for the neural network


actor_list = []

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(4.0)  # seconds

    world = client.get_world()  # world connection
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("bmw")[0]  # fetch the first model of bmw
    print("selected blueprint is {}".format(bp))

    vehicle_spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, vehicle_spawn_point)
    vehicle.set_autopilot(False)  # making sure its not in autopilot!
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)

    # sensory data
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_bp.set_attribute("fov", "110")

    camera_spawn_point = carla.Transform(carla.Location(x=5, z=2))  # TODO: fine-tune these values!
    sensor = world.spawn_actor(camera_bp, camera_spawn_point, attach_to=vehicle)
    actor_list.append(sensor)
    sensor.listen(lambda data: process_sensory_data(data))

    time.sleep(7)

finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")
