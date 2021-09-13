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


# ==============================================================================
# -- agent class ---------------------------------------------------------------
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
