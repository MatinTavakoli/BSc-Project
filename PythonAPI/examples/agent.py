import glob
import os
import sys
import random
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

actor_list = []

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)  # seconds

    world = client.get_world()  # world connection
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("bmw")[0]  # fetch the first model of bmw
    print("selected blueprint is {}".format(bp))

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(False)  # making sure its not in autopilot!
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)
    time.sleep(7)

finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")
