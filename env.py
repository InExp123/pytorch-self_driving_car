import random
import time
import numpy as np
import cv2
import math
import settings as st
import sys

class CarEnv:
    SHOW_CAM = st.SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = st.IM_WIDTH
    im_height = st.IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = st.carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", "{}".format(self.im_width))
        self.rgb_cam.set_attribute("image_size_y", "{}".format(self.im_height))
        self.rgb_cam.set_attribute("fov", "110")

        transform = st.carla.Transform(st.carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(st.carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(st.carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        # print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        # print("i3.shape: ",i3.shape)
        # print("np.transpose(i3).shape: ",i3.transpose(2, 0, 1).shape)
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        # sys.exit("Exit from process_img")
        self.front_camera = i3.transpose(2, 0, 1)

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(st.carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(st.carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(st.carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 35:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + st.SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None

