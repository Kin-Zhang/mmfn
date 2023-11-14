import os
import time
import cv2
import carla
from collections import deque

import torch
import carla
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
from mmfn_utils.models.model_img import MMFN
from mmfn_utils.datasets.config import GlobalConfig
from mmfn_utils.datasets.dataloader import scale_and_crop_image, lidar_to_histogram_features, transform_2d_points
from expert_agent.common.carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
import expert_agent.common.utils as utils

from planner import RoutePlanner
import math

def get_entry_point():
	return 'MMFNAgent'


class MMFNAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.lidar_processed = list()
		self.track = autonomous_agent.Track.MAP
		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.input_buffer = {'rgb': deque(), 'lidar': deque(), 'gps': deque(), 'thetas': deque(), 'opendrive': deque(), 'radar': deque()}

		self.config = GlobalConfig()
		self.net = MMFN(self.config, 'cuda')
		print('loading model')
		# for DDP model load use
		state_dict = torch.load(os.path.join(self.config_path.model_path, 'best_model.pth'), map_location=torch.device('cpu'))
		pretrained_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
		self.net.load_state_dict(pretrained_dict)
		# from collections import OrderedDict
		# new_state_dict = OrderedDict()

		# for k, v in state_dict.items():
		# 	if 'module' not in k:
		# 		k = 'module.'+k
		# 	else:
		# 		k = k.replace('features.module.', 'module.features.')
		# 	new_state_dict[k]=v
		# self.net.load_state_dict(new_state_dict)
		# self.net.load_state_dict(torch.load(os.path.join(self.config_path.model_path, 'best_model.pth')))
		print('load model success')
		self.net.cuda()
		self.net.eval()
		
		self.imu_data = utils.imu_msg()
		self.agent_pose = utils.LocalizationOperator()
		self.prev_lidar = None

	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):
		return [
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb'
					},
				{   
					'type': 'sensor.lidar.ray_cast',
					'x': 1.3, 'y': 0.0, 'z': 2.5,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
					'id': 'lidar'
					},
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
					},
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'speed'
					},
				{
					'type': 'sensor.opendrive_map', 
					'reading_frequency': 20,  
					'id': 'opendrive'
					},
				{
					'type': 'sensor.other.radar', 
					'x': 2.8, 'y': 0.0, 'z': 1.00, 
					'roll': 0.0, 'pitch': 5.0, 'yaw': 0.0, 
					'fov': 35, 'id': 'radar_front'
					},
				{
					'type': 'sensor.other.radar', 
					'x': -2.8, 'y': 0.0, 'z': 1.00, 
					'roll': 0.0, 'pitch': 5.0, 'yaw': -180, 
					'fov': 35, 'id': 'radar_rear'
					},
				]

	def save_map(self, input_data):
			self.CarlaMap = carla.Map('map', input_data['opendrive'][1]['opendrive'])
			self.birdview_producer = BirdViewProducer(
				self.CarlaMap,
				target_size=PixelDimensions(width=256, height=256),
				render_lanes_on_junctions=True,
				pixels_per_meter=8,
				crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
			)

	def tick(self, input_data, timestamp):
		# 只保存第一次map
		if self.step == -1:
			self.save_map(input_data)

		self.step += 1

		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		radar_front = np.hstack((input_data['radar_front'][1], np.ones((input_data['radar_front'][1].shape[0], 1))))
		radar_rear = np.hstack((input_data['radar_rear'][1], np.zeros((input_data['radar_rear'][1].shape[0], 1))))
		radar_all = np.concatenate((radar_front,radar_rear),axis=0)
		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]
		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0
		
		# smooth localization
		self.agent_vel = np.around(input_data['speed'][1]['speed'],2)
		self.agent_loc = utils.from_gps(input_data['gps'][1][0],input_data['gps'][1][1],input_data['gps'][1][2])

		self.imu_data.update(input_data['imu'], timestamp)
		self.yaw, self.agent_rot = utils.from_imu(self, self.imu_data)
		self.agent_pose.update_pose(self.agent_loc, self.agent_vel, self.imu_data, self.yaw, self.agent_rot, USE_EKF=False)
		agent_tf = self.agent_pose._pose_msg.transform
		
		# 大致给了框 通过现有的定位来生成map地图
		birdview = self.birdview_producer.produce(agent_tf, np.array([2.51,1.07]))
		rgb_birdview = cv2.cvtColor(BirdViewProducer.as_rgb(birdview), cv2.COLOR_BGR2RGB)

		result = {
				'rgb': rgb,
				'lidar': input_data['lidar'][1][:, :3],
				'gps': gps,
				'speed': speed,
				'compass': compass,
				'opendrive': rgb_birdview,
				'radar': radar_all,
				}
		
		pos = self._get_position(result)
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos)
		# next_wp_fake = np.zeros(next_wp.shape)
		# next_wp_fake[0] = -next_wp[1]
		# next_wp_fake[1] = next_wp[0]
		# next_wp = next_wp_fake
		result['next_command'] = next_cmd.value

		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		return result

	@torch.no_grad()
	def run_step(self, input_data, timestamp):

		# init control
		control = carla.VehicleControl()
		control.steer = 0.0
		control.throttle = 0.0
		control.brake = 0.0

		# init 初始化
		if not self.initialized:
			self._init()

			# get first data all
			tick_data = self.tick(input_data, timestamp)
			rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
			self.input_buffer['gps'].append(tick_data['gps'])
			self.input_buffer['thetas'].append(tick_data['compass'])
			self.input_buffer['opendrive'].append(tick_data['opendrive'])
			self.input_buffer['radar'].append(tick_data['radar'])
			self.prev_lidar = tick_data['lidar']

			return control
		
		tick_data = self.tick(input_data, timestamp)

		# 因为lidar hz无法设置 必须收两次才能get all lidar
		if self.step == 1:
			lidar = np.append(tick_data['lidar'], self.prev_lidar,axis=0)
			self.input_buffer['lidar'].append(lidar)
			self.prev_lidar = tick_data['lidar']

			return control

		# 初始化完成后进入 此时buffer里已经有了第一二帧的数据，len buffer = 1

		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
											torch.FloatTensor([tick_data['target_point'][1]])]

		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

		rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), crop=self.config.input_resolution)).unsqueeze(0)
		rgb_map = torch.from_numpy(np.transpose(Image.fromarray(tick_data['opendrive']), (2,0,1)))
		self.input_buffer['opendrive'].popleft()
		self.input_buffer['opendrive'].append(rgb_map.to('cuda', dtype=torch.float32))
		self.input_buffer['rgb'].popleft()
		self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))

		lidar = np.append(tick_data['lidar'], self.prev_lidar,axis=0)
		self.input_buffer['lidar'].popleft()
		self.input_buffer['lidar'].append(lidar)

		self.input_buffer['radar'].popleft()
		self.input_buffer['radar'].append(tick_data['radar'])

		self.input_buffer['gps'].popleft()
		self.input_buffer['gps'].append(tick_data['gps'])

		self.input_buffer['thetas'].popleft()
		self.input_buffer['thetas'].append(tick_data['compass'])

		# transform the lidar point clouds to local coordinate frame
		ego_theta = self.input_buffer['thetas'][-1]
		ego_x, ego_y = self.input_buffer['gps'][-1]


		for i, lidar_point_cloud in enumerate(self.input_buffer['lidar']):
			curr_theta = self.input_buffer['thetas'][i]
			curr_x, curr_y = self.input_buffer['gps'][i]
			lidar_point_cloud[:,1] *= -1 # inverts x, y
			lidar_transformed = transform_2d_points(lidar_point_cloud,
					np.pi/2-curr_theta, -curr_x, -curr_y, np.pi/2-ego_theta, -ego_x, -ego_y)
			lidar_transformed = torch.from_numpy(lidar_to_histogram_features(lidar_transformed, crop=self.config.input_resolution)).unsqueeze(0)
			
			self.lidar_processed = list()
			self.lidar_processed.append(lidar_transformed.to('cuda', dtype=torch.float32))


		self.pred_wp = self.net(self.input_buffer['rgb'], self.lidar_processed, \
								[map_list for map_list in self.input_buffer['opendrive']], \
								None, None, None, \
								target_point, gt_velocity)
		# print('step: ', self.step, 'pred_wp: ', self.pred_wp, 'target_waypoint',target_point, 'velocity', round(tick_data['speed'],2))
		steer, throttle, brake, metadata = self.net.control_pid(self.pred_wp, gt_velocity)
		self.pid_metadata = metadata

		if brake < 0.05: brake = 0.0
		if throttle > brake: brake = 0.0

		control = carla.VehicleControl()
		control.steer = float(steer)
		control.throttle = float(throttle)
		control.brake = float(brake)
		self.prev_lidar = tick_data['lidar']

		return control

	def destroy(self):
		del self.net
