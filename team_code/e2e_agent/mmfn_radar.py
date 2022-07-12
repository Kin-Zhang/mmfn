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
from mmfn_utils.models.model_rad import MMFN
from mmfn_utils.datasets.config import GlobalConfig
from mmfn_utils.datasets.dataloader import scale_and_crop_image, lidar_to_histogram_features, transform_2d_points, radar_to_size
import expert_agent.common.utils as utils
from planner import RoutePlanner
from expert_agent.common.utils import build_rmap, RoughMap

from planner import RoutePlanner
import math

def get_entry_point():
    return 'RadarVecAgent'


class RadarVecAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.MAP
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False


        self.input_buffer = {'rgb': deque(), 'lidar': deque(), 'gps': deque(), 'thetas': deque(), 'vectormap_lanes': deque(), 'radar': deque()}
        self.config = GlobalConfig()
        self.net = MMFN(self.config, 'cuda')
        print('loading model')
        
        # for DDP model load use
        # state_dict = torch.load(os.path.join(self.config_path.model_path, 'best_model.pth'))
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()

        # for k, v in state_dict.items():
        #     if 'module' not in k:
        #         k = 'module.'+k
        #     else:
        #         k = k.replace('features.module.', 'module.features.')
        #     new_state_dict[k]=v
        # self.net.load_state_dict(new_state_dict)
        self.net.load_state_dict(torch.load(os.path.join(self.config_path.model_path, 'best_model.pth')))
        print('load model success')
        
        self.net.cuda()
        self.net.eval()
        
        self.imu_data = utils.imu_msg()
        self.agent_pose = utils.LocalizationOperator()
        self.prev_lidar = None
        self.rough_map = RoughMap(self.config.up, self.config.down, self.config.left, self.config.right, self.config.lane_node_num, self.config.feature_num)
        self.rough_map_have_load = False

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
                    'reading_frequency': 30,  
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
        tmp_dir = os.path.join(self.config.tmp_town_for_save_opendrive,"opendrive")
        if not os.path.exists(tmp_dir): 
            os.makedirs(tmp_dir)
        opendrive_tmp_save_path = os.path.join(tmp_dir, "opstr.txt")
        print("Loading success for map, save path is:", opendrive_tmp_save_path)
        with open(opendrive_tmp_save_path, "w") as text_file:
            text_file.write(input_data['opendrive'][1]['opendrive'])
        lib_path = os.path.abspath('../../../assets/package')
        build_rmap([tmp_dir], lib_path)
        self.rough_map.read(os.path.join(tmp_dir,"a.rmap"))
        print("load rough_map which lane_num = ", len(self.rough_map.lanes))
        self.rough_map_have_load = True
            

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
        
        result = {
                'rgb': rgb,
                'lidar': input_data['lidar'][1],
                'gps': gps,
                'speed': speed,
                'compass': compass,
                'radar': radar_all,
                }
        
        pos = self._get_position(result)
        result['gps'] = pos
        pose2d = np.array([result['gps'][0], result['gps'][1], result['compass']]).astype(np.float)
        vectormap_lanes = self.rough_map.process(pose2d)
        if vectormap_lanes.shape[0] == 0:
            vectormap_lanes = np.zeros((1,10,5))
            print("warning, the vehicle is out of lane")
        
        result['vectormap_lanes'] = vectormap_lanes
        next_wp, next_cmd = self._route_planner.run_step(pos)
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

        if (self.rough_map_have_load == False) and (not ("opendrive" in input_data)):
            return control

        # init 初始化
        if not self.initialized:
            self._init()

            # get first data all
            tick_data = self.tick(input_data, timestamp)
            rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
            self.input_buffer['gps'].append(tick_data['gps'])
            self.input_buffer['thetas'].append(tick_data['compass'])
            self.input_buffer['vectormap_lanes'].append(tick_data['vectormap_lanes'])
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

        self.input_buffer['vectormap_lanes'].popleft()
        self.input_buffer['vectormap_lanes'].append(tick_data['vectormap_lanes'])

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
            lidar_transformed = transform_2d_points(lidar_point_cloud[...,:3],
                    np.pi/2-curr_theta, -curr_x, -curr_y, np.pi/2-ego_theta, -ego_x, -ego_y)
            lidar_transformed = torch.from_numpy(lidar_to_histogram_features(lidar_transformed, crop=self.config.input_resolution)).unsqueeze(0)
            
            self.lidar_processed = list()
            self.lidar_processed.append(lidar_transformed.to('cuda', dtype=torch.float32))

        vectormap_lanes =  self.input_buffer['vectormap_lanes'][-1]
        max_lane_num = lane_num = vectormap_lanes.shape[0]
        # print('vectormap:', vectormap_lanes.shap,e)

        vectormap_lanes = torch.tensor([[vectormap_lanes]]).to('cuda', dtype=torch.float32)
        lane_num =  torch.tensor([[lane_num]]).to('cuda', dtype=torch.int)
        max_lane_num = lane_num

        vectormaps = [vectormap_lanes, lane_num, max_lane_num]
        radar_list = radar_to_size(self.input_buffer['radar'][0], (81,5))
        data_test, radar_adj, radar = [], [], []
        for i in range(81):
            data_test.append(radar_list[:,1] - radar_list[i,1])
        radar_adj.append(torch.tensor(data_test).to('cuda', dtype=torch.float32))
        radar.append(torch.tensor(radar_list).to('cuda', dtype=torch.float32))
        self.pred_wp = self.net(self.input_buffer['rgb'], self.lidar_processed, None,\
                                        vectormaps,
                                        radar, 
                                        radar_adj, 
                                        target_point, gt_velocity)
        # (fronts, lidars, maps, vectormaps, radar, radar_adj, target_point, gt_velocity)
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
