import random
import numpy as np
import carla
import math
import cv2
from PIL import Image

from common.map_agent import MapAgent
from common.planner_controller import PIDController
from common.utils import *

def get_entry_point():
    return 'MMFNPilot'

def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])

class MMFNPilot(MapAgent):

    # for stop signs
    PROXIMITY_THRESHOLD = 30.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        

    def _init(self):
        self._ego_transform = None

        # 根据道路宽度和交叉路口的半径来改变
        self.too_close_dis = 999
        self.stop_cross_line_at_red = False
        self._over_time = False
        self._stop_for_change_lane = 0
        
        # TODO through config
        self.near_by_dis = 30
        # 最好留出余地
        self._distance_between_change_lane = 8
        self.close_obs_speed_threshold = 1
        self.max_throttle = 0.8
        self.speed_delta = 0.8
        self.consider_angle = 120
        self.STOP_THRESHOLD = 8
        self.red_angle_diff = 20
        
        # 沿道路宽度多少以内为直线方向，同时这个参数与 前方路径点 是否有障碍物相关
        self.precentage_of_lane_staright = 0.7
        self._near_object = {"vehicle":[], "traffic_light": [], "walker": [], "stop": [], "car_infront": [], "behind": []}

        super()._init()

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        # for stop signs
        self._target_stop_sign = None # the stop sign affecting the ego vehicle
        self._stop_completed = False # if the ego vehicle has completed the stop sign
        self._affected_by_stop = False # if the ego vehicle is influenced by a stop sign
        
        self._prev_lane_id = self._map.get_waypoint(self._vehicle.get_transform().location).lane_id
        self._prev_road_id = self._map.get_waypoint(self._vehicle.get_transform().location).road_id

    def run_step(self, input_data, timestamp):
        
        control = carla.VehicleControl(0.0,0.0,1.0)

        # 0. 初始化相关
        if not self.initialized and not ("opendrive" in input_data):
            return control

        if not self.initialized:
            self.save_map(input_data)
            self._init()
        
        # 1. 每次进来tick到自己的数据 并 更新所需内容
        data = self.tick(input_data, timestamp)
        self.update_info()
        pos = carla2numpy(self._ego_transform.location)

        # 2. 提取目标点
        near_node, near_command = self._waypoint_planner.run_step(pos)
        far_node, far_command = self._command_planner.run_step(pos)
        
        # 3. 控制 -> 内部有 rules-based
        steer, throttle, brake, target_speed, reverse = self._get_control(pos, near_node, far_node)

        control.steer = steer + 1e-2 * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)
        control.reverse = reverse
        
        if not self.debug_print_mmfn:
            # 4. 保存 -> 可注释
            if self.step % 10 == 0 and self.save_path is not None and self.step!=0:
                self.save(near_node, far_node, near_command, steer, throttle, brake, target_speed, data, reverse)
            self.prev_lidar = data['lidar']
        
        return control
    
    def _get_control(self, pos, target, far_target):
        # get truth
        theta = self._vehicle.get_transform().rotation.yaw
        speed = np.linalg.norm(carla2numpy(self._ego_velocity))

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        
        # 判断减速和刹车 -> 刹车中有基于规则的
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        vehicle, walker, change_lane, light, stop_sign = self._should_brake(target, far_target)
        brake = any(x is not None for x in [vehicle, walker, change_lane, light, stop_sign])#
        # 实施减速和刹车
        target_speed = 4.0 if should_slow else 7.0
        target_speed = target_speed if not brake else 0.0

        # change lane 停车计时
        if change_lane is not None and light is None and stop_sign is None :
            s2 = np.linalg.norm(carla2numpy(change_lane.get_velocity()))
            if s2 < self.close_obs_speed_threshold:
                self._stop_for_change_lane += 1
        elif not self._over_time:
            self._stop_for_change_lane = 0
        
        if self._stop_for_change_lane > 20:
            self._over_time = True
            brake = any(x is not None for x in [vehicle, walker, light, stop_sign])
            if change_lane is None:
                self._stop_for_change_lane = 0
                self._over_time = False

        # 0. 红绿灯处摆正位置 后面无车后退，后面有车则前进
        if self.stop_cross_line_at_red is not None and walker is None:
            target_speed = 0.4
            reverse = (len(self._near_object["behind"])==0)
            angle_turn = (-1 if reverse else 1) * self.stop_cross_line_at_red/90
            steer = self._turn_controller.step(angle_turn)
            brake = False
            if self.debug_print_mmfn:
                print(self.step, "correct our yaw heading the diff deg:", self.stop_cross_line_at_red, reverse)
        else:
            reverse = False
        
        # 1. 跟车策略
        now_wyp = self._map.get_waypoint(self._ego_transform.location)
        next_lane_id = pos2wp(self._map, self._waypoint_planner.route[0][0]).lane_id
        near_lane_id = pos2wp(self._map, target).lane_id
        now_lane_id  = now_wyp.lane_id

        # 不在换道状态, 不在junction
        if near_lane_id == (now_lane_id and next_lane_id)  and not now_wyp.is_junction \
            and len(self._near_object["car_infront"]) == 1 and abs(self._ego_transform.rotation.pitch)<5 \
            and not any(x is not None for x in [walker, change_lane, light, stop_sign]):
            target_vehicle = self._near_object["car_infront"][0]
            obs_wyp = self._map.get_waypoint(target_vehicle.get_location())
            if not obs_wyp.is_junction:
                obs2ego = cal_distance(self._vehicle, target_vehicle)
                obs_vel = float(np.linalg.norm(carla2numpy(target_vehicle.get_velocity())))
                if obs2ego > max(self.STOP_THRESHOLD, 1.5*np.linalg.norm(carla2numpy(self._vehicle.get_velocity()))) and \
                    obs_vel > self.close_obs_speed_threshold*0.5:
                    target_speed = obs_vel
                    brake = False
                    if self.debug_print_mmfn:
                        print(self.step, "follow car", target_vehicle.type_id, target_vehicle.id)
        
        # PID
        delta = np.clip(target_speed - speed, 0.0, self.speed_delta)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        
        # record use
        self.should_slow = int(should_slow)
        self.should_brake = int(brake)
        self.angle = angle
        self.angle_unnorm = angle_unnorm
        self.angle_far_unnorm = angle_far_unnorm

        if brake:
            steer *= 0.5
            throttle = 0.0
        
        return steer, throttle, brake, target_speed, reverse

    def empty_info(self):
        self._near_object = {"vehicle":[], "traffic_light": [], "walker": [], "stop": [], "car_infront": [], "behind": []}
    
    def update_info(self):
        self.empty_info()

        actors = self._world.get_actors()
        self._ego_transform = self._vehicle.get_transform()
        self._ego_velocity = self._vehicle.get_velocity() #km/h
        hero_vehicle = self._vehicle
        hero_waypoint = self._map.get_waypoint(self._ego_transform.location)

        self.too_close_dis = hero_waypoint.lane_width * 0.9
        # ================= UPDATE NEAR THINGS =========================== #
        theta = self._vehicle.get_transform().rotation.yaw
        pos = carla2numpy(self._ego_transform.location)
        # update vehicle
        for v in actors.filter('*vehicle*'):
            if v.id == hero_vehicle.id:
                continue
            obs_wyp = self._map.get_waypoint(v.get_location())
            p2 = carla2numpy(v.get_location())
            p3 = carla2numpy(obs_wyp.transform.location)
            obs2wy_dis = np.linalg.norm(p3 - p2)
            distance = cal_distance(v, hero_vehicle)

            # 1.25m 大概是车宽
            if distance < self.near_by_dis and obs2wy_dis <= 1.25:
                angle_unnorm = self._get_angle_to(pos, theta, carla2numpy(v.get_transform().location))
                if abs(angle_unnorm) < self.consider_angle:
                    self._near_object["vehicle"].append(v)
                elif distance < self.near_by_dis/2:
                    self._near_object["behind"].append(v)
        
        # update traffic  and stop through tick
        self._near_object["traffic_light"] = self._traffic_lights
        self._near_object["stop"] = self._stop_signs

        # update walker
        for v in actors.filter('*walker*'):
            obs_wyp = self._map.get_waypoint(v.get_location())
            p2 = carla2numpy(v.get_location())
            distance = cal_distance(v, hero_vehicle)
            if distance < self.near_by_dis:
                angle_unnorm = self._get_angle_to(pos, theta, carla2numpy(v.get_transform().location))
                # remove obs behind us
                if abs(angle_unnorm) < self.consider_angle:
                    self._near_object["walker"].append(v)
        
        # update the nearest car in front of ego in same road
        near_dis = np.inf
        track_vehicle = None
        now_wyp = self._map.get_waypoint(self._ego_transform.location)
        p1 = carla2numpy(self._ego_transform.location)
        for target_vehicle in self._near_object["vehicle"]:
            obs_wyp = self._map.get_waypoint(target_vehicle.get_location())
            p2 = carla2numpy(target_vehicle.get_location())
            if now_wyp.road_id == obs_wyp.road_id and now_wyp.lane_id == obs_wyp.lane_id:
                distance = np.linalg.norm(p2-p1)
                if distance < near_dis:
                    near_dis = distance
                    track_vehicle = target_vehicle
        if track_vehicle is not None:
            self._near_object["car_infront"].append(track_vehicle)

    def _should_brake(self, target, far_target):

        vehicle = self._is_vehicle_hazard(self._near_object["vehicle"], target)
        walker = self._is_walker_hazard(self._near_object["walker"])
        light = self._is_light_red(self._near_object["traffic_light"])
        stop_sign = self._is_stop_sign_hazard(self._near_object["stop"])
        change_lane = self._is_change_lane(self._near_object["vehicle"]+self._near_object["behind"], target, far_target)

        # 单独判断 停车时 是否~~跨线~~ 换成是否水平于道路
        target_next = carla2numpy(self._map.get_waypoint(self._ego_transform.location).next(1)[0].transform.location)
        angle_unnorm = self._get_angle_to(carla2numpy(self._ego_transform.location), self._vehicle.get_transform().rotation.yaw, target_next)
        if light is not None and abs(angle_unnorm) > self.red_angle_diff:
            self.stop_cross_line_at_red = angle_unnorm
        else:
            self.stop_cross_line_at_red = None

        return vehicle, walker, change_lane, light, stop_sign

    def _point_inside_boundingbox(self, point, bb_center, bb_extent):
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    def _get_forward_speed(self, transform=None, velocity=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed

    def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._world.get_map().get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self._point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                affected = True

        return affected

    def _is_stop_sign_hazard(self, stop_sign_list):
        if self._affected_by_stop:
            if not self._stop_completed:
                current_speed = self._get_forward_speed()
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True
                    return None
                else:
                    return self._target_stop_sign
            else:
                # reset if the ego vehicle is outside the influence of the current stop sign
                if not self._is_actor_affected_by_stop(self._vehicle, self._target_stop_sign):
                    self._affected_by_stop = False
                    self._stop_completed = False
                    self._target_stop_sign = None
                return None

        ve_tra = self._vehicle.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._world.get_map().get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in stop_sign_list:
                if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
                    # this stop sign is affecting the vehicle
                    self._affected_by_stop = True
                    self._target_stop_sign = stop_sign
                    return self._target_stop_sign

        return None

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return affecting

        return None

    def _is_walker_hazard(self, walkers_list):
        # TODO need test
        o1 = _orientation(self._ego_transform.rotation.yaw)
        p1 = carla2numpy(self._vehicle.get_location())
        v1 = carla2numpy(self._ego_velocity)
        stop_dis = np.clip(max(self.STOP_THRESHOLD, 2*np.linalg.norm(v1)), 0, self.STOP_THRESHOLD*1.5)/2 # increases the threshold distance
        
        for walker in walkers_list:
            o2 = _orientation(walker.get_transform().rotation.yaw)
            v2 = carla2numpy(walker.get_velocity())
            p2 = carla2numpy(walker.get_location())
            angle_between_heading = angle2heading(o1,o2)
            dis = np.linalg.norm(p2-p1)

            obs_wyp = self._map.get_waypoint(walker.get_location())
            p2 = carla2numpy(walker.get_location())
            p3 = carla2numpy(obs_wyp.transform.location)
            obs2wy_dis = np.linalg.norm(p3 - p2)
            
            if dis < stop_dis and np.linalg.norm(v2) > 0.1 and obs2wy_dis < 1:
                if self.debug_print_mmfn:
                    print(self.step, "walker angle", angle_between_heading, "dis: ", dis)
                return walker
            # else judge ttc
            ttc_small = TTC_Time(p1, p2, v1, v2, self._map.get_waypoint(self._ego_transform.location).lane_width / 2)
            ttc_large = TTC_Time(p1, p2, v1, v2, self._map.get_waypoint(self._ego_transform.location).lane_width)
            if ttc_small < 8:
                if self.debug_print_mmfn:
                    print(self.step, "small walker angle", angle_between_heading, "dis: ", dis)
                return walker
            if ttc_large <8 and angle_between_heading > 60 and np.linalg.norm(v2) > 0.5  and obs2wy_dis < 2:
                if self.debug_print_mmfn:
                    print(self.step, "large walker angle", angle_between_heading, "dis: ", dis)
                return walker
        return None

    def _is_vehicle_hazard(self, vehicle_list, target):

        o1 = _orientation(self._ego_transform.rotation.yaw)
        p1 = carla2numpy(self._ego_transform.location)
        v1 = np.linalg.norm(carla2numpy(self._vehicle.get_velocity()))
        
        # 作为前向刹停的距离值
        s1 = np.clip(max(self.STOP_THRESHOLD, 2*v1), 0, self.STOP_THRESHOLD*1.5) # increases the threshold distance
        now_wyp = self._map.get_waypoint(self._ego_transform.location)
        
        for target_vehicle in vehicle_list:
            obs_wyp = self._map.get_waypoint(target_vehicle.get_location())
            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = carla2numpy(target_vehicle.get_location())
            s2 = np.linalg.norm(carla2numpy(target_vehicle.get_velocity()))
        
            # 坐标间向量
            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(o1.dot(p2_p1_hat)))
            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = angle2heading(o1,o2)
            ttc = TTC_Time(p1, p2, carla2numpy(self._ego_velocity), carla2numpy(target_vehicle.get_velocity()), self.too_close_dis)
            
            # -1. 如果就在即将要去的路径点上
            is_in_my_way = self._if_in_my_way(target, obs_wyp, now_wyp, near2dis = s1/2 - 2)
            if is_in_my_way:
                if self.debug_print_mmfn:
                    print(self.step, "in my way", target_vehicle.type_id, target_vehicle.id, " dis: ", distance)
                return target_vehicle
            
            # 0. 如果不在junction内 不同road_id直接不考虑 同一road_id 不同lane_id不考虑
            if now_wyp.is_junction is False and obs_wyp.is_junction is False:
                # 速度高了还是要考虑的 -> 针对隔壁车道车突然插进来 -> 同时双方heading角度也要考虑
                if s2<7 and s2>1 and v1<7 and angle_between_heading > self.red_angle_diff * 0.9:
                    if now_wyp.road_id != obs_wyp.road_id:
                        continue
                    elif now_wyp.lane_id != obs_wyp.lane_id:
                        continue
            
            # 1. ttc 判断，对方速度接近于0时 不考虑
            if ttc < self.STOP_THRESHOLD and s2 > self.close_obs_speed_threshold:
                if self.debug_print_mmfn:
                    print(self.step, "judge ttc", ttc, target_vehicle.type_id, target_vehicle.id, " dis: ", distance)
                return target_vehicle

            # 2. 跟车时刹车，距离小于刹车阈值 且车辆不在两侧范围， angle to car 适应型
            elif distance < s1 and angle_to_car < np.rad2deg(np.arcsin(now_wyp.lane_width *0.7/s1)) and angle_between_heading < self.consider_angle:
                if self.debug_print_mmfn:
                    print(self.step, "stop id", target_vehicle.type_id, target_vehicle.id, " dis", distance, "vel",v1, "ang",angle_to_car)
                return target_vehicle

            # 3. junction处取消angle_to_car的限制
            elif now_wyp.is_junction and obs_wyp.is_junction and s2 > self.close_obs_speed_threshold * 0.9:
                if distance < s1 and ((angle_to_car < 80 and angle_between_heading < 180 * 0.9) or (angle_to_car < np.rad2deg(np.arcsin(now_wyp.lane_width*self.precentage_of_lane_staright/s1)))):
                    junction_box = now_wyp.get_junction().bounding_box
                    junction_rad = np.linalg.norm(carla2numpy(junction_box.extent))
                    ego_dis_to_junc = np.linalg.norm(carla2numpy(junction_box.location) - p1)
                    obs_dis_to_junc = np.linalg.norm(carla2numpy(junction_box.location) - p2)
                    if ego_dis_to_junc < junction_rad/2 and obs_dis_to_junc< junction_rad/2:
                        if self.debug_print_mmfn:
                            print(self.step, "junction stop id", target_vehicle.type_id, target_vehicle.id, " dis", distance, "vel",v1, "ang",angle_between_heading)
                        return target_vehicle

        return None
        
    def _is_change_lane(self, vehicle_list, target, far_target):
        # 5m 前的远处点
        farn_lane_id = pos2wp(self._map, far_target, z=self._ego_transform.location.z).lane_id
        farn_road_id = pos2wp(self._map, far_target, z=self._ego_transform.location.z).road_id
        
        # 2m 前的临近点
        near_wyp = pos2wp(self._map, target, z=self._ego_transform.location.z)
        # 2m 前点往回走
        near_pre_wyp = near_wyp.previous(2)[0]
        near_lane_id = near_wyp.lane_id
        near_road_id = near_wyp.road_id

        # 下一个目标点lane_id
        next_wyp = pos2wp(self._map, self._waypoint_planner.route[0][0], z=self._ego_transform.location.z)
        next_lane_id = next_wyp.lane_id
        next_road_id = next_wyp.road_id

        # 自身所处的lane_id
        now_wyp = self._map.get_waypoint(self._ego_transform.location)
        now_lane_id = now_wyp.lane_id
        now_road_id = now_wyp.road_id

        # 确认一下 2m目标点往回的lane id和road 和现在不一致 不然同属一个
        if (now_lane_id == near_pre_wyp.lane_id and now_road_id == near_pre_wyp.road_id) \
            or (now_lane_id == next_lane_id and now_lane_id == near_lane_id and now_lane_id == farn_lane_id):
            return None
        
        max_id = max(near_lane_id, next_lane_id, now_lane_id, farn_lane_id, near_pre_wyp.lane_id)
        min_id = min(near_lane_id, next_lane_id, now_lane_id, farn_lane_id, near_pre_wyp.lane_id)

        # 自身朝向与位置
        p1 = carla2numpy(self._vehicle.get_location())
        o1 = _orientation(self._ego_transform.rotation.yaw)
        v1 = np.linalg.norm(carla2numpy(self._vehicle.get_velocity()))

        for target_vehicle in vehicle_list:

            obs_wyp = self._map.get_waypoint(target_vehicle.get_location())
            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = carla2numpy(target_vehicle.get_location())
            v2 = np.linalg.norm(carla2numpy(target_vehicle.get_velocity()))

            # 在目标车道的车 (目标lane_id and 目标road_id)
            if (min_id <= obs_wyp.lane_id <= max_id and obs_wyp.lane_id != now_lane_id) and \
               (obs_wyp.road_id == (now_road_id or near_road_id or next_road_id or near_pre_wyp.road_id or farn_road_id)):
                # 远点的话 需要双重确认road_id也为远点的road id
                if (obs_wyp.lane_id == farn_lane_id and obs_wyp.road_id != farn_road_id):
                    continue
                # 在换道车道内的判断距离
                angle_between_heading = angle2heading(o1,o2)
                p2 = carla2numpy(target_vehicle.get_location())
                distance = np.linalg.norm(p2-p1)
                threshold_dis = np.clip(max(self._distance_between_change_lane, v1*2, v2*2), 0, self.STOP_THRESHOLD*1.5)
                if distance < threshold_dis and angle_between_heading < self.consider_angle * 0.65:
                    distance = np.linalg.norm(p2 - p1)
                    angle_to_car = np.degrees(np.arccos(o1.dot((p2 - p1)/ (distance + 1e-4))))
                    angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
                    # 基本无速度 又不是红绿灯处停车 is_junction状态
                    if v2 < 0.1 and not near_wyp.is_junction:
                        continue
                    
                    if self.debug_print_mmfn:
                        print(self.step, "change ", target_vehicle.type_id, target_vehicle.id, "dis:", round(distance), "lane id: obs", obs_wyp.lane_id, "next", next_lane_id, "now", now_lane_id, "near prev", near_pre_wyp.lane_id)
                        print("road_id now", now_road_id,"next", next_road_id, "obs", obs_wyp.road_id, "near", near_road_id, "near prev", near_pre_wyp.road_id)
                    return target_vehicle

        return None

    def _get_angle_to(self, pos, yaw, target):
        aim = target - pos
        yaw_rad = np.deg2rad(yaw)
        # 旋转矩阵
        R = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad),  np.cos(yaw_rad)],
            ])
        # 乘上yaw的旋转矩阵 -> 就直接在yaw角度上的坐标系了
        diff_v = R.T.dot(aim)
        # 直接就是diff_v
        angle = -np.degrees(np.arctan2(-diff_v[1], diff_v[0]))
        return angle
    
    def _if_in_my_way(self, target, obs_wyp, hero_waypoint, near2dis=2):
        near_wp = pos2wp(self._map, target)
        for i in np.linspace(0.1,near2dis,10):
            near_next_wp = near_wp.next(i)
            if len(near_next_wp)>0:
                near_next_wp = near_next_wp[0]
                obs2near_dis = np.linalg.norm(carla2numpy(obs_wyp.transform.location) - carla2numpy(near_next_wp.transform.location))
                if obs2near_dis < (hero_waypoint.lane_width*self.precentage_of_lane_staright/2):
                    return True
        return False