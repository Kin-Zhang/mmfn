
"""
@Author: Kin
@Date: 2021/12/30
@TODO: 
    1. Add EKF on localization -> fix opendrive rgb shaking
    2. 
@Reference:
    1. https://github.com/erdos-project/pylot
"""


import carla
import math
from collections import deque
import numpy as np
from .pylot_utils import Quaternion, Rotation, Vector3D
import weakref

def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1) # how many seconds until collision

    return collides, p1 + x[0] * v1

def TTC_Time(pa,pb,va,vb , close_dis):
    maxt = 999
    rv = va - vb
    rp = pb - pa

    # 速度方向相反 -> vb速度va快 也算其中之一
    if rp.dot(rv) < 0.0:
        ttc = maxt
    else:
        a = np.linalg.norm(rv)
        # 速度基本一致 无需考虑
        if a <1e-4:
            return maxt
        rv_project2_rp = rp*rp.dot(rv)/rp.dot(rp)
        rp_project2_rv = rv*rv.dot(rp)/rv.dot(rv)
        dis_have_no_vel = np.linalg.norm(rp - rp_project2_rv)
        if dis_have_no_vel > close_dis:
            return maxt
        ttc = np.linalg.norm(rp)/np.linalg.norm(rv_project2_rp)
    return ttc

def TTC_Judge(pa, pb, va, vb):
    """
    reference: http://motion.cs.umn.edu/PowerLaw/
    """
    # TODO based on two vehicle length
    ra = 1.5
    rb = 1.5
    maxt = 999
    
    p = pb - pa #relative position
    rv = vb - va #relative velocity
    
    a = rv.dot(rv)
    b = 2*rv.dot(p)
    c = p.dot(p) - (ra + rb)**2
    
    det = b*b - 4*a*c
    t1 = maxt; t2 = maxt
    if (det > 0):
        t1 = (-b + math.sqrt(det))/(2*a)
        t2 = (-b - math.sqrt(det))/(2*a)
    t = min(t1,t2)
    
    if (t < 0 and max(t1,t2) > 0): #we are colliding
        t = 0 #maybe should be 0?
    if (t < 0): t = maxt
    if (t > maxt): t = maxt
    
    return t

def angle2heading(o1,o2):
    angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))
    angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)
    return angle_between_heading

def pos2wp(carla_map, pos, z=0.0):
    x = pos[0]
    y = pos[1]
    pos_loc = carla.Location(x=float(x),y=float(y),z=float(z))
    # get_waypoint(self, location, project_to_road=True, lane_type=carla.LaneType.Driving)
    wp = carla_map.get_waypoint(pos_loc)
    return wp
    
def carla2numpy(carla_vector, normalize=False):
    result = np.array([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result

def cal_distance(actor_1, actor_2):
    """
    actor_1, actor_2 (carla.Actor)
    """
    if abs(abs(actor_1.get_transform().location.z) - abs(actor_2.get_transform().location.z)) > 2.0 and abs(actor_1.get_transform().rotation.pitch)<5 and abs(actor_2.get_transform().rotation.pitch)<5:
       return 999
    else:
        return math.sqrt((actor_1.get_transform().location.x - actor_2.get_transform().location.x)**2 
                        + (actor_1.get_transform().location.y - actor_2.get_transform().location.y)**2)
    

def from_gps(latitude, longitude, altitude=0):
    """Creates Location from GPS (latitude, longitude, altitude).

    This is the inverse of the _location_to_gps method found in
    https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py
    """
    EARTH_RADIUS_EQUA = 6378137.0
    # The following reference values are applicable for towns 1 through 7,
    # and are taken from the corresponding OpenDrive map files.
    # LAT_REF = 49.0
    # LON_REF = 8.0
    # TODO: Do not hardcode. Get the references from the open drive file.
    LAT_REF = 0.0
    LON_REF = 0.0

    scale = math.cos(LAT_REF * math.pi / 180.0)
    basex = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * LON_REF
    basey = scale * EARTH_RADIUS_EQUA * math.log(
        math.tan((90.0 + LAT_REF) * math.pi / 360.0))

    x = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * longitude - basex
    y = scale * EARTH_RADIUS_EQUA * math.log(
        math.tan((90.0 + latitude) * math.pi / 360.0)) - basey

    # This wasn't in the original method, but seems to be necessary.
    y *= -1
    agent_loc = carla.Location(x=x,y=y,z=altitude)
    return agent_loc

def from_imu(self, imu_msg):
    if np.isnan(imu_msg.compass):
        yaw = self._last_yaw
    else:
        compass = np.degrees(imu_msg.compass)
        if compass < 270:
            yaw = compass - 90
        else:
            yaw = compass - 450
        self._last_yaw = yaw
    agent_rot = carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
    return yaw, agent_rot

class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        self.is_cross_line = False
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))
    
    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        print('Crossed line %s' % ' and '.join(text))
        self.is_cross_line = True if len(event.crossed_lane_markings)!=0 else False
class pose_msg(object):
    def __init__(self, transform=carla.Transform(),vel=None):
        self.transform = transform
        self.velocity_vector = vel
    
    def get_data(self, pos, vel, rot):
        self.transform = carla.Transform(carla.Location(pos[0],pos[1],pos[2]),carla.Rotation(pitch=rot.pitch,yaw=rot.yaw,roll=rot.roll))
        self.velocity_vector = vel
        return self

class imu_msg(object):
    def __init__(self):
        self.accelerometer = Vector3D()
        self.gyroscope = Vector3D()
        self.compass = 0.0
        self.timestamp = 0.0
    def update(self,imu_data,timestamp):
        self.accelerometer.x = imu_data[1][0]
        self.accelerometer.y = imu_data[1][1]
        self.accelerometer.z = imu_data[1][2]
        self.gyroscope.x = imu_data[1][3]
        self.gyroscope.y = imu_data[1][4]
        self.gyroscope.z = imu_data[1][5]
        self.compass = imu_data[1][6]
        self.timestamp = timestamp

class LocalizationOperator(object):
    def __init__(self):
        self._pose_msg = pose_msg()

        # Gravity vector.
        self._g = np.array([0, 0, -9.81])

        # Previous timestamp values.
        self._last_pose_estimate = None
        self._last_timestamp = None

        # NOTE: At the start of the simulation, the vehicle drops down from
        # the sky, during which the IMU values screw up the calculations.
        # This boolean flag takes care to start the prediction only when the
        # values have stabilized.
        self._is_started = False

        # Constants required for the Kalman filtering.
        var_imu_f, var_imu_w, var_gnss = 0.5, 0.5, 0.1
        self.__Q = np.identity(6)
        self.__Q[0:3, 0:3] = self.__Q[0:3, 0:3] * var_imu_f
        self.__Q[3:6, 3:6] = self.__Q[3:6, 3:6] * var_imu_w

        self.__F = np.identity(9)

        self.__L = np.zeros([9, 6])
        self.__L[3:9, :] = np.identity(6)

        self.__R_GNSS = np.identity(3) * var_gnss
        self.yaw = 0
        self._last_covariance = np.zeros((9, 9))
        self._last_yaw = 0
    def __skew_symmetric(self, v):
        """Skew symmetric form of a 3x1 vector."""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
                        dtype=np.float64)
    
    def estimate(self, imu_msg):
        # Retrieve the messages for this timestamp.
        if self._last_pose_estimate is None or (abs(imu_msg.accelerometer.y) > 100 and not self._is_started):
            self._last_pose_estimate = self._pose_msg
            self._last_timestamp = imu_msg.timestamp
        else:
            self._is_started = True
            
            # Initialize the delta_t
            current_ts = imu_msg.timestamp
            delta_t = (current_ts - self._last_timestamp)

            # Estimate the rotation.
            last_rotation_estimate = Quaternion.from_rotation(self._last_pose_estimate.transform.rotation)
            rotation_estimate = (last_rotation_estimate * Quaternion.from_angular_velocity(imu_msg.gyroscope, delta_t))

            # Transform the IMU accelerometer data from the body frame to the world frame, and retrieve location and velocity estimates.
            accelerometer_data = last_rotation_estimate.matrix.dot(imu_msg.accelerometer.as_numpy_array()) + self._g
            Vector3D_loc = Vector3D(self._last_pose_estimate.transform.location.x,self._last_pose_estimate.transform.location.y,self._last_pose_estimate.transform.location.z)
            last_location_estimate = Vector3D_loc.as_numpy_array()
            last_velocity_estimate = self._last_pose_estimate.velocity_vector.as_numpy_array()

            # Estimate the location.
            location_estimate = last_location_estimate + (delta_t * last_velocity_estimate) + (((delta_t**2) / 2.0) * accelerometer_data)

            # Estimate the velocity.
            velocity_estimate = last_velocity_estimate + (delta_t * accelerometer_data)
            # velocity_estimate = self._pose_msg.velocity_vector.as_numpy_array()
            # Fuse the GNSS values using an EKF to fix drifts and noise in the estimates.

            # Linearize the motion model and compute Jacobians.
            self.__F[0:3, 3:6] = np.identity(3) * delta_t
            self.__F[3:6, 6:9] = last_rotation_estimate.matrix.dot(-self.__skew_symmetric(accelerometer_data.reshape((3, 1)))) * delta_t

            # Fix estimates using GNSS
            Vector3D_loc = Vector3D(self._pose_msg.transform.location.x,self._pose_msg.transform.location.y,self._pose_msg.transform.location.z)
            gnss_reading = Vector3D_loc.as_numpy_array()
            (location_estimate,
             velocity_estimate,
             rotation_estimate,) = self.__update_using_gnss(location_estimate, 
                                                            velocity_estimate,
                                                            rotation_estimate,
                                                            gnss_reading,
                                                            delta_t)

            current_pose = self._pose_msg.get_data(pos = location_estimate,vel = velocity_estimate, rot = rotation_estimate.as_rotation())

            # Set the estimates for the next iteration.
            self._last_timestamp = current_ts
            self._last_pose_estimate = current_pose

    def __update_using_gnss(self, location_estimate, velocity_estimate,
                            rotation_estimate, gnss_reading, delta_t):
        # Construct H_k = [I, 0, 0] (shape=(3, 9))
        H_k = np.zeros((3, 9))
        H_k[:, :3] = np.identity(3)

        # Propogate uncertainty.
        Q = self.__Q * delta_t * delta_t
        self._last_covariance = (self.__F.dot(self._last_covariance).dot(
            self.__F.T)) + (self.__L.dot(Q).dot(self.__L.T))

        # Compute Kalman gain. (shape=(9, 3))
        K_k = self._last_covariance.dot(
            H_k.T.dot(
                np.linalg.inv(
                    H_k.dot(self._last_covariance.dot(H_k.T)) +
                    self.__R_GNSS)))

        # Compute error state. (9x3) x ((3x1) - (3x1)) = shape(9, 1)
        delta_x_k = K_k.dot(gnss_reading - location_estimate)

        # Correct predicted state.
        corrected_location_estimate = location_estimate + delta_x_k[0:3]
        corrected_velocity_estimate = velocity_estimate + delta_x_k[3:6]
        roll, pitch, yaw = delta_x_k[6:]
        corrected_rotation_estimate = Quaternion.from_rotation(Rotation(roll=roll, pitch=pitch, yaw=yaw)) * rotation_estimate

        # Fix the covariance.
        self._last_covariance = (np.identity(9) - K_k.dot(H_k)).dot(self._last_covariance)

        return (
            corrected_location_estimate,
            corrected_velocity_estimate,
            corrected_rotation_estimate,
        )

    def update_pose(self, agent_loc, agent_vel, imu_msg, yaw, agent_rot, USE_EKF=False):
        self._pose_msg.transform = carla.Transform(agent_loc, agent_rot)
        self._pose_msg.velocity_vector = Vector3D(agent_vel * np.cos(yaw), agent_vel * np.sin(yaw), 0)
        if USE_EKF:
            self.estimate(imu_msg)
        return self._pose_msg.transform

import os
from tqdm import tqdm
def build_rmap(all_path: list, lib_path):
    print("start to build rmap. map_number:", len(all_path))
    opendrive2vec = os.path.join(lib_path,"rough_map_node")
    is_error = False
    for path in tqdm(all_path):
        cmd2 = opendrive2vec + " " + path
        print(cmd2)
        tmp = os.popen(cmd2).readlines()
        # print(tmp[0])
        if (tmp[0][-2:]!="ok"):
            is_error = True
            break
    if (is_error):
        print("Error in build rmap")
    else:
        print("build rmap successfully")
    return is_error


from cv2 import pointPolygonTest
import numpy as np

from shapely.geometry import Polygon,Point, MultiPoint
import shapely
import math
class RoughLane:
    lane_info: Polygon
    lane_nodes: MultiPoint

class RoughMap:
    #[lane_num]
    lanes: list

    def __init__(self, up:float, down:float, left:float, right:float, lane_node_num:int, feature_num:int):
        self.polygon = Polygon(
            [[up, -left],
            [-down, -left],
            [-down, right],
            [up, right]]
        )
        self.feature_num = feature_num
        self.lane_node_num = lane_node_num

    def read(self, file_path) -> None:
       f = open(file_path)
       f.readline()
       lane_num = int(f.readline().strip().split(" ")[1])
       self.lanes= []
       tmp = []
       for i in range(lane_num):
           rough_lane = RoughLane()
           node_num = int(f.readline().strip().split(" ")[1])
           lane_info = np.array(f.readline().strip().split()).astype(np.float).reshape(4,2)
           rough_lane.lane_info = Polygon(lane_info)
           lane_nodes = []
           for j in range(node_num):
               node_data = np.array(f.readline().strip().split(" ")).astype(np.float)
               lane_nodes.append(node_data)
               tmp.append([node_data[0],node_data[1]])

           rough_lane.lane_nodes = np.array(lane_nodes)
           
           self.lanes.append(rough_lane)
       tmp = np.array(tmp)

    def process(self, pose2d: np.ndarray):
        """
        Params:
            pose2d:[3] -> [x,y,theta]
        Return:
            res: [num, lane_node_num, feature_num]
        """
        
        x,y,theta = pose2d

        polygon_1 = shapely.affinity.rotate(self.polygon, theta*180/math.pi, origin=(0,0))
        polygon_2 = shapely.affinity.translate(polygon_1, x, y)
        
        res = []
        
        for lane in self.lanes:
            if polygon_2.disjoint(lane.lane_info):
                continue
            points = []
            for lane_node in lane.lane_nodes:
                p = Point(lane_node[0], lane_node[1])
                p = shapely.affinity.translate(p, -x, -y)
                p = shapely.affinity.rotate(p, -theta*180/math.pi, origin=(0,0))
                point = [p.x, p.y] + list(lane_node[2:])
                points.append(point)
            if len(points) < self.lane_node_num:
                points = points  + [[0.0]*self.feature_num]*(self.lane_node_num-len(points))

            res.append(points)
        res = np.array(res)
        return res

        

