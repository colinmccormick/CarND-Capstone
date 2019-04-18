#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
#from scipy.spatial.transform import Rotation as R
import math
import numpy as np

STATE_COUNT_THRESHOLD = 3
TESTING_WITHOUT_IMG = False # Set to False to remove the dependency on Simulator light states

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoint_tree = None
        self.waypoints_2d = None
        self.camera_image = None
        self.lights = []
        self.closest_light = None
        self.camera_info = None
             
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
        sub7 = rospy.Subscriber('/camera_info', CameraInfo, self.camera_cb)
        
        # get camera info
        #calib_yaml = rospy.get_param("/grasshopper_calibration_yaml")
        #self.camera_info = yaml_to_CameraInfo(calib_yaml)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.image_counter = 0

        rospy.spin()
        
    def camera_cb(self, msg):
        print("Got camera info")
        self.camera_info = msg
        
    def pose_cb(self, msg):
        self.pose = msg

        if TESTING_WITHOUT_IMG:
            #Remove -- Temporary code - sub for image processing
            light_wp, state = self.process_traffic_lights()
            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

        # Remove till here

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights
        #rospy.loginfo(self.lights)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """

        # Only check every 3rd image (reduce latency)
        self.image_counter += 1
        if self.image_counter % 3 != 0:           
            return
        # End speed-up

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()
           
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        return closest_idx

    def get_sim_light_state(self, light_idx):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        # Dummy code to be removed later
        #rospy.loginfo('Light state: %s', self.lights[light_idx].state)
        if(self.lights):
            #rospy.loginfo(self.lights[light_idx].state)
            return self.lights[light_idx].state
        else:
            rospy.loginfo('Lights uninit')
        return TrafficLight.GREEN

    def get_tl_coords_in_image(self, coords_in_world):
        """Get transform from (X,Y,Z) world coords to (x,y) camera coords. 
        See https://github.com/udacity/CarND-Capstone/issues/24
        Args:
            coords_in_world : TrafficLight coordinates
        """
        
        self.listener.waitForTransform("/world", "/base_link", rospy.Time(), rospy.Duration(1.0))
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/world", "/base_link", now, rospy.Duration(1.0))
            (trans,rot) = self.listener.lookupTransform("/world", "/base_link", now)
            print("Got map transform")
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.log_err("Couldn't find camera to map transform.")
            print("Can't get map transform")
                
        # do 3D rotation and translation of light coords from world to car frame
        x_world = coords_in_world.x
        y_world = coords_in_world.y
        z_world = coords_in_world.z
        e = tf.transformations.euler_from_quaternion(rot)
        cos_yaw = math.cos(e[2])
        sin_yaw = math.sin(e[2])
        x_car = x_world * cos_yaw - y_world * sin_yaw + trans[0]
        y_car = x_world * sin_yaw + y_world * cos_yaw + trans[1]
        z_car = z_world + trans[2]
     
        # use camera projection matrix to translate world coords to camera pixel coords
        # http://docs.ros.org/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        uvw = np.dot(self.camera_info.P,[x_car,y_car,z_car,1])
        camera_x = uvw[0]/uvw[2]
        camera_y = uvw[1]/uvw[2]
        
        #focal_length = 2300
        #half_image_width = 400
        #half_image_height = 300
        #x_offset = -30
        #y_offset = 340
        #half_image_width = 400
        #half_image_height = 300
        
        return (camera_x,camera_y)
    
    def get_light_state(self, tl_idx):
        """Determines the current color of the traffic light
        Args:
            tl_idx (int): index of light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # find light in this image using coord transform
        (x,y) = self.get_tl_coords_in_image(self.lights[tl_idx].pose.pose.position)
        print("Camera pixel coords: {0},{1}".format(x,y))
        # TODO: crop image around this pixel coord to get traffic light image
        # cv_image = (cropping...)
        
        return self.light_classifier.get_classification(cv_image)
                
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its color
        and the best waypoint for stopping.
        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        line = None
        max_visible_dist = 100 # units
        line_wp_idx = -1
        tl_idx = -1

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

        #TODO find the closest visible traffic light (if one exists)
        diff = len(self.waypoints.waypoints)
        for i, temp_closest_stop_position in enumerate(stop_line_positions):
            line = stop_line_positions[i]
            wp_idx = self.get_closest_waypoint(line[0], line[1])
            d = wp_idx - car_position
            if d >= 0 and d < diff and d < max_visible_dist:
                diff = d
                line_wp_idx = wp_idx
                tl_idx = i
                line = temp_closest_stop_position

        # if light and self.closest_light != light:
        #     rospy.loginfo('TL coming up: %d', diff)
        #     rospy.loginfo('Light details(%d): %s', tl_idx, light)
        #     self.closest_light = light
        #     state = self.get_sim_light_state(tl_idx)
        #     return line_wp_idx, state
        # elif not light and self.closest_light:
        #     rospy.loginfo('Free.. speed up')
        #     self.closest_light = None
        #     return -1, TrafficLight.UNKNOWN

        if tl_idx > 0:
            if TESTING_WITHOUT_IMG:
                state = self.get_sim_light_state(tl_idx)
                return line_wp_idx, state
                
            else:
                # Fix this for image processing.
                print("Calling get_light_state")
                state = self.get_light_state(tl_idx)
                print("Light state: {0}".format(state))
                return -1, TrafficLight.UNKNOWN

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
