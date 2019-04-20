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
import os

import tensorflow

#import matplotlib.pyplot as plt


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
        self.TLDetector_pub = rospy.Publisher('/traffic_light_detector', Image, queue_size=1)
        self.most_confident_detection = rospy.Publisher('/traffic_light_most_confident', Image, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.image_counter = 0

        self.detection_graph = None
        self.category_index = None
        self.tensor_dict = {}
        self.image_tensor = None
        self.sess = None
        
        self.loadModel()
        
        self.detections = None
        self.hasDetections = False

        rate = rospy.Rate(5)  # Drop rate to 5 Hz to handle latency
        
        rospy.spin()
    

    #Leading pretrained model
    def loadModel(self):
        rospy.loginfo("Tensorflow version: %s", tensorflow.__version__)
        MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
        PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
        PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

        self.detection_graph = tensorflow.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tensorflow.GraphDef()
          with tensorflow.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tensorflow.import_graph_def(od_graph_def, name='')

        #category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        self.category_index = {1:"traffic lights"}
        rospy.loginfo("Detection graph loaded")
        
        rospy.loginfo("Initializing Sesson")
        with self.detection_graph.as_default():
            self.sess =tensorflow.Session()
            # Get handles to input and output tensors
            ops = tensorflow.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                  self.tensor_dict[key] = tensorflow.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in self.tensor_dict:
                # The following processing is only for single image
                detection_boxes = tensorflow.squeeze(self.tensor_dict['detection_boxes'], [0])
                detection_masks = tensorflow.squeeze(self.tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tensorflow.cast(self.tensor_dict['num_detections'][0], tensorflow.int32)
                detection_boxes = tensorflow.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tensorflow.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tensorflow.cast(tf.greater(detection_masks_reframed, 0.5), tensorflow.uint8)
                # Follow the convention by adding back the batch dimension
                self.tensor_dict['detection_masks'] = tensorflow.expand_dims(detection_masks_reframed, 0)
            self.image_tensor = tensorflow.get_default_graph().get_tensor_by_name('image_tensor:0')        
        rospy.loginfo("Sesson Initialized")

    def getBBox(self):
      #rospy.loginfo("Getting bboxes")
      cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, desired_encoding="passthrough")
        
      image_np = np.asarray(cv_image)
      image = np.expand_dims(image_np, axis=0)

      # Run inference
      output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:output_dict['detection_masks'] = output_dict['detection_masks'][0]

      detections = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

      # Visualize detected bounding boxes.
      num_detections = int(output_dict['num_detections'])
      rows = detections.shape[0]
      cols = detections.shape[1]
      for i in range(num_detections):
          bbox = [float(v) for v in output_dict['detection_boxes'][i]]
          score = output_dict['detection_scores'][i]
          if score > 0.3:
              x = bbox[1] * cols
              y = bbox[0] * rows
              right = bbox[3] * cols
              bottom = bbox[2] * rows
              cv2.rectangle(detections, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)


      image_message = self.bridge.cv2_to_imgmsg(detections, encoding="passthrough")



      self.TLDetector_pub.publish(image_message)
      return output_dict


    def camera_cb(self, msg):
        #print("Got camera info")
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
        if self.image_counter % 5 != 0:           
            return
        # End speed-up

        self.has_image = True
        self.hasDetections = False
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
            #print("Got map transform")
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.log_err("Couldn't find camera to map transform.")
            #print("Can't get map transform")
                
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
        crop_cv_image = cv_image.copy()
        
        num_detections = int(self.detections['num_detections'])        
        rows = cv_image.shape[0]
        cols = cv_image.shape[1]

        index = np.argmax(self.detections['detection_scores'])
        #rospy.loginfo('most confident detection = ', index)
        #rospy.loginfo('max score = ', self.detections['detection_scores'][index])
        
        for i in range(num_detections):
            bbox = [float(v) for v in self.detections['detection_boxes'][i]]
            score = self.detections['detection_scores'][i]
            if score > 0.3:
                x = int(bbox[1] * cols)
                y = int(bbox[0] * rows)
                right = int(bbox[3] * cols)
                bottom = int(bbox[2] * rows)

                crop_cv_image = cv_image[y:bottom, x:right].copy()
                image_message = self.bridge.cv2_to_imgmsg(crop_cv_image, encoding="passthrough")
                self.most_confident_detection.publish(image_message)
                return self.light_classifier.get_classification(crop_cv_image)
                
        return TrafficLight.UNKNOWN
        
                
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
        if tl_idx >= 0:
            if TESTING_WITHOUT_IMG:
                state = self.get_sim_light_state(tl_idx)
                return line_wp_idx, state
                
            else:
                # Fix this for image processing.
                self.detections = self.getBBox()
                self.hasDetections = True
                state = self.get_light_state(tl_idx)
                #print("Light state: {0}".format(state))
                return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
