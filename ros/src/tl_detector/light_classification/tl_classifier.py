from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy
import tensorflow as tf
import os

# inference code using pre-trained model taken from here:
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

class TLClassifier(object):
    def __init__(self):
        
        # set up parameters
        # labels from: https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt
        self.TRAFFIC_LIGHT_LABEL = 10
        self.DETECTION_THRESHOLD = 0.8
        
        # load frozen graph
        self.path_to_frozen_graph = 'light_classification/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')
            
            # get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
 
    def detect_traffic_light(self, image):
        # detects traffic light in an image
        
        with self.detection_graph.as_default():
            with tf.Session() as sess:

                # run inference
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                image_for_inference = np.array(image).reshape(1,600,800,3)
                output_dict = sess.run(self.tensor_dict,feed_dict={image_tensor: image_for_inference})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                
                # check for traffic light above threshold score
                # only need to find one (multiple lights in simulator are all the same color)
                top_score = output_dict['detection_scores'][0]
                if (output_dict['detection_classes'][0] == self.TRAFFIC_LIGHT_LABEL) and (top_score > self.DETECTION_THRESHOLD):
                    print("Found traffic light with certainty: {0}".format(top_score))
                    print(output_dict['detection_boxes'][0])
                    # crop image to detection box of traffic light
                    height, width, color = image.shape
                    # unclear how to interpret detection box values
                    #y1,x1,y2,x2 = output_dict['detection_boxes'][0]
                    x1,y1,x2,y2 = output_dict['detection_boxes'][0]
                    tl_image = image[int(y1*height):int(y2*height),int(x1*width):int(x2*width),:]
                    cv2.imshow("traffic light", tl_image)
                    cv2.waitKey(1)
                    return tl_image
                else:
                    return np.array(0)
                        
    def get_classification(self, image):

        #cv2.imshow("traffic light", image)
        #cv2.waitKey(1)

        answer = {}
        temp = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        green_lower = np.array([60,80,80])
        green_upper = np.array([80,255,255])
        
        answer['green'] = {}
        answer['green']['mask'] = (cv2.inRange(temp, green_lower, green_upper))
        
        green_score = cv2.countNonZero(answer['green']['mask'])
        answer['green']['score'] = green_score
        
        yellow_lower = np.array([20,80,80])
        yellow_upper = np.array([40,255,255])
        
        answer['yellow'] = {}
        answer['yellow']['mask'] = (cv2.inRange(temp, yellow_lower, yellow_upper))
        
        yellow_score = cv2.countNonZero(answer['yellow']['mask'])
        answer['yellow']['score'] = yellow_score
        
        red_lower = np.array([0,60,60])
        red_upper = np.array([10,255,255])
        
        answer['red'] = {}
        answer['red']['mask'] = (cv2.inRange(temp, red_lower, red_upper))
        
        red_score = cv2.countNonZero(answer['red']['mask'])
        answer['red']['score'] = red_score

        if(answer['green']['score'] > 200):
            rospy.loginfo('GREEN')
            return TrafficLight.GREEN

        if(answer['yellow']['score'] > 200):
            rospy.loginfo('YELLOW')
            return TrafficLight.YELLOW

        if(answer['red']['score'] > 200):
            rospy.loginfo('RED')
            return TrafficLight.RED
        
        rospy.loginfo('UNKNOWN')
        return TrafficLight.UNKNOWN
