from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self):
        pass

    def get_classification(self, image):

        cv2.imshow("traffic light", image)
        cv2.waitKey(1)

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
