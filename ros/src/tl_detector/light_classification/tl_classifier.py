from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy

import operator

class TLClassifier(object):
    def __init__(self):
        pass

    def get_classification(self, image):

        answer = {}
        temp = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        green_lower = np.array([40,100,100])
        green_upper = np.array([80,255,255])
        
        answer['green'] = {}
        answer['green']['mask'] = (cv2.inRange(temp, green_lower, green_upper))
        
        green_score = cv2.countNonZero(answer['green']['mask'])
        answer['green']['score'] = green_score
        
        yellow_lower = np.array([80,100,100])
        yellow_upper = np.array([100,255,255])
        
        answer['yellow'] = {}
        answer['yellow']['mask'] = (cv2.inRange(temp, yellow_lower, yellow_upper))
        
        yellow_score = cv2.countNonZero(answer['yellow']['mask'])
        answer['yellow']['score'] = yellow_score
        
        red_lower = np.array([100,100,100])
        red_upper = np.array([150,255,255])
        
        answer['red'] = {}
        answer['red']['mask'] = (cv2.inRange(temp, red_lower, red_upper))
        
        red_score = cv2.countNonZero(answer['red']['mask'])
        answer['red']['score'] = red_score

        numbers = {}
        for k in answer.keys():
            numbers[k] = answer[k]['score']
            
        
        #rospy.loginfo(numbers)
        result = max(numbers.items(), key=operator.itemgetter(1))[0]
        
        if(result == 'green'):
            #rospy.loginfo('GREEN')
            return TrafficLight.GREEN

        if(result == 'yellow'):
            #rospy.loginfo('YELLOW')
            return TrafficLight.YELLOW

        if(result == 'red'):
            #rospy.loginfo('RED')
            return TrafficLight.RED
        
        #rospy.loginfo('UNKNOWN')
        return TrafficLight.UNKNOWN
