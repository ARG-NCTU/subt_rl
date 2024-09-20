#! /usr/bin/env python3
import os
import rospy
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from sensor_msgs.msg import LaserScan, Joy
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import json

class GoalNav(object):
    def __init__(self):
        super().__init__()
        self.max_dis = 10  # meters
        self.laser_n = 4
        self.pos_n = 10
        self.frame = rospy.get_param("~frame", "odom")

        #0.5
        self.count = 0
        self.last_pos = None
        self.last_time = None
        self.time_diff = 0
        self.start_record = False
        self.goal_list = []

        # subscriber, timer
        self.sub_joy = rospy.Subscriber("joy", Joy, self.cb_joy, queue_size=1)
        self.sub_odom = rospy.Subscriber(
            "odom_in", PoseStamped, self.cb_odom, queue_size=1)
        # self.timer = rospy.Timer(rospy.Duration(0.1), self.inference)

    def cb_joy(self, msg):
        logo_button = 8

        if msg.buttons[logo_button] == 1 and self.start_record is False:
            self.start_record = True
            self.goal_list = []
            print("start record")
            # sleep 1s
            rospy.sleep(1)
            
        elif msg.buttons[logo_button] == 1 and self.start_record is True:
            self.start_record = False
            print("stop record")
            # write goal_list to file
            self.write_goal_list_to_json()
            rospy.sleep(1)

    def cb_odom(self, msg):
        if self.start_record is False:
            print("not start record")
            return

        # caculate angle diff
        new_pos = [msg.pose.position.x, msg.pose.position.y]

        if self.last_pos is None:
            self.last_pos = new_pos
            self.goal_list.append(new_pos)

        # distance beteween new_pos and last_pos
        diff = np.sqrt((new_pos[0]-self.last_pos[0])**2+(new_pos[1]-self.last_pos[1])**2)

        if diff > 3:
            self.goal_list.append(new_pos)
            self.last_pos = new_pos
            print("new goal: ", new_pos)
            # write goal_list to file
            self.write_goal_list_to_json()
        
    def write_goal_list_to_json(self):
        with open('/home/argsubt/subt-virtual/catkin_ws/src/subt-core/subt_rl/src/goal_list.json', 'w') as f:
            json.dump(self.goal_list, f)
        
        

if __name__ == "__main__":
    rospy.init_node("goal_nav_rl")
    goalNav = GoalNav()
    rospy.spin()
