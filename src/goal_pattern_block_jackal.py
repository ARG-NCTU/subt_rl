#! /usr/bin/env python3
import os
import rospy
import tensorflow as tf
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Joy
from scipy.spatial.transform import Rotation as R
import pyivp
from gazebo_msgs.msg import ModelState
from nav_msgs.msg import Path
from std_msgs.msg import Bool

class GoalNav(object):
    def __init__(self):
        super().__init__()
        self.max_dis = 10  # meters
        self.laser_n = 4
        self.pos_n = 10
        self.frame = rospy.get_param("~frame", "map")
        self.goal_dis = rospy.get_param("~goal_dis", 4)
        self.veh = rospy.get_param("~veh", "wamv")
        self.task = 0
        self.pub_goal = rospy.Publisher("goal_out", PoseStamped, queue_size=1)
        # self.pub_start_draw_path = rospy.Publisher("/start_task", Bool, queue_size=1)
        self.auto = 0
        self.goal = None
        self.pos_track = None
        self.velocity_track = None

        self.laser_stack = None
        self.last_pos = None
        self.last_time = None
        self.time_diff = 0
        self.velocity = 0
        self.last_omega = 0
        self.omega_gamma = 0.25

        self.vel_ratio = 0
        self.goal_list = []
        self.generate_goal()
        self.set_wamv2 = True
        # network
        obs_dim = 243
        action_dim = 2

        # pub cmd
        self.pub_cmd = rospy.Publisher("cmd_out", Twist, queue_size=1)
        
        self.stop = Twist()
        self.stop.angular.x =0
        self.stop.angular.y =0
        self.stop.angular.z =0
        self.stop.linear.x =0
        self.stop.linear.y =0
        self.stop.linear.z =0
        self.pub_cmd.publish(self.stop)

        # trajectory visualization
        self.pub_wamv2_path = rospy.Publisher('path', Path, queue_size=1)
        self.wamv2_path = Path()
        # subscriber, timer
        self.sub_joy = rospy.Subscriber("joy", Joy, self.cb_joy, queue_size=1)
        self.sub_odom = rospy.Subscriber("odom_in", PoseStamped, self.cb_odom, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.inference)

    

    def generate_goal(self):
        self.move_wamv3_wamv4()
        pattern_block = pyivp.string_to_seglist(" format=lawnmower, x = 0, y = 0, height = 3, width = 8, lane_width = 1.5,\
                        rows = north - south, startx = 0, starty = 0, degs = 180")
        goal_pair = pattern_block.size()
        for i in range(1, goal_pair):
            s_x = pattern_block.get_vx(i)
            s_y = pattern_block.get_vy(i)
            print(s_x, s_y)
            self.goal_list.append([s_x, s_y])
        print(self.goal_list)
        # set wamv2 to the first goal
        self.task = 1


       
    def scale_pose(self, value):
        if value > 0:
            return math.log(1 + value)
        elif value < 0:
            return -math.log(1 + abs(value))

    def cb_joy(self, msg):
        start_button = 7
        back_button = 6

        if (msg.buttons[start_button] == 1) and not self.auto:
            rospy.loginfo('go auto')
            self.auto = 1

        elif msg.buttons[back_button] == 1 and self.auto:
            self.auto = 0
            rospy.loginfo('go manual')

    def goal_pubilsher(self,goal_array):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = goal_array[0]
        goal.pose.position.y = goal_array[1]

        self.pub_goal.publish(goal)
        
    def cb_odom(self, msg):
        new_pos = np.array(
            [msg.pose.position.x, msg.pose.position.y])
        diff = self.goal - new_pos
        time = rospy.get_rostime()

        self.last_pos = new_pos
        self.last_time = time

    def inference(self, event):
        if self.goal is None and self.task < len(self.goal_list):
            self.goal= np.array([self.goal_list[self.task][0], self.goal_list[self.task][1]])
            self.goal_pubilsher(self.goal)
            print(f'no goal and give new goal {self.goal}')
        elif self.goal is None:
            self.pub_cmd.publish(self.stop)
            print('no goal')
            rospy.signal_shutdown('finish')
            return

        if self.auto == 0:
            self.pub_cmd.publish(self.stop)
            return

        dis = np.linalg.norm(self.goal-self.last_pos)
        if dis < self.goal_dis:
            rospy.loginfo("goal reached")
            self.task += 1
            self.goal = None

            return


if __name__ == "__main__":
    rospy.init_node("goal_pattern_block_rl")
    goalNav = GoalNav()
    rospy.spin()