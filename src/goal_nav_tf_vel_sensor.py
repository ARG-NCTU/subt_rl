#! /usr/bin/env python3
import os
import rospy
import tensorflow as tf
import numpy as np
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Joy
from std_msgs.msg import Bool
from scipy.spatial.transform import Rotation as R
import yaml

class GoalNav(object):
    def __init__(self):
        super().__init__()
        self.record_yaml = False
        self.record_list = []
        self.max_dis = 10  # meters
        self.laser_n = 4
        self.pos_n = 10
        self.frame = rospy.get_param("~frame", "odom")
        self.mission = rospy.get_param("~mission", "cave_240")
        self.goal_dis = rospy.get_param("~goal_dis", 4)
        self.action_scale = {'linear': rospy.get_param(
            '~linear_scale', 0.45), 'angular': rospy.get_param("~angular_scale", 0.45)}
        #0.5
        self.auto = 0
        self.goal = None
        self.pos_track = None
        self.velocity_track = None
        self.laser_stack = None
        self.count = 0
        self.last_pos = None
        self.last_time = None
        self.time_diff = 0
        self.velocity = 0

        self.last_omega = 0
        self.omega_gamma = 0.25

        self.vel_ratio = 0

        # network
        obs_dim = 243
        action_dim = 2
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)
        my_dir = os.path.abspath(os.path.dirname(__file__))
        if self.mission == "cave_240":
            model_path = os.path.join(my_dir, "../model/cave/snapshots/policy")
        elif self.mission == "curriculum_90":
            model_path = os.path.join(my_dir, "../model/degree_90_curriculum/snapshots/policy")
        
        self.policy_network = tf.saved_model.load(model_path)

        # pub cmd
        self.pub_cmd = rospy.Publisher("cmd_out", Twist, queue_size=1)
        self.pub_visual_laser = rospy.Publisher("visual_laser", LaserScan, queue_size=1)

        # subscriber, timer
        self.sub_joy = rospy.Subscriber("joy", Joy, self.cb_joy, queue_size=1)
        self.sub_goal = rospy.Subscriber(
            "goal_in", PoseStamped, self.cb_goal, queue_size=1)
        self.sub_odom = rospy.Subscriber(
            "odom_in", PoseStamped, self.cb_odom, queue_size=1)
        self.sub_laser = rospy.Subscriber(
            "laser_in",  LaserScan, self.cb_laser, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.inference)

    def scale_pose(self, value):
        if value > 0:
            return math.log(1 + value)
        elif value < 0:
            return -math.log(1 + abs(value))

    def cb_joy(self, msg):
        start_button = 7
        back_button = 6
        if (msg.buttons[start_button] == 1) and not self.auto:
            self.auto = 1
            rospy.loginfo('go auto')
        elif msg.buttons[back_button] == 1 and self.auto:
            self.auto = 0
            rospy.loginfo('go manual')

    def cb_goal(self, msg):
        if msg.header.frame_id != self.frame:
            self.goal = None
            return

        self.goal = np.array([
            msg.pose.position.x, msg.pose.position.y])
        # print(self.goal)

    def cb_odom(self, msg):
        if self.goal is None:
            self.pos_track = None
            return

        # caculate angle diff
        new_pos = np.array(
            [msg.pose.position.x, msg.pose.position.y])
        diff = self.goal - new_pos
        time = rospy.get_rostime()

        if self.last_time is not None and self.last_pos is not None:
            self.time_diff = (time.to_nsec()-self.last_time.to_nsec())/1000000000
            distance = math.sqrt((new_pos[0]-self.last_pos[0])**2+(new_pos[1]-self.last_pos[1])**2)
            ## map frame, 0.001
            if self.time_diff == 0:
                self.time_diff = 0.067
            self.velocity = (distance/self.time_diff)
            # print("velocity: ", self.velocity)
            # print("time_diff: ", self.time_diff)

        self.velocity = np.array([self.velocity])
        r = R.from_quat([msg.pose.orientation.x,
                         msg.pose.orientation.y,
                         msg.pose.orientation.z,
                         msg.pose.orientation.w])
        yaw = r.as_euler('zyx')[0]
        angle = math.atan2(diff[1], diff[0]) - yaw
        if angle >= np.pi:
            angle -= 2*np.pi
        elif angle <= -np.pi:
            angle += 2*np.pi

        # update pose tracker
        diff = np.array([self.scale_pose(v) for v in diff])
        track_pos = np.append(diff, angle)
        # print("velocity: ", self.velocity)
        if self.pos_track is None:
            self.pos_track = np.tile(track_pos, (self.pos_n, 1))
        else:
            self.pos_track[:-1] = self.pos_track[1:]
            self.pos_track[-1] = track_pos

        if self.velocity_track is None:
            self.velocity_track = np.tile(float(self.velocity), (self.pos_n, 1))
        else:
            self.velocity_track[:-1] = self.velocity_track[1:]
            self.velocity_track[-1] = float(self.velocity)
        # print("pos_track: ", self.pos_track)
        # print("velocity_track: ", self.velocity_track)
        self.last_pos = new_pos
        self.last_time = time

    def cb_laser(self, msg):
        ranges = np.array(msg.ranges)
        # if(self.count<=100):
        #     # print("90 degree")
        #     visual_laser = LaserScan()
        #     visual_laser = msg
        #     visual_laser.ranges = ranges
        #     for i in range(len(visual_laser.ranges)):
        #         if(i<=74 or i>=165):
        #             visual_laser.ranges[i] = 100
        #     self.pub_visual_laser.publish(visual_laser)

        #     for i in range(len(ranges)):
        #         if(i<=74 or i>=165):
        #             ranges[i] = self.max_dis
        
        # elif(self.count>100 and self.count<=200):
        #     # print("240 degree")
        #     visual_laser = LaserScan()
        #     visual_laser = msg
        #     visual_laser.ranges = ranges
        #     self.pub_visual_laser.publish(visual_laser)

        # else:
        #     # print("reset count")
        #     self.count = 0

        # visual_laser = LaserScan()
        # visual_laser = msg
        # visual_laser.ranges = ranges
        # for i in range(len(visual_laser.ranges)):
        #     if(i<=74 or i>=165):
        #         visual_laser.ranges[i] = 100
        # self.pub_visual_laser.publish(visual_laser)

        # for i in range(len(ranges)):
        #     if(i<=74 or i>=165):
        #         ranges[i] = self.max_dis
        
        ranges = np.clip(ranges, 0, self.max_dis)
        
        # print(ranges)
        if self.laser_stack is None:
            self.laser_stack = np.tile(ranges, (self.laser_n, 1))
        else:
            self.laser_stack[:-1] = self.laser_stack[1:]
            self.laser_stack[-1] = ranges

    def inference(self, event):
        if self.goal is None:
            print("no goal")
            return
        if self.pos_track is None:
            print("no pos track")
            return
        if self.laser_stack is None:
            print("no laser")
            return
        if self.auto == 0:
            print("no auto")
            return
        if self.record_yaml:
            frame_data = {'range':self.laser_stack.tolist(),'relative_pose':self.pos_track.tolist(), 'velocity': self.velocity_track.tolist()}
            self.record_list.append(frame_data)
            print('frame_data lenght',len(self.record_list))

        self.count += 1
        # print("inference")
        dis = np.linalg.norm(self.goal-self.last_pos)
        if dis < self.goal_dis:
            rospy.loginfo("goal reached")
            self.goal = None
            cmd = Twist()
            cmd.linear.x = 0
            cmd.angular.z = 0
            self.pub_cmd.publish(cmd)
            return

        # self.vel_ratio = rospy.get_param("/velocity_mode", 4) * (1./5)

        # reshape
        laser = self.laser_stack.reshape(-1)
        track = self.pos_track.reshape(-1)
        vel = self.velocity_track.reshape(-1)
        state = np.append(laser, track)
        state = np.append(state, vel)

        state = tf.convert_to_tensor([state], dtype=tf.float32)

        action = self.policy_network(state)[0].numpy()
        self.last_omega = self.omega_gamma * \
            action[1] + (1-self.omega_gamma)*self.last_omega

        cmd = Twist()
        if action[0] < 0:
            print("slow down")
            
        cmd.linear.x = action[0]*self.action_scale['linear']

        cmd.angular.z = self.last_omega * \
            self.action_scale['angular']

        self.pub_cmd.publish(cmd)


if __name__ == "__main__":
    rospy.init_node("goal_nav_rl")
    goalNav = GoalNav()
    rospy.spin()
