#! /usr/bin/env python3

'''
This script is used to evaluate the performance of the rl trained model in the Gazebo simulation environment.
According to the mission, the robot will navigate to the goal point pair in different environments, cave or office.
Each goal point pair will be navigated ten times.
Each environment will be evaluated by two sets of different goal point pairs.
The goal point pairs are defined in the goal_point_pair_1 and goal_point_pair_2 dictionary.
The result of the evaluation will be save in the csv file, and print the success rate of each goal point pair.
The mission will be difined fail if the robot stays in the similar position for 30 seconds.
'''

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

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState

class GoalNavEvaluation():
    def __init__(self):
        super().__init__()
        self.record_list = []
        self.max_dis = 10  # meters
        self.laser_n = 4
        self.pos_n = 10
        self.frame = rospy.get_param("~frame", "odom")
        #  mission : cave_240, cave_90, office_240, office_90, cave_90_240, office_90_240
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
        self.time_change_laser = rospy.get_rostime()
        self.laser_mode = 1

        self.last_omega = 0
        self.omega_gamma = 0.25
        self.vel_ratio = 0

        # goal point pair
        self.goal_point_pair_1 = {"cave" : {"start" : [62.03, -0.53, 0], "end" : [80.2, -10.44, -1.52]}, "office" : {"start" : [-6.48, 1.99, 0], "end" : [2.83, 2.84, 0]}}
        self.goal_point_pair_2 = {"cave" : {"start" : [100.58, -0.38, 0], "end" : [117.88, 5.77, 0.75]} , "office" : {"start" : [5.48, 3.86, 1.6], "end" : [2.34, 17.93, -3.12]}}

        # network
        obs_dim = 243
        action_dim = 2
        gpu = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu[0], True)
        my_dir = os.path.abspath(os.path.dirname(__file__))

        # load model
        # model_path = os.path.join(my_dir, "../model/{folder_name}/snapshots/policy")
        # model_path = os.path.join(my_dir, "../model/cave/snapshots/policy")
        model_path = os.path.join(my_dir, "../model/forest_load/snapshots/policy")
        self.policy_network = tf.saved_model.load(model_path)

        # pub cmd
        self.pub_cmd = rospy.Publisher("cmd_out", Twist, queue_size=1)
        self.pub_visual_laser = rospy.Publisher("visual_laser", LaserScan, queue_size=1)
        self.pub_goal = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)

        # subscriber, timer
        self.sub_joy = rospy.Subscriber("joy", Joy, self.cb_joy, queue_size=1)
        self.sub_goal = rospy.Subscriber(
            "goal_in", PoseStamped, self.cb_goal, queue_size=1)
        self.sub_odom = rospy.Subscriber(
            "odom_in", PoseStamped, self.cb_odom, queue_size=1)
        self.sub_laser = rospy.Subscriber(
            "laser_in",  LaserScan, self.cb_laser, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.inference)

        # evaulation
        self.evaluation_round = 0
        self.Max_Evaluation_Time = 10
        self.current_evaluation_time = 0
        self.success = {0: 0, 1: 0}

        # decide if the robot is in the similar position for 30 seconds
        self.similar_position = 0
        self.stay_position = None

        # init environment
        self.reset_model = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.initial_state()


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
        
        # decide if the robot is in the similar position for 30 seconds
        if self.stay_position is None:
            self.stay_position = new_pos
        else:
            if (abs(new_pos[0]-self.stay_position[0]) < 0.1) and (abs(new_pos[1]-self.stay_position[1]) < 0.1):
                self.similar_position += 1
            else:
                self.similar_position = 0
                self.stay_position = new_pos

        self.last_pos = new_pos
        self.last_time = time

    def cb_laser(self, msg):
        ranges = np.array(msg.ranges)
        visual_laser = LaserScan()
        visual_laser = msg
        visual_laser.ranges = ranges

        if self.mission == "cave_240" or self.mission == "office_240":
            self.laser_mode = 1

        if self.mission == "cave_90" or self.mission == "office_90":
            self.laser_mode = -1
        
        if self.mission == "cave_90_240" or self.mission == "office_90_240":
            time_now = rospy.get_rostime()
            # change laser mode every 5 seconds
            if (time_now.to_sec() - self.time_change_laser.to_sec()) > 5:
                rospy.loginfo("change laser mode")
                self.time_change_laser = time_now
                self.laser_mode = -self.laser_mode

        if self.laser_mode == -1:
            for i in range(len(visual_laser.ranges)):
                if(i<=74 or i>=165):
                    visual_laser.ranges[i] = 100

        self.pub_visual_laser.publish(visual_laser)
        ranges = np.clip(ranges, 0, self.max_dis)
        
        # laser stack will overlay 4 laser scans together. Hence, when the laser stack is None, we will tile the first laser scan 4 times.
        if self.laser_stack is None:
            self.laser_stack = np.tile(ranges, (self.laser_n, 1))
        else:
            self.laser_stack[:-1] = self.laser_stack[1:]
            self.laser_stack[-1] = ranges

    def publish_goal(self):
        goal = None
        if self.mission == "cave_240" or self.mission == "cave_90" or self.mission == "cave_90_240":
            if self.evaluation_round == 0:
                goal = self.goal_point_pair_1["cave"]["end"]
            else:
                goal = self.goal_point_pair_2["cave"]["end"]
        else:
            if self.evaluation_round == 0:
                goal = self.goal_point_pair_1["office"]["end"]
            else:
                goal = self.goal_point_pair_2["office"]["end"]
        
        r = R.from_euler('xyz', [0, 0, goal[2]])
        quaternion = r.as_quat()
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position.x = goal[0]
        goal_pose.pose.position.y = goal[1]
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.x = quaternion[0]
        goal_pose.pose.orientation.y = quaternion[1]
        goal_pose.pose.orientation.z = quaternion[2]
        goal_pose.pose.orientation.w = quaternion[3]
        goal_pose.header.stamp = rospy.Time.now()
        self.pub_goal.publish(goal_pose)

    def reset(self):
        self.pos_track = None
        self.laser_stack = None
        self.count = 0
        self.last_pos = None
        self.last_time = None
        self.time_diff = 0
        self.velocity = 0
        self.time_change_laser = rospy.get_rostime()
        self.laser_mode = 1
        self.last_omega = 0
        self.omega_gamma = 0.25
        self.vel_ratio = 0
        self.similar_position = 0
        self.stay_position = None

    def initial_state(self):
        self.state_msg = ModelState()
        self.state_msg.model_name = "X1"

        start_position = None
        if(self.mission == "cave_240" or self.mission == "cave_90" or self.mission == "cave_90_240"):
            if self.evaluation_round == 0:
                start_position = self.goal_point_pair_1["cave"]["start"]
                print(f"Mission: {self.mission}, Environment: cave, Goal point pair 1")
            else:
                start_position = self.goal_point_pair_2["cave"]["start"]
                print(f"Mission: {self.mission}, Environment: cave, Goal point pair 2")
        else:
            if self.evaluation_round == 0:
                start_position = self.goal_point_pair_1["office"]["start"]
                print(f"Mission: {self.mission}, Environment: office, Goal point pair 1")
            else:
                start_position = self.goal_point_pair_2["office"]["start"]
                print(f"Mission: {self.mission}, Environment: office, Goal point pair 2")
        
        r = R.from_euler('xyz', [0, 0, start_position[2]])
        quaternion = r.as_quat()
        
        self.state_msg.pose.position.x = start_position[0]
        self.state_msg.pose.position.y = start_position[1]
        self.state_msg.pose.position.z = 0.0
        self.state_msg.pose.orientation.x = quaternion[0]
        self.state_msg.pose.orientation.y = quaternion[1]
        self.state_msg.pose.orientation.z = quaternion[2]
        self.state_msg.pose.orientation.w = quaternion[3]
        
        self.reset_model(self.state_msg)
    
    def inference(self, event):
        if self.evaluation_round == 1 and self.current_evaluation_time == self.Max_Evaluation_Time:
            print("evaluation finished")
            print("goal point pair 1 success time: ", self.success[0])
            print("goal point pair 2 success time: ", self.success[1])
            print("goal point pair 1 success rate: ", self.success[0]/self.Max_Evaluation_Time)
            print("goal point pair 2 success rate: ", self.success[1]/self.Max_Evaluation_Time)
            print("total success rate: ", (self.success[0]+self.success[1])/(2*self.Max_Evaluation_Time))
            return

        # if the current evaluation time equals to the max evaluation time, we will change the goal point pair.
        if self.current_evaluation_time == self.Max_Evaluation_Time and self.evaluation_round < 1:
            self.current_evaluation_time = 0
            self.evaluation_round += 1
            self.reset()
            self.initial_state()

        self.publish_goal()
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
        
        self.count += 1
        dis = np.linalg.norm(self.goal-self.last_pos)
        
        if dis < self.goal_dis:
            self.success[self.evaluation_round] += 1
            self.current_evaluation_time += 1
            rospy.loginfo("goal reached")
            rospy.loginfo("Mission: %s, Evaluation Round: %d, Current evaluation time: %d ,Success: %d", self.mission, self.evaluation_round, self.current_evaluation_time, self.success[self.evaluation_round])
            cmd = Twist()
            cmd.linear.x = 0
            cmd.angular.z = 0
            self.pub_cmd.publish(cmd)
            # if the robot reaches the goal point pair, we will reset the robot to the start point of the goal point pair and intialize the environment.
            self.reset()
            self.initial_state()
        elif self.similar_position > 300 or self.count > 3000:
            self.current_evaluation_time += 1
            rospy.loginfo("mission failed")
            rospy.loginfo("Mission: %s, Evaluation Round: %d, Current evaluation time: %d ,Success: %d", self.mission, self.evaluation_round, self.current_evaluation_time, self.success[self.evaluation_round])
            cmd = Twist()
            cmd.linear.x = 0
            cmd.angular.z = 0
            self.pub_cmd.publish(cmd)
            # if the robot stays in the similar position for 30 seconds, we will reset the robot to the start point of the goal point pair and intialize the environment.
            self.reset()
            self.initial_state()
        else:

            # reshape
            laser = self.laser_stack.reshape(-1)
            track = self.pos_track.reshape(-1)
            vel = self.velocity_track.reshape(-1)
            state = np.append(laser, track)
            # state = np.append(state, vel)

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
    goalNav = GoalNavEvaluation()
    rospy.spin()
