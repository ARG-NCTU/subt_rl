#! /usr/bin/env python3
import os
import rospy
import tensorflow as tf
import numpy as np
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Joy
from std_msgs.msg import Bool,Int16
from scipy.spatial.transform import Rotation as R
from pozyx_simulation.msg import uwb_data
import time
class GoalNav(object):
	def __init__(self):
		super().__init__()
		self.max_dis = 10  # meters
		self.laser_n = 4
		self.pos_n = 15
		self.action_scale = {'linear': rospy.get_param(
			'~linear_scale', 0.33), 'angular': rospy.get_param("~angular_scale", 0.18)}

		self.auto = 0
		self.goal = 1
		self.start = 0
		self.pos_track = None
		self.laser_stack = None
		self.last_pos = None
		self.reach_goal = False

		self.last_omega = 0
		self.omega_gamma = 0.25

		self.vel_ratio = 0

		# network
		gpu = tf.config.experimental.list_physical_devices('GPU')
		tf.config.experimental.set_memory_growth(gpu[0], True)
		my_dir = os.path.abspath(os.path.dirname(__file__))
		model_path = os.path.join(my_dir, "../model/goal_uwb/policy")
		self.policy_network = tf.saved_model.load(model_path)

		# pub cmd
		self.pub_cmd = rospy.Publisher("/X2/x2_velocity_controller/cmd_vel", Twist, queue_size=1)
		self.pub_start = rospy.Publisher("uwb_start_index", Int16, queue_size=1)

		# subscriber, timer
		self.sub_joy = rospy.Subscriber("/joy", Joy, self.cb_joy, queue_size=1)
		self.subt_uwb = rospy.Subscriber("/uwb_data_distance", uwb_data, self.cb_uwb, queue_size=1)
		self.sub_laser = rospy.Subscriber("/RL/scan",  LaserScan, self.cb_laser, queue_size=1)
		self.timer = rospy.Timer(rospy.Duration(0.1), self.inference)




	def euler_from_quaternion(self, x, y, z, w):
		"""
		Convert a quaternion into euler angles (roll, pitch, yaw)
		roll is rotation around x in radians (counterclockwise)
		pitch is rotation around y in radians (counterclockwise)
		yaw is rotation around z in radians (counterclockwise)
		"""
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = math.atan2(t0, t1)

		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch_y = math.asin(t2)

		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)

		return roll_x, pitch_y, yaw_z # in radians

	def cb_joy(self, msg):
		start_button = 7
		back_button = 6

		if (msg.buttons[start_button] == 1) and not self.auto:
			self.auto = 1
			rospy.loginfo('go auto')
		elif msg.buttons[back_button] == 1 and self.auto:
			self.auto = 0
			rospy.loginfo('go manual')


	def rotate(self, radians):
		msg = PoseStamped()
		msg = rospy.wait_for_message('/truth_map_posestamped', PoseStamped, timeout=5)

		r,p, yaw = self.euler_from_quaternion(msg.pose.orientation.x,\
											msg.pose.orientation.y,\
											msg.pose.orientation.z,\
											msg.pose.orientation.w)
		yaw_begin = yaw
		while(abs(yaw - yaw_begin) < abs(radians)):
			print("rotating ",abs(yaw - yaw_begin))
			cmd = Twist()
			cmd.linear.x = 0
			cmd.angular.z = 0.25 if radians>0 else -0.25
			self.pub_cmd.publish(cmd)
			time.sleep(0.5)
			msg = rospy.wait_for_message('/truth_map_posestamped', PoseStamped, timeout=5)
			r, p, yaw = self.euler_from_quaternion(msg.pose.orientation.x,\
												msg.pose.orientation.y,\
												msg.pose.orientation.z,\
												msg.pose.orientation.w)


	def cb_uwb(self, msg):
		if msg.distance[self.goal]*0.001 < 1: self.reach_goal = True
		track_pos = np.array([msg.distance[self.start]*0.001, msg.distance[self.goal]*0.001])
		if self.pos_track is None:
			self.pos_track = np.tile(track_pos, (self.pos_n, 1))
		else:
			self.pos_track[:-1] = self.pos_track[1:]
			np.append(self.pos_track, track_pos)

	def cb_laser(self, msg):
		ranges = np.array(msg.ranges)
		ranges = np.clip(ranges, 0, self.max_dis)

		if self.laser_stack is None:
			self.laser_stack = np.tile(ranges, (self.laser_n, 1))
		else:
			self.laser_stack[:-1] = self.laser_stack[1:]
			self.laser_stack[-1] = ranges

	def switch_goal(self):
		if (self.goal==3):return # Done

		self.start = self.goal
		self.goal += 1
		# if((self.goal == 4) or (self.goal == 8)):
		# 	msg = PoseStamped()
		# 	msg = rospy.wait_for_message('/truth_map_posestamped', PoseStamped, timeout=5)
		#
		# 	r,p, yaw_begin = self.euler_from_quaternion(msg.pose.orientation.x,\
		# 										msg.pose.orientation.y,\
		# 										msg.pose.orientation.z,\
		# 										msg.pose.orientation.w)
		# 	yaw = yaw_begin
		# 	if(self.goal == 4):
		# 		self.goal = 5
		# 		# self.rotate(-1.04)
		# 	if(self.goal == 8):
		# 		self.goal = 9
		# 		# self.rotate(1.04)



	def inference(self, event):
		int16 = Int16()
		int16.data = self.start
		self.pub_start.publish(int16)
		if self.pos_track is None:
			rospy.loginfo("uwb is None")
			return
		if self.laser_stack is None:
			rospy.loginfo("laser is None")
			return
		if self.auto == 0:
			rospy.loginfo("Not auto")
			return




		if self.reach_goal:
			rospy.loginfo("goal reached anchor"+str(self.goal))
			self.switch_goal()
			self.reach_goal = False
			return


		# reshape
		laser = self.laser_stack.reshape(-1)
		track = self.pos_track.reshape(-1)
		state = np.append(laser, track)

		state = tf.convert_to_tensor([state], dtype=tf.float32)

		action = self.policy_network(state)[0].numpy()
		self.last_omega = self.omega_gamma * \
			action[1] + (1-self.omega_gamma)*self.last_omega

		cmd = Twist()
		cmd.linear.x = action[0]*self.action_scale['linear']
		cmd.angular.z = self.last_omega * \
			self.action_scale['angular']

		self.pub_cmd.publish(cmd)
		rospy.loginfo( "Moving to anchor " + str(self.goal) )


if __name__ == "__main__":
	rospy.init_node("goal_nav_rl")
	goalNav = GoalNav()
	rospy.spin()
