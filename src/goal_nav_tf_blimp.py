#! /usr/bin/env python3
import os
import rospy
import tensorflow as tf
import numpy as np
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Joy, PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Bool
from scipy.spatial.transform import Rotation as R


class PID_control():
	def __init__(self, p_name, P=0.2, I=0.0, D=0.0):
		print("Turn on PID", p_name, "Control")
		self.Kp = P
		self.Ki = I
		self.Kd = D

		self.sample_time = 0.00
		self.current_time = rospy.get_time()
		self.last_time = self.current_time

		self.clear()

	def clear(self):
		"""Clears PID computations and coefficients"""
		self.SetPoint = 0.0

		self.PTerm = 0.0
		self.ITerm = 0.0
		self.DTerm = 0.0
		self.last_error = 0.0

		# Windup Guard
		self.int_error = 0.0
		self.windup_guard = 20.0

		self.output = 0.0


	def update(self, feedback_value):
		"""Calculates PID value for given reference feedback
		.. math::
			u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
		.. figure:: images/pid_1.png
			:align:   center
			Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
		"""
		error = self.SetPoint - feedback_value

		self.current_time = rospy.get_time()
		delta_time = self.current_time - self.last_time
		delta_error = error - self.last_error

		if (delta_time >= self.sample_time):
			self.PTerm = self.Kp * error
			self.ITerm += error * delta_time

			if (self.ITerm < -self.windup_guard):
				self.ITerm = -self.windup_guard
			elif (self.ITerm > self.windup_guard):
				self.ITerm = self.windup_guard

			self.DTerm = 0.0
			if delta_time > 0:
				self.DTerm = delta_error / delta_time

			# Remember last time and last error for next calculation
			self.last_time = self.current_time
			self.last_error = error

		self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

	def setKp(self, proportional_gain):
		"""Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
		self.Kp = proportional_gain

	def setKi(self, integral_gain):
		"""Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
		self.Ki = integral_gain

	def setKd(self, derivative_gain):
		"""Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
		self.Kd = derivative_gain

	def setWindup(self, windup):
		"""Integral windup, also known as integrator windup or reset windup,
		refers to the situation in a PID feedback controller where
		a large change in setpoint occurs (say a positive change)		a large change in setpoint occurs (say a positive change)
		and the integral terms accumulates a significant error
		during the rise (windup), thus overshooting and continuing
		to increase as this accumulated error is unwound
		(offset by errors in the other direction).
		The specific problem is the excess overshooting.
		"""
		self.windup_guard = windup

	def setSampleTime(self, sample_time):
		"""PID that should be updated at a regular interval.
		Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
		"""
		self.sample_time = sample_time
################################################################################

class GoalNav(object):
	def __init__(self):
		super().__init__()
		# constant height
		self.constant_height = PID_control("cool robot control", P=0.7, I=0.01, D=0.02)
		self.height = 0.45
		self.target_height = rospy.get_param("~target_height", 0.45)

		self.max_dis = 10  # meters
		self.laser_n = 4
		self.pos_n = 10
		self.frame = rospy.get_param("~frame", "map")
		self.action_scale = {'linear': rospy.get_param(
			'~linear_scale', 1.5), 'angular': rospy.get_param("~angular_scale", 0.8)}

		self.auto = 0
		self.goal = None
		self.pos_track = None
		self.laser_stack = None
		self.last_pos = None

		self.last_omega = 0
		self.omega_gamma = 0.25

		self.vel_ratio = 0

		# network
		obs_dim = 243
		action_dim = 2
		gpu = tf.config.experimental.list_physical_devices('GPU')
		tf.config.experimental.set_memory_growth(gpu[0], True)
		my_dir = os.path.abspath(os.path.dirname(__file__))
		model_path = os.path.join(my_dir, "../model/goal/policy")
		self.policy_network = tf.saved_model.load(model_path)

		# pub cmd
		self.pub_cmd = rospy.Publisher("cmd_out", Twist, queue_size=1)

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

	def constant_height_once(self):
		try:
			msg = rospy.wait_for_message(
				'/blimp/points', PointCloud2, timeout=5)
		except:
			print('fail to receive message')
		height = 0
		for p in point_cloud2.read_points(msg, field_names = ("x", "y", "z"), skip_nans=True):
			if (p[0]**2 + p[1]**2)**0.5 < 1.6 and (p[0]**2 + p[1]**2)**0.5 > 0.6:
				if p[2]<height: height=p[2]

		if height == 0: self.height = 1
		else : self.height = abs(height)
		# print("height= ", self.height)

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
		self.height = msg.pose.position.z
		# caculate angle diff
		new_pos = np.array(
			[msg.pose.position.x, msg.pose.position.y])
		diff = self.goal - new_pos
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
		self.last_pos = new_pos

	def cb_laser(self, msg):
		ranges = np.array(msg.ranges)
		ranges = np.clip(ranges, 0, self.max_dis)

		if self.laser_stack is None:
			self.laser_stack = np.tile(ranges, (self.laser_n, 1))
		else:
			self.laser_stack[:-1] = self.laser_stack[1:]
			self.laser_stack[-1] = ranges

	def inference(self, event):
		if self.goal is None:
			rospy.loginfo("self.goal is None")
			return

		if self.pos_track is None:
			rospy.loginfo("self.pos_track is None")
			return

		if self.laser_stack is None:
			rospy.loginfo("self.laser_stack is None")
			return

		if self.auto == 0:
			rospy.loginfo("not auto")
			return

		dis = np.linalg.norm(self.goal-self.last_pos)
		if dis < 1.5:
			rospy.loginfo("goal reached")
			# self.goal = None
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
		cmd.linear.x = -1 * action[0]*self.action_scale['linear']
		cmd.angular.z = -1 * self.last_omega * self.action_scale['angular']

		self.constant_height_once()
		# constant height (PID)
		self.constant_height.update(float(self.target_height-self.height))
		u = self.constant_height.output
		cmd.linear.z = u

		self.pub_cmd.publish(cmd)

if __name__ == "__main__":
	rospy.init_node("goal_nav_rl")
	goalNav = GoalNav()
	rospy.spin()
