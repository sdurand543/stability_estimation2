# Lint as: python3

"""The balanced stance leg controller."""

from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import copy
import math

import numpy as np
from typing import Any, Mapping, Sequence, Tuple
from collections import defaultdict

try:
  from mpc_controller import gait_generator as gait_generator_lib
  from mpc_controller import leg_controller
except:  #pylint: disable=W0702
  print("You need to install motion_imitation")
  print("Either run python3 setup.py install --user in this repo")
  print("or use pip3 install motion_imitation --user")
  sys.exit()


# PID FOR WEIGHT DISTRIBUTION
_KZ = 3

_RELATIVE_KP = [np.array([0.02, 0.25, 0.3]), np.array([0.02, 0.25, 0.3]), np.array([0.02, 0.25, 0.3]), np.array([0.02, 0.25, 0.3])]
_RELATIVE_KI = [np.array([0.01, 0.05, 0.01]), np.array([0.01, 0.05, 0.01]), np.array([0.01, 0.05, 0.01]), np.array([0.01, 0.05, 0.01])]
_RELATIVE_KD = [np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01])]

# PID FOR RETAINING POSE
_ABSOLUTE_KP = [np.array([0.05, 0.2, 0.7]), np.array([0.05, 0.2, 0.7]), np.array([0.05, 0.2, 0.7]), np.array([0.05, 0.2, 0.7])]
_ABSOLUTE_KI = [np.array([0.05, 0.9, 0.3]), np.array([0.05, 0.9, 0.3]), np.array([0.05, 0.9, 0.3]), np.array([0.05, 0.9, 0.3])]
_ABSOLUTE_KD = [np.array([0.05, 0.05, 0.2]), np.array([0.05, 0.05, 0.2]), np.array([0.05, 0.05, 0.2]), np.array([0.05, 0.05, 0.2])]


class BalancedStanceLegController(leg_controller.LegController):
  """Controls the stance leg position using a PID controller and joint angle model prediction
  The procedure is as follows:
  1.) Estimate the local positions of all legs using the joint angles on the robot.
  2.) Find the mid-point of the triangle
  3.) Estimate the relative_errorrror of the center of mass (implicitly at the origin) from the mid-point of the triangle.
  4.) Apply inverse kinematics to the legs to try and move them such that the center of the mass is centered.

  Additionally account for:
  - gait_generation: only participating 'STANCE' legs will be considered as part of the static walking gait polygon
  - desired_speed: The desired speed will offset our desired mid-point such that the center of mass tilts in the direction of our desired motion
      - (but no more than within the static walking gait 'triangle' bounds)
  - desired_height: To Set The Height of The Legs
  """
  def __init__(
      self,
      robot: Any,
      gait_generator: Any,
      state_estimator: Any,
      desired_speed: Tuple[float, float],
      desired_height: float,
  ):
    """Initializes the class.

    Args:
      robot: A robot instance.
      gait_generator: Generates the leg-state pattern.
      state_estimator: Estiamtes the CoM speeds.
      desired_speed: Behavior parameters. X-Y speed.
      desired_height: Desired standing height.
    """
    self._robot = robot
    self._state_estimator = state_estimator    
    self._gait_generator = gait_generator    
    self._desired_height = desired_height

    # PID_CONTROLLER
    self.relative_error_integral = defaultdict(float)
    self.relative_error_prev = defaultdict(float)
    self.absolute_error_integral = defaultdict(float)
    self.absolute_error_prev = defaultdict(float)

    # Bound in Box
    # front_right, front_left, back_right, back_left
    max_dist_offset = 0.15
    self._initial_foot_positions = robot.GetFootPositionsInBaseFrame()

    # front_right
    self._min_leg0_pos = self._initial_foot_positions[0] + np.array([max_dist_offset, -max_dist_offset, -max_dist_offset])
    self._max_leg0_pos = self._initial_foot_positions[0] + np.array([-max_dist_offset, max_dist_offset, max_dist_offset])

    # front_left
    self._min_leg1_pos = self._initial_foot_positions[1] + np.array([-max_dist_offset, -max_dist_offset, -max_dist_offset])
    self._max_leg1_pos = self._initial_foot_positions[1] + np.array([max_dist_offset, max_dist_offset, max_dist_offset])

    # back_right
    self._min_leg2_pos = self._initial_foot_positions[2] + np.array([max_dist_offset, max_dist_offset, -max_dist_offset])
    self._max_leg2_pos = self._initial_foot_positions[2] + np.array([-max_dist_offset, -max_dist_offset, max_dist_offset])

    # back_left
    self._min_leg3_pos = self._initial_foot_positions[3] + np.array([-max_dist_offset, max_dist_offset, -max_dist_offset])
    self._max_leg3_pos = self._initial_foot_positions[3] + np.array([max_dist_offset, -max_dist_offset, max_dist_offset])

    self._joint_angles = {}

    self.reset(0)

  def reset(self, current_time: float) -> None:
    """Called during the start of a swing cycle.

    Args:
      current_time: The wall time in seconds.
    """
    del current_time
  
  def update(self, current_time: float) -> None:
    """Called at each control step.

    Args:
      current_time: The wall time in seconds.
    """
    del current_time

  """Controls the stance leg position using a PID controller and joint angle model prediction
  The procedure is as follows:
  1.) Estimate the local positions of all legs using the joint angles on the robot.
  2.) Find the mid-point of the triangle
  3.) Estimate the error of the center of mass (implicitly at the origin) from the mid-point of the triangle.
  4.) Apply inverse kinematics to the legs to try and move them such that the center of the mass is centered.

  Additionally account for:
  - gait_generation: only participating 'STANCE' legs will be considered as part of the static walking gait polygon
  - desired_speed: The desired speed will offset our desired mid-point such that the center of mass tilts in the direction of our desired motion
      - (but no more than within the static walking gait 'triangle' bounds)
  - desired_height: To Set The Height of The Legs
  """


  def get_action(self) -> Mapping[Any, Any]:

    # ESTIMATED WEIGHTS
    body_mass = 7
    leg_mass = 0.3
    total_mass = body_mass + 4 * leg_mass
    
    # Get all Foot Positions
    hip_positions = self._robot.GetHipPositionsInBaseFrame()
    foot_positions = self._robot.GetFootPositionsInBaseFrame()

    # Get a List of STANCE Feet and Calculate Desired COM Position
    com_position_2d = np.array([0., 0.], dtype=np.float64) # this is also 'the position of the robot weighted by body_mass'
    desired_com_position_2d = np.array([0., 0.], dtype=np.float64)
    stance_legs = {} # Mapping[leg_id, (foot_position_bf, hip_offset_bf)]
    for leg_id, leg_state in enumerate(self._gait_generator.leg_state):
      foot_position_bf = foot_positions[leg_id]
      hip_offset = hip_positions[leg_id]
      com_position_2d += foot_position_bf[0:2] * leg_mass
      if leg_state is gait_generator_lib.LegState.STANCE:
        stance_legs[leg_id] = ( foot_position_bf, hip_offset )
        desired_com_position_2d += foot_position_bf[0:2]

    com_position_2d /= total_mass
    #com_position_2d = np.array([0, 0], dtype=np.float64)
    desired_com_position_2d /= len(stance_legs)

    # For Each Leg
    for leg_id, foot_info in stance_legs.items():
      foot_position = foot_info[0]
      hip_offset = foot_info[1]
      
      relative_error_2d = com_position_2d - desired_com_position_2d
      relative_error = np.array([relative_error_2d[0], relative_error_2d[1], -_KZ * np.dot(relative_error_2d, foot_position[0:2])])
      absolute_error = self._initial_foot_positions[leg_id] - foot_position

      # Run PID on the Error
      # Update PID State
      alpha = 0.3
      self.relative_error_integral[leg_id] = alpha * self.relative_error_integral[leg_id] + relative_error
      relative_error_derivative = relative_error - self.relative_error_prev[leg_id]
      self.relative_error_prev[leg_id] = relative_error
      self.absolute_error_integral[leg_id] = alpha * self.absolute_error_integral[leg_id] + absolute_error
      absolute_error_derivative = absolute_error - self.absolute_error_prev[leg_id]
      self.absolute_error_prev[leg_id] = absolute_error

      #print("Relative_Error for leg", "Leg:", leg_id, "Relative_Error:", relative_error)

      u = (_RELATIVE_KP[leg_id] * relative_error
           + _RELATIVE_KI[leg_id] * self.relative_error_integral[leg_id]
           + _RELATIVE_KD[leg_id] * relative_error_derivative
           + _ABSOLUTE_KP[leg_id] * absolute_error
           + _ABSOLUTE_KI[leg_id] * self.absolute_error_integral[leg_id]
           + _ABSOLUTE_KD[leg_id] * absolute_error_derivative)


      #print("U:", u)

      # Get the Next Position by Adding Error
      foot_position_next_3d = foot_position + u

      # Bound in Box
      # front_right, front_left, back_right, back_left
      '''
      if leg_id == 0: # front_right
        foot_position_next_3d = np.clip(foot_position_next_3d, self._min_leg0_pos, self._max_leg0_pos)
      if leg_id == 1: # front_left
        foot_position_next_3d = np.clip(foot_position_next_3d, self._min_leg1_pos, self._max_leg1_pos)
      if leg_id == 2: # back_right
        foot_position_next_3d = np.clip(foot_position_next_3d, self._min_leg1_pos, self._max_leg1_pos)
      if leg_id == 3: # back_left
        foot_position_next_3d = np.clip(foot_position_next_3d, self._min_leg1_pos, self._max_leg1_pos)
      '''
      
      # Apply IK
      joint_ids, joint_angles = (
        self._robot.ComputeMotorAnglesFromFootLocalPosition(
            leg_id, foot_position_next_3d))

      # Update the stored joint angles as needed.
      for joint_id, joint_angle in zip(joint_ids, joint_angles):
        self._joint_angles[joint_id] = (joint_angle, leg_id)

      # Return Action
      action = {}
      kps = self._robot.GetMotorPositionGains()
      kds = self._robot.GetMotorVelocityGains()
      for joint_id, joint_angle_leg_id in self._joint_angles.items():
        leg_id = joint_angle_leg_id[1]
        if self._gait_generator.desired_leg_state[
            leg_id] == gait_generator_lib.LegState.STANCE:
          # This is a hybrid action for PD control.
          action[joint_id] = (joint_angle_leg_id[0], kps[joint_id], 0,
                              kds[joint_id], 0)

    return action, None
