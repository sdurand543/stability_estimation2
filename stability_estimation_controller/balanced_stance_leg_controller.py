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

_KP = [np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01])]
_KI = [np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01])]
_KD = [np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01]), np.array([0.01, 0.01, 0.01])]

class BalancedStanceLegController(leg_controller.LegController):
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
    self._last_leg_state = gait_generator.desired_leg_state
    self.desired_speed = np.array((desired_speed[0], desired_speed[1], 0))
    self._desired_height = desired_height

    # PID_CONTROLLER
    self.error_integral = defaultdict(float)
    self.error_prev = defaultdict(float)
    
    self._joint_angles = None
    self._phase_switch_foot_local_position = None
    self.reset(0)

  def reset(self, current_time: float) -> None:
    """Called during the start of a swing cycle.

    Args:
      current_time: The wall time in seconds.
    """
    del current_time
    self._last_leg_state = self._gait_generator.desired_leg_state
    self._phase_switch_foot_local_position = (
      self._robot.GetFootPositionsInBaseFrame())
    self._joint_angles = {}

  def update(self, current_time: float) -> None:
    """Called at each control step.

    Args:
      current_time: The wall time in seconds.
    """
    del current_time
    new_leg_state = self._gait_generator.desired_leg_state

    # Detects phase switch for each leg so we can remember the feet position at
    # the beginning of the swing phase.
    for leg_id, state in enumerate(new_leg_state):
      if (state == gait_generator_lib.LegState.SWING
          and state != self._last_leg_state[leg_id]):
        self._phase_switch_foot_local_position[leg_id] = (
          self._robot.GetFootPositionsInBaseFrame()[leg_id])
    self._last_leg_state = copy.deepcopy(new_leg_state)

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
    
    # Get all Foot Positions
    foot_positions = self._robot.GetFootPositionsInBaseFrame()
    
    # Get a List of STANCE Feet and Calculate Desired COM Positions
    desired_com_position_2d = np.array([0., 0.], dtype=np.float64)
    stance_legs = {} # Mapping[leg_id, foot_position_bf]
    for leg_id, leg_state in enumerate(self._gait_generator.leg_state):
      if leg_state in (gait_generator_lib.LegState.STANCE,
                       gait_generator_lib.LegState.LOSE_CONTACT): #include EARLY_CONTACT
        foot_position_bf = foot_positions[leg_id]
        stance_legs[leg_id] = foot_position_bf
        desired_com_position_2d += foot_position_bf[0:1]

    # Center of STANCE Leg Polygon
    desired_com_position_2d = desired_com_position_2d / len(stance_legs)

    # Delegate Desired Motion
    desired_com_position = np.array([desired_com_position_2d[0],
                                     desired_com_position_2d[1],
                                     self._desired_height],
                                    dtype=np.float64)

    for leg_id, foot_position in stance_legs.items():
      
      desired_foot_position_2d = foot_position[0:1] - desired_com_position_2d
      desired_foot_position = np.array(
        [desired_foot_position_2d[0], desired_foot_position_2d[1],
         -self._desired_height],
        dtype=np.float64)
      
      # Update PID State
      error = desired_foot_position - foot_position
      self.error_integral[leg_id] += error
      error_derivative = error - self.error_prev[leg_id]
      self.error_prev[leg_id] = error
      
      # Get Input Actuation
      u = (_KP[leg_id] * error
           + _KI[leg_id] * self.error_integral[leg_id]
           + _KD[leg_id] * error_derivative)

      # Get Total Position
      target_foot_position = foot_position + u

      # IK Solution to Target Position
      joint_ids, joint_angles = (
        self._robot.ComputeMotorAnglesFromFootLocalPosition(
            leg_id, foot_position))
      
      # Update the stored joint angles as needed.
      for joint_id, joint_angle in zip(joint_ids, joint_angles):
        self._joint_angles[joint_id] = (joint_angle, leg_id)

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

    print("action:", action)
    return action, None
