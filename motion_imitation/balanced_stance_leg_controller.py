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

_KP = [np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])]
_KI = [np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])]
_KD = [np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])]

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
    self._desired_com = np.array([0, 0, 0], dtype=np.float64)
    foot_positions = self._robot.GetFootPositionsInBaseFrame()
  
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
    foot_forces = self._robot.GetFootForce()
    hip_positions = self._robot.GetHipPositionsInBaseFrame()
    foot_positions = self._robot.GetFootPositionsInBaseFrame()

    # Get a List of STANCE Feet and Calculate Desired COM Position
    desired_com_position_2d = np.array([0., 0.], dtype=np.float64)
    stance_legs = {} # Mapping[leg_id, (foot_position_bf, hip_offset_bf)]
    total_force = 0
    for leg_id, leg_state in enumerate(self._gait_generator.leg_state):
      if leg_state is gait_generator_lib.LegState.STANCE:
        #print("STANCE for leg:", leg_id, leg_state)
        hip_offset = hip_positions[leg_id]
        foot_position_bf = foot_positions[leg_id]
        stance_legs[leg_id] = ( foot_position_bf, hip_offset )
        #total_force += foot_forces[str((leg_id + 1) * 5)]
        desired_com_position_2d += foot_position_bf[0:2] #* foot_forces[str((leg_id + 1) * 5)]

    desired_com_position_2d /= len(stance_legs)

    # Calculate the Error
    error = - desired_com_position_2d

    # For Each Leg
    for leg_id, foot_info in stance_legs.items():
      foot_position = foot_info[0]
      hip_offset = foot_info[1]

      # Run PID on the Error
      # Update PID State
      alpha = 0.3
      self.error_integral[leg_id] = alpha * self.error_integral[leg_id] + error
      error_derivative = error - self.error_prev[leg_id]
      self.error_prev[leg_id] = error
      #print("Error for leg", "Leg:", leg_id, "Error:", error)

      u = (_KP[leg_id] * error
           + _KI[leg_id] * self.error_integral[leg_id]
           + _KD[leg_id] * error_derivative)

      print("U:", u)

      # Get the Next Position by Adding Error
      foot_position_next_2d = foot_position[0:2] + u

      # Append Height
      foot_position_next_3d = np.array([foot_position_next_2d[0], foot_position_next_2d[1], -self._desired_height])

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


  def get_action2(self) -> Mapping[Any, Any]:

    # Get all Foot Positions
    foot_forces = self._robot.GetFootForce()
    hip_positions = self._robot.GetHipPositionsInBaseFrame()
    foot_positions = self._robot.GetFootPositionsInBaseFrame()
    
    # Get a List of STANCE Feet and Calculate Desired COM Positions
    desired_com_position_2d = np.array([0., 0.], dtype=np.float64)
    stance_legs = {} # Mapping[leg_id, (foot_position_bf, hip_offset_bf)]
    total_force = 0
    for leg_id, leg_state in enumerate(self._gait_generator.leg_state):
      if leg_state is gait_generator_lib.LegState.STANCE:
        print("STANCE for leg:", leg_id, leg_state)
        hip_offset = hip_positions[leg_id]
        foot_position_bf = foot_positions[leg_id]
        stance_legs[leg_id] = ( foot_position_bf, hip_offset )
        #total_force += foot_forces[str((leg_id + 1) * 5)]
        desired_com_position_2d += foot_position_bf[0:2] #* foot_forces[str((leg_id + 1) * 5)]
    
    # Center of STANCE Leg Polygon
    desired_com_position_2d = desired_com_position_2d / len(stance_legs) #/ (total_force + 0.001)

    desired_com_position_2d += np.array([0.0, 0.0])
    desired_com_position_2d = np.array([-0.02, 0.022])

    # Delegate Desired Motion
    desired_com_position = np.array([desired_com_position_2d[0],
                                     desired_com_position_2d[1],
                                     0],
                                    dtype=np.float64)

    for leg_id, foot_info in stance_legs.items():
      foot_position = foot_info[0]
      hip_offset = foot_info[1]
      boxed_foot_position = foot_position

      # Taking into Account Hips (Getting Minimum Breadth)

      '''
      min_hips = 0.5
      max_hips = 3
      if leg_id == 0: # front right
        boxed_foot_position[0] = max(min_hips * hip_offset[0], foot_position[0])
        boxed_foot_position[1] = min(min_hips * hip_offset[1], foot_position[1])
        boxed_foot_position[0] = min(max_hips * hip_offset[0], foot_position[0])
        boxed_foot_position[1] = max(max_hips * hip_offset[1], foot_position[1])
      if leg_id == 1: # front left
        boxed_foot_position[0] = max(min_hips * hip_offset[0], foot_position[0])
        boxed_foot_position[1] = max(min_hips * hip_offset[1], foot_position[1])
        boxed_foot_position[0] = min(max_hips * hip_offset[0], foot_position[0])
        boxed_foot_position[1] = min(max_hips * hip_offset[1], foot_position[1])
      if leg_id == 2: # back right
        boxed_foot_position[0] = min(min_hips * hip_offset[0], foot_position[0])
        boxed_foot_position[1] = min(min_hips * hip_offset[1], foot_position[1])
        boxed_foot_position[0] = max(max_hips * hip_offset[0], foot_position[0])
        boxed_foot_position[1] = max(max_hips * hip_offset[1], foot_position[1])
      if leg_id == 3: # back left
        boxed_foot_position[0] = min(min_hips * hip_offset[0], foot_position[0])
        boxed_foot_position[1] = max(min_hips * hip_offset[1], foot_position[1])
        boxed_foot_position[0] = max(max_hips * hip_offset[0], foot_position[0])
        boxed_foot_position[1] = min(max_hips * hip_offset[1], foot_position[1])
      '''


      print("Desired Center of Mass", "Leg:", leg_id, "COM:", desired_com_position)
      #exit()
      print("Position of for leg", "Leg:", leg_id, "Position:", foot_position)

      
      desired_foot_position_2d = boxed_foot_position[0:2] - desired_com_position_2d
      desired_foot_position = np.array(
        [desired_foot_position_2d[0], desired_foot_position_2d[1],
         -self._desired_height],
        dtype=np.float64)
      
      print("Desired Position for leg", "Leg:", leg_id, "Desired Position:", desired_foot_position)

      # Update PID State
      alpha = 0.3
      error = desired_foot_position - foot_position
      self.error_integral[leg_id] = alpha * self.error_integral[leg_id] + error
      error_derivative = error - self.error_prev[leg_id]
      self.error_prev[leg_id] = error

      print("Error for leg", "Leg:", leg_id, "Error:", error)
      
      # Get Input Actuation
      u = (_KP[leg_id] * error
           + _KI[leg_id] * self.error_integral[leg_id]
           + _KD[leg_id] * error_derivative)
      
      print("U:", u)

      # Get Total Position
      target_foot_position = foot_position + [-u[0], -u[1], u[2]]

      print("Target foot position for leg", "Leg:", leg_id, "Target Foot Pos:", target_foot_position)


      # IK Solution to Target Position
      joint_ids, joint_angles = (
        self._robot.ComputeMotorAnglesFromFootLocalPosition(
            leg_id, target_foot_position))
      
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
    #exit()
    return action, None
