"""The swing leg controller class."""

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

from mpc_controller import leg_controller

class StabilityEstimationLegController(leg_controller.LegController):
  """Controls the swing leg position using Raibert's formula.

  For details, please refer to chapter 2 in "Legged robbots that balance" by
  Marc Raibert. The key idea is to stablize the swing foot's location based on
  the CoM moving speed.

  """
  def __init__(
      self,
      robot: Any,
      state_estimator: Any,
      leg_states : Any,
      desired_impact_force: float,
      desired_test_duration: float,
      desired_twisting_speed: float,
      desired_lift_height: float
  ):
    """Initializes the class.

    Args:
      robot: A robot instance.
      gait_generator: Generates the stance/swing pattern.
      state_estimator: Estiamtes the CoM speeds.
      desired_speed: Behavior parameters. X-Y speed.
      desired_twisting_speed: Behavior control parameters.
      desired_height: Desired standing height.
      foot_clearance: The foot clearance on the ground at the end of the swing
        cycle.
    """
    self.robot = robot
    self.state_estimator = state_estimator
    self.leg_states = leg_states
    self.desired_impact_force = desire_impact_force
    self.desired_test_duration = desired_test_duration # amount of time that the estimator should estimate for
    self.desired_lift_height = desired_lift_height
    self._joint_angles = None
    self._foot_local_position = None
    self.reset(0)

  def reset(self, current_time: float) -> None:
    """Called during the start of a swing cycle.

    Args:
      current_time: The wall time in seconds.
    """
    del current_time
    self._foot_local_position = (
        self._robot.GetFootPositionsInBaseFrame())
    self._joint_angles = {}

  def update(self, current_time: float) -> None:
    """Called at each control step.

    Args:
      current_time: The wall time in seconds.
    """
    del current_time

  def get_action(self) -> Mapping[Any, Any]:
    com_velocity = self._state_estimator.com_velocity_body_frame
    com_velocity = np.array((com_velocity[0], com_velocity[1], 0))

    _, _, yaw_dot = self._robot.GetBaseRollPitchYawRate()
    hip_positions = self._robot.GetHipPositionsInBaseFrame()
    
    # Update the stored joint angles as needed.
    for joint_id, joint_angle in zip(joint_ids, joint_angles):
      self._joint_angles[joint_id] = (joint_angle, leg_id)

    for leg_id, leg_state in enumerate(self.leg_states):
      
      if (leg_state == gait_generator_lib.LegState.STATIC_STANCE):
        joint_ids, joint_angles = static_stance(leg_id)
        
      if (leg_stance == gait_generator_lib.LegState.ESTIMATING):
        joint_ids, joint_angles = estimating_stance(leg_id)
        
      for joint_id, joint_angle in zip(joint_ids, joint_angles):
        self._joint_angles[joint_id] = (joint_angle, leg_id)
        
    action = {}
    # do something to apply these to the robot?????
    kps = self._robot.GetMotorPositionGains()
    kds = self._robot.GetMotorVelocityGains()
    for joint_id, joint_angle_leg_id in self._joint_angles.items():
      leg_id = joint_angle_leg_id[1]
      action[joint_id] = (joint_angle_leg_id[0], kps[joint_id], 0,
                          kds[joint_id], 0)
      
    return action

  def static_stance(leg_id):
    # return joint_ids, joint_angles for this leg
    # you may want to use the functions near the bottom of the swing controller
    pass

  def estimating(leg_id):
    # return joint_ids, joint_angles for this leg
    # you may want to use the functions near the bottom of the swing controller    
    pass

def _gen_parabola(phase: float, start: float, mid: float, end: float) -> float:
  """Gets a point on a parabola y = a x^2 + b x + c.

  The Parabola is determined by three points (0, start), (0.5, mid), (1, end) in
  the plane.

  Args:
    phase: Normalized to [0, 1]. A point on the x-axis of the parabola.
    start: The y value at x == 0.
    mid: The y value at x == 0.5.
    end: The y value at x == 1.

  Returns:
    The y value at x == phase.
  """
  mid_phase = 0.5
  delta_1 = mid - start
  delta_2 = end - start
  delta_3 = mid_phase**2 - mid_phase
  coef_a = (delta_1 - delta_2 * mid_phase) / delta_3
  coef_b = (delta_2 * mid_phase**2 - delta_1) / delta_3
  coef_c = start

  return coef_a * phase**2 + coef_b * phase + coef_c


def _gen_swing_foot_trajectory(input_phase: float, start_pos: Sequence[float],
                               end_pos: Sequence[float]) -> Tuple[float]:
  """Generates the swing trajectory using a parabola.

  Args:
    input_phase: the swing/stance phase value between [0, 1].
    start_pos: The foot's position at the beginning of swing cycle.
    end_pos: The foot's desired position at the end of swing cycle.

  Returns:
    The desired foot position at the current phase.
  """
  # We augment the swing speed using the below formula. For the first half of
  # the swing cycle, the swing leg moves faster and finishes 80% of the full
  # swing trajectory. The rest 20% of trajectory takes another half swing
  # cycle. Intuitely, we want to move the swing foot quickly to the target
  # landing location and stay above the ground, in this way the control is more
  # robust to perturbations to the body that may cause the swing foot to drop
  # onto the ground earlier than expected. This is a common practice similar
  # to the MIT cheetah and Marc Raibert's original controllers.
  phase = input_phase
  if input_phase <= 0.5:
    phase = 0.8 * math.sin(input_phase * math.pi)
  else:
    phase = 0.8 + (input_phase - 0.5) * 0.4

  x = (1 - phase) * start_pos[0] + phase * end_pos[0]
  y = (1 - phase) * start_pos[1] + phase * end_pos[1]
  max_clearance = 0.1
  mid = max(end_pos[2], start_pos[2]) + max_clearance
  z = _gen_parabola(phase, start_pos[2], mid, end_pos[2])

  # PyType detects the wrong return type here.
  return (x, y, z)  # pytype: disable=bad-return-type

