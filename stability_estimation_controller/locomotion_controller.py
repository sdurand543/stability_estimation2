"""A model based controller framework."""

from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import numpy as np
import time
from typing import Any, Callable

class LocomotionController(object):
  """Generates the quadruped locomotion.

  The actual effect of this controller depends on the composition of each
  individual subcomponent.

  """
  def __init__(
      self,
      robot: Any,
      state_estimator,
      stability_estimation_leg_controller,
      stable_stance_leg_controller,
      clock,
  ):
    """Initializes the class.

    Args:
      robot: A robot instance.
      gait_generator: Generates the leg swing/stance pattern.
      state_estimator: Estimates the state of the robot (e.g. center of mass
        position or velocity that may not be observable from sensors).
      stability_estimation_leg_controller: Generates motor actions for swing legs.
      stable_stance_leg_controller: Generates motor actions for stance legs.
      clock: A real or fake clock source.
    """
    self._robot = robot
    self._clock = clock
    self._reset_time = self._clock()
    self._time_since_reset = 0
    self._state_estimator = state_estimator
    self._stability_estimation_leg_controller = stability_estimation_leg_controller
    self._stable_stance_leg_controller = stable_stance_leg_controller

  @property
  def stability_estimation_leg_controller(self):
    return self._stability_estimation_leg_controller

  @property
  def stable_stance_leg_controller(self):
    return self._stable_stance_leg_controller

  @property
  def state_estimator(self):
    return self._state_estimator

  def reset(self):
    self._reset_time = self._clock()
    self._time_since_reset = 0
    self._state_estimator.reset(self._time_since_reset)
    self._stability_estimation_leg_controller.reset(self._time_since_reset)
    self._stable_stance_leg_controller.reset(self._time_since_reset)

  def update(self):
    self._time_since_reset = self._clock() - self._reset_time
    self._state_estimator.update(self._time_since_reset)
    self._stability_estimation_leg_controller.update(self._time_since_reset)
    self._stable_stance_leg_controller.update(self._time_since_reset)

  def get_action(self):
    """Returns the control ouputs (e.g. positions/torques) for all motors."""
    stable_placement_action = self._stability_estimation_leg_controller.get_action()
    static_stance_action, qp_sol = self._stable_stance_leg_controller.get_action()
    action = []
    for joint_id in range(self._robot.num_motors):
      if joint_id in stable_placement_action:
        action.extend(stable_placement_action[joint_id])
      else:
        assert joint_id in stable_stance_action
        action.extend(stable_stance_action[joint_id])
    action = np.array(action, dtype=np.float32)

    return action, dict(qp_sol=qp_sol)
