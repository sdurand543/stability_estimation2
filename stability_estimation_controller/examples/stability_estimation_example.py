"""Example of whole body controller on A1 robot."""
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from absl import app
from absl import flags
from absl import logging
from datetime import datetime
import numpy as np
import os
import scipy.interpolate
import time
import math

import pybullet_data
from pybullet_utils import bullet_client
import pybullet# pytype:disable=import-error

from mpc_controller import three_leg_balance_gait_generator as three_leg_balance_openloop_gait_generator
from mpc_controller import three_leg_balance_swing_up as three_leg_balance_raibert_swing_leg_controller
from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
#from mpc_controller import torque_stance_leg_controller
#import mpc_osqp
from mpc_controller import torque_stance_leg_controller_quadprog as torque_stance_leg_controller
from stability_estimation_controller import balanced_stance_leg_controller

from motion_imitation.robots import a1
from motion_imitation.robots import robot_config
from motion_imitation.robots.gamepad import gamepad_reader

import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime

flags.DEFINE_string("logdir", None, "where to log trajectories.")
flags.DEFINE_bool("use_gamepad", False,
                  "whether to use gamepad to provide control input.")
flags.DEFINE_bool("use_real_robot", False,
                  "whether to use real robot or simulation")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_float("max_time_secs", 30., "maximum time to run the robot.")
FLAGS = flags.FLAGS

_NUM_SIMULATION_ITERATION_STEPS = 300
_MAX_TIME_SECONDS = 30.

_STANCE_DURATION_SECONDS = [
    0.4
] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).

# Standing
# _DUTY_FACTOR = [1.] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0., 0., 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
# )

# Tripod
# _DUTY_FACTOR = [.8] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0.25, 0.5, 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.SWING,
# )

# Trotting
_DUTY_FACTOR = [1] * 4
_INIT_PHASE_FULL_CYCLE = [0, 0, 0, 0]

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
)


def _generate_example_linear_angular_speed(t):
  """Creates an example speed profile based on time for demo purpose."""
  vx = 0
  vy = 0
  wz = 0

  time_points = (0, 30, 30, 30, 30, 30, 30)
  speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz),
                  (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="previous",
                                     fill_value="extrapolate",
                                     axis=0)(t)

  return speed[0:3], speed[3], False

def _setup_three_leg_controller(robot):
    """Demonstrates how to create a locomotion controller."""
    desired_speed = (0, 0)
    desired_twisting_speed = 0

    gait_generator = three_leg_balance_openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor=_DUTY_FACTOR,
      initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
      initial_leg_state=_INIT_LEG_STATE,
      lift_time = 0.5)

    state_estimator = com_velocity_estimator.COMVelocityEstimator(
      robot, window_size=window_size)
    
    window_size = 20 if not FLAGS.use_real_robot else 1

    controller = balanced_stance_leg_controller.BalancedStanceLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_height=robot.MPC_BODY_HEIGHT
    )
    
    '''
    torque_stance_leg_controller.TorqueStanceLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_body_height=robot.MPC_BODY_HEIGHT
        # ,qp_solver = mpc_osqp.QPOASES #or mpc_osqp.OSQP
    )
    '''

    '''
    controller = locomotion_controller.LocomotionController(
        robot=robot,
        gait_generator=gait_generator,
        state_estimator=state_estimator,
        swing_leg_controller=sw_controller,
        stance_leg_controller=st_controller,
        clock=robot.GetTimeSinceReset)
    '''
    
    return controller



def _setup_gait_controller(robot):
  """Demonstrates how to create a locomotion controller."""
  desired_speed = (0, 0)
  desired_twisting_speed = 0

  gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor=_DUTY_FACTOR,
      initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
      initial_leg_state=_INIT_LEG_STATE)
  time.sleep(0.2)
  window_size = 20 if not FLAGS.use_real_robot else 1
  state_estimator = com_velocity_estimator.COMVelocityEstimator(
      robot, window_size=window_size)
  sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_height=robot.MPC_BODY_HEIGHT,
      foot_clearance=0.01)

  st_controller = torque_stance_leg_controller.TorqueStanceLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_body_height=robot.MPC_BODY_HEIGHT
      #,qp_solver = mpc_osqp.QPOASES #or mpc_osqp.OSQP
      )

  controller = locomotion_controller.LocomotionController(
      robot=robot,
      gait_generator=gait_generator,
      state_estimator=state_estimator,
      swing_leg_controller=sw_controller,
      stance_leg_controller=st_controller,
      clock=robot.GetTimeSinceReset)
  return controller


def _update_controller_params(controller, lin_speed, ang_speed):
  controller.swing_leg_controller.desired_speed = lin_speed
  controller.swing_leg_controller.desired_twisting_speed = ang_speed
  controller.stance_leg_controller.desired_speed = lin_speed
  controller.stance_leg_controller.desired_twisting_speed = ang_speed

def main(argv):
  """Runs the locomotion controller example."""
  del argv # unused

  # Construct simulator
  if FLAGS.show_gui and not FLAGS.use_real_robot:
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
  else:
    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
  p.setPhysicsEngineParameter(numSolverIterations=30)
  p.setTimeStep(0.001)
  p.setGravity(0, 0, -9.8)
  p.setPhysicsEngineParameter(enableConeFriction=0)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  p.loadURDF("plane100.urdf")

  #Terrain Environment
  boxHalfLength = 1
  boxHalfWidth = 1
  boxHalfHeight = 0.1
  sh_colBox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
  block2=p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [1.3,0.6,-0.05],baseOrientation=[0.0,0.0,0.0,1])
  #p.changeDynamics(block2, -1, linearDamping=0,angularDamping=0, rollingFriction = 0, spinningFriction = 0, lateralFriction = 0.4)

  # Construct robot class:
  if FLAGS.use_real_robot:
    from motion_imitation.robots import a1_robot
    robot = a1_robot.A1Robot(
        pybullet_client=p,
        motor_control_mode=robot_config.MotorControlMode.HYBRID,
        enable_action_interpolation=False,
        time_step=0.002,
        action_repeat=1)
  else:
    robot = a1.A1(p,
                  motor_control_mode=robot_config.MotorControlMode.HYBRID,
                  enable_action_interpolation=False,
                  reset_time=2,
                  time_step=0.002,
                  action_repeat=1)

  controller = _setup_three_leg_controller(robot)

  controller.reset()
  if FLAGS.use_gamepad:
    gamepad = gamepad_reader.Gamepad()
    command_function = gamepad.get_command
  else:
    command_function = _generate_example_linear_angular_speed

  if FLAGS.logdir:
    logdir = os.path.join(FLAGS.logdir,
                          datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(logdir)

  start_time = robot.GetTimeSinceReset()
  current_time = start_time
  com_vels, imu_rates, actions = [], [], []
  
  action_start = robot.GetMotorAngles()
  action_initial = np.array([0., 0.9, -2 * 0.9] * 4)  # intial height set by low-level controller - ~0.24m
  #action_final = np.array([0., 1., -2*1.] + [0., 0.9, -2 * 0.9] * 3)
  action_final = np.array([0.3, 0.9, -2 * 0.9] + [0.3, 0.9, -2 * 0.9] + [0.3, 0.6, -2 * 0.9] + [0.3, 0.6, -2 * 0.9])

  cumulative_foot_forces = []
  timesteps = 0

  num_steps_to_reset = 5000
  for cur_step_ in range(num_steps_to_reset):
      balance_action = controller.get_action()
      action = action_initial * (
        num_steps_to_reset - cur_step_) / num_steps_to_reset + action_final * cur_step_ / num_steps_to_reset
      robot.Step(action, robot_config.MotorControlMode.POSITION)

      time_dict = {'current_time': timesteps}
      foot_forces = robot.GetFootForce()
      print("Foot Forces:", foot_forces)
      df_dict = dict(time_dict)
      df_dict.update(foot_forces)
      cumulative_foot_forces.append(df_dict)
      timesteps += 1
      controller.update()

      time.sleep(robot.time_step)

  num_steps_to_reset = 5000

  action_initial = np.copy(action_final) #= np.array([-0.15, 0.9, -2 * 0.9] + [0.15, 0.9, -2 * 0.9] + [-0.15, 0.7, -2 * 0.9] + [0.15, 0.7, -2 * 0.9])#np.array([0., 1.2, -2 * 1.2] + [0.15, 0.9, -2 * 0.9] + [0.0, 1, -2 * 0.9] + [0.15, 1, -2 * 0.9])
  desired_foot_position = [0.181,0.15,-0.03]
  action_final = np.array([0.3, 0.9, -2 * 0.9] + robot.ComputeMotorAnglesFromFootLocalPosition(1, desired_foot_position)[1] + [0.3, 0.65, -2 * 0.9] + [0.3, 0.65, -2 * 0.9])
  for cur_step_ in range(num_steps_to_reset):
      action = action_initial * (
              num_steps_to_reset - cur_step_) / num_steps_to_reset + action_final * cur_step_ / num_steps_to_reset
      robot.Step(action, robot_config.MotorControlMode.POSITION)

      time_dict = {'current_time': timesteps}
      foot_forces = robot.GetFootForce()
      print("Foot Forces:", foot_forces)
      df_dict = dict(time_dict)
      df_dict.update(foot_forces)
      cumulative_foot_forces.append(df_dict)
      timesteps += 1
      controller.update()

      time.sleep(robot.time_step)

  num_steps_to_reset = 10000

  action_initial = np.copy(action_final)
  desired_foot_position = [0.35,0.15,0]
  action_final = np.array([0.3, 0.9, -2 * 0.9] + robot.ComputeMotorAnglesFromFootLocalPosition(1, desired_foot_position)[1] + [0.3, 0.65, -2 * 0.9] + [0.3, 0.65, -2 * 0.9])

  for cur_step_ in range(num_steps_to_reset):
      action = action_initial * (
              num_steps_to_reset - cur_step_) / num_steps_to_reset + action_final * cur_step_ / num_steps_to_reset
      robot.Step(action, robot_config.MotorControlMode.POSITION)

      time_dict = {'current_time': timesteps}
      foot_forces = robot.GetFootForce()
      print("Foot Forces:", foot_forces)
      df_dict = dict(time_dict)
      df_dict.update(foot_forces)
      cumulative_foot_forces.append(df_dict)
      timesteps += 1
      controller.update()

      time.sleep(robot.time_step)

  num_steps_to_reset = 10000

  action_initial = np.copy(action_final)
  desired_foot_position = [0.35,0.15,-0.3]
  action_final = np.array([0.3, 0.9, -2 * 0.9] + robot.ComputeMotorAnglesFromFootLocalPosition(1, desired_foot_position)[1] + [0.3, 0.65, -2 * 0.9] + [0.3, 0.65, -2 * 0.9])

  for cur_step_ in range(num_steps_to_reset):
      action = action_initial * (
              num_steps_to_reset - cur_step_) / num_steps_to_reset + action_final * cur_step_ / num_steps_to_reset
      robot.Step(action, robot_config.MotorControlMode.POSITION)

      time_dict = {'current_time': timesteps}
      foot_forces = robot.GetFootForce()
      print("Foot Forces:", foot_forces)
      df_dict = dict(time_dict)
      df_dict.update(foot_forces)
      cumulative_foot_forces.append(df_dict)
      timesteps += 1
      controller.update()

      trigger_foot_force = foot_forces['10']
      if trigger_foot_force > 15:
          action_initial = robot.GetMotorAngles()
          action_initial = action
          print('contact')
          controller.update()
          break

      time.sleep(robot.time_step)

  num_steps_to_reset = 1000

  desired_foot_position = robot.GetFootPositionsInBaseFrame()[1]
  desired_foot_position[2] = desired_foot_position[2] + 0.1
  action_final = np.array([0.3, 0.9, -2 * 0.9] + robot.ComputeMotorAnglesFromFootLocalPosition(1, desired_foot_position)[1] + [0.3, 0.65, -2 * 0.9] + [0.3, 0.65, -2 * 0.9])

  for cur_step_ in range(num_steps_to_reset):
      action = action_initial * (
              num_steps_to_reset - cur_step_) / num_steps_to_reset + action_final * cur_step_ / num_steps_to_reset
      robot.Step(action, robot_config.MotorControlMode.POSITION)

      time_dict = {'current_time': timesteps}
      foot_forces = robot.GetFootForce()
      print("Foot Forces:", foot_forces)
      df_dict = dict(time_dict)
      df_dict.update(foot_forces)
      cumulative_foot_forces.append(df_dict)
      timesteps += 1
      controller.update()

      time.sleep(robot.time_step)

  num_steps_to_reset = 5000

  action_initial = action_final
  desired_foot_position = robot.GetFootPositionsInBaseFrame()[1]
  desired_foot_position[0] = desired_foot_position[0] - 0.23
  action_final = np.array([0.3, 0.9, -2 * 0.9] + robot.ComputeMotorAnglesFromFootLocalPosition(1, desired_foot_position)[1] + [0.3, 0.65, -2 * 0.9] + [0.3, 0.65, -2 * 0.9])

  for cur_step_ in range(num_steps_to_reset):
      action = action_initial * (
              num_steps_to_reset - cur_step_) / num_steps_to_reset + action_final * cur_step_ / num_steps_to_reset
      robot.Step(action, robot_config.MotorControlMode.POSITION)

      time_dict = {'current_time': timesteps}
      foot_forces = robot.GetFootForce()
      print("Foot Forces:", foot_forces)
      df_dict = dict(time_dict)
      df_dict.update(foot_forces)
      cumulative_foot_forces.append(df_dict)
      timesteps += 1
      controller.update()

      time.sleep(robot.time_step)

  num_steps_to_reset = 2000
  action_initial = np.copy(action_final)
  action_final = action_start
  for cur_step_ in range(num_steps_to_reset):
      action = action_initial * (
                  num_steps_to_reset - cur_step_) / num_steps_to_reset + action_final * cur_step_ / num_steps_to_reset
      robot.Step(action, robot_config.MotorControlMode.POSITION)

      time_dict = {'current_time': timesteps}
      foot_forces = robot.GetFootForce()
      print("Foot Forces:", foot_forces)
      df_dict = dict(time_dict)
      df_dict.update(foot_forces)
      cumulative_foot_forces.append(df_dict)
      timesteps += 1
      controller.update()

      time.sleep(robot.time_step)

  num_steps_to_reset = 5000


  num_steps_to_reset = 1000

  action_initial = np.copy(action_final)

  for cur_step_ in range(num_steps_to_reset):
      action = action_initial * (
              num_steps_to_reset - cur_step_) / num_steps_to_reset + action_final * cur_step_ / num_steps_to_reset
      robot.Step(action, robot_config.MotorControlMode.POSITION)

      time_dict = {'current_time': timesteps}
      foot_forces = robot.GetFootForce()
      print("Foot Forces:", foot_forces)
      df_dict = dict(time_dict)
      df_dict.update(foot_forces)
      cumulative_foot_forces.append(df_dict)
      timesteps += 1
      controller.update()

      time.sleep(robot.time_step)

  global _DUTY_FACTOR
  global _INIT_PHASE_FULL_CYCLE
  global _INIT_LEG_STATE
  _DUTY_FACTOR = [0.6] * 4
  _INIT_PHASE_FULL_CYCLE = [0.9,0,0,0.9]
  _INIT_LEG_STATE = (
      gait_generator_lib.LegState.SWING,
      gait_generator_lib.LegState.STANCE,
      gait_generator_lib.LegState.STANCE,
      gait_generator_lib.LegState.SWING
    )
  controller = _setup_gait_controller(robot)
  controller.reset()
  controller.update()

  if FLAGS.use_gamepad:
    gamepad = gamepad_reader.Gamepad()
    command_function = gamepad.get_command
  else:
    command_function = _generate_example_linear_angular_speed

  if FLAGS.logdir:
    logdir = os.path.join(FLAGS.logdir,
                          datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(logdir)

  start_time = robot.GetTimeSinceReset()
  current_time = start_time
  com_vels, imu_rates, actions = [], [], []
  while current_time - start_time < FLAGS.max_time_secs:
    print('work')
    #time.sleep(0.0008) #on some fast computer, works better with sleep on real A1?
    start_time_robot = current_time
    start_time_wall = time.time()
    # Updates the controller behavior parameters.
    lin_speed, ang_speed, e_stop = command_function(current_time) #command_function(current_time)
    if current_time > 0.5:
      lin_speed = np.array([0.1,0,0])
    if e_stop:
      logging.info("E-stop kicked, exiting...")
      break
    _update_controller_params(controller, lin_speed, ang_speed)
    controller.update()
    hybrid_action, _ = controller.get_action()
    com_vels.append(np.array(robot.GetBaseVelocity()).copy())
    imu_rates.append(np.array(robot.GetBaseRollPitchYawRate()).copy())

    actions.append(hybrid_action)
    #print("Motor Controls: ")
    #print(hybrid_action)
    robot.Step(hybrid_action)
    current_time = robot.GetTimeSinceReset()

    if not FLAGS.use_real_robot:
      expected_duration = current_time - start_time_robot
      actual_duration = time.time() - start_time_wall
      if actual_duration < expected_duration:
        time.sleep(expected_duration - actual_duration)
    #print("actual_duration=", actual_duration)
  if FLAGS.use_gamepad:
    gamepad.stop()

  if FLAGS.logdir:
    np.savez(os.path.join(logdir, 'action.npz'),
             action=actions,
             com_vels=com_vels,
             imu_rates=imu_rates)
    logging.info("logged to: {}".format(logdir))

  df = pd.DataFrame(cumulative_foot_forces)
  df.plot(x='current_time', y=['5', '10', '15', '20'], kind='line')
  plt.savefig("foot_force_plots/" + str(datetime.datetime.utcnow()) + "_foot_forces_plot.png")
  # plt.show()


if __name__ == "__main__":
  app.run(main)
