"""
CPG in polar coordinates based on: 
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. IJspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
import time
import numpy as np
import matplotlib
from sys import platform

from env.quadruped import Quadruped
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg")
else: # linux
  matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.quadruped_gym_env import QuadrupedGymEnv

import env.configs_a1 as robot_config


class HopfNetwork():
  """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.  

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)
  """
  def __init__(self,
                mu=1**2,                # converge to sqrt(mu)
                omega_swing=1*2*np.pi,  # MUST EDIT
                omega_stance=1*2*np.pi, # MUST EDIT
                gait="TROT",            # change depending on desired gait
                coupling_strength=1,    # coefficient to multiply coupling matrix
                couple=True,            # should couple
                time_step=0.001,        # time step 
                ground_clearance=0.05,  # foot swing height 
                ground_penetration=0.01,# foot stance penetration into ground 
                robot_height=0.25,      # in nominal case (standing) 
                des_step_len=0.04,      # desired step length 
                ):
    
    ###############
    # initialize CPG data structures: amplitude is row 0, and phase is row 1
    self.X = np.zeros((2,4))

    # save parameters 
    self._mu = mu
    self._omega_swing = omega_swing
    self._omega_stance = omega_stance  
    self._couple = couple
    self._coupling_strength = coupling_strength
    self._dt = time_step
    self._set_gait(gait)

    # set oscillator initial conditions  
    self.X[0,:] = np.random.rand(4) * .1
    self.X[1,:] = self.PHI[0,:] 

    # save body and foot shaping
    self._ground_clearance = ground_clearance 
    self._ground_penetration = ground_penetration
    self._robot_height = robot_height 
    self._des_step_len = des_step_len


  def _set_gait(self,gait):
    """ For coupling oscillators in phase space. 
    [TODO] update all coupling matrices
    """
    self.PHI_trot = np.array([[0, -1, -1, 0],
                              [ 1, 0, 0, 1 ],
                              [ 1, 0, 0, 1 ],
                              [0, -1, -1, 0]])*np.pi

    self.PHI_walk = np.array([[ 0, -2, -1, 1],
                              [ 2, 0, 1, -1 ],
                              [ 1, -1, 0, 2 ],
                              [-1, 1, -2, 0]])*np.pi/2

    self.PHI_pace = np.array([[ 0, 1, 0, 1 ],
                              [-1, 0, -1, 0],
                              [ 0, 1, 0, 1 ],
                              [-1, 0, -1, 0]])*np.pi

    self.PHI_bound = np.array([[0, 0, -1, -1],
                              [0, 0, -1, -1],
                              [ 1, 1, 0, 0 ],
                              [ 1, 1, 0, 0 ]])*np.pi

    if gait == "TROT":
      print('TROT')
      self.PHI = self.PHI_trot
    elif gait == "PACE":
      print('PACE')
      self.PHI = self.PHI_pace
    elif gait == "BOUND":
      print('BOUND')
      self.PHI = self.PHI_bound
    elif gait == "WALK":
      print('WALK')
      self.PHI = self.PHI_walk
    else:
      raise ValueError( gait + 'not implemented.')


  def update(self):
    """ Update oscillator states. """

    # update parameters, integrate
    self._integrate_hopf_equations()
    
    # map CPG variables to Cartesian foot xz positions (Equations 8, 9) 
    r = self.X[0,:] 
    theta = self.X[1,:]
    x = -self._des_step_len * r * np.cos(theta) # [TODO]
    z = np.zeros(x.shape)
    i = 0
    for theta_i in theta:
      if np.sin(theta_i) > 0:
        z[i] = -self._robot_height+self._ground_clearance*np.sin(theta_i)
      else:
        z[i] = -self._robot_height+self._ground_penetration*np.sin(theta_i)
      i+=1
    # [TODO]

    return x, z
      
        
  def _integrate_hopf_equations(self):
    """ Hopf polar equations and integration. Use equations 6 and 7. """
    # bookkeeping - save copies of current CPG states 
    X = self.X.copy()
    X_dot = np.zeros((2,4))
    alpha = 50

    # loop through each leg's oscillator
    for i in range(4):
      # get r_i, theta_i from X
      r = X[0,i] # [TODO]
      theta  = X[1,i]
      # compute r_dot (Equation 6)
      r_dot = alpha*(self._mu-r**2)*r # [TODO]
      # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
      theta = theta%(2*np.pi)
      if (theta>=0 and theta<=np.pi):
        theta_dot = self._omega_swing # [TODO]
      elif (theta>np.pi and theta<=2*np.pi):
        theta_dot = self._omega_stance

      # loop through other oscillators to add coupling (Equation 7)
      if self._couple:
        for j in range(4):
          if j!=i:
            r_j = X[0,j]
            theta_j = X[1,j]%(2*np.pi)
            theta_dot += r_j*self._coupling_strength*np.sin(theta_j-theta-self.PHI[i,j]) # [TODO]

      # set X_dot[:,i]
      X_dot[:,i] = [r_dot, theta_dot]

    # integrate 
    self.X = X + X_dot*self._dt # [TODO]
    # mod phase variables to keep between 0 and 2pi
    self.X[1,:] = self.X[1,:] % (2*np.pi)
    self.X_dot = X_dot



if __name__ == "__main__":

  ADD_CARTESIAN_PD = True
  TIME_STEP = 0.001
  foot_y = 0.0838 # this is the hip length 
  sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

  env = QuadrupedGymEnv(render=True,              # visualize
                      on_rack=False,              # useful for debugging! 
                      isRLGymInterface=False,     # not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="PD",
                      add_noise=False,    # start in ideal conditions
                      record_video=True
                      )

  # Trot --------------------------------------
  omg_stance = 3*np.pi
  omg_swing = 4*omg_stance
  couple_strength = 1
  gc =0.03
  gp =0.01 
  h =0.25
  step_len=0.03
  # joint PD gains
  kp=np.array([150,70,70])
  kd=np.array([5,1,1])
  # Cartesian PD gains
  kpCartesian = np.diag([2500]*3)
  kdCartesian = np.diag([40]*3)

  # Pace ----------------------------------------
  # omg_stance = 4*np.pi
  # omg_swing = 3*omg_stance
  # couple_strength = 1
  # gc =0.05
  # gp =0.01
  # h =0.25
  # step_len=0.04
  # # joint PD gains
  # kp=np.array([150,70,70])
  # kd=np.array([5,1,1])
  # # Cartesian PD gains
  # kpCartesian = np.diag([2500]*3)
  # kdCartesian = np.diag([40]*3)

  # Walk -----------------------------------------
  omg_stance = 3*np.pi
  omg_swing = 4*omg_stance
  couple_strength = 1
  gc =0.03
  gp =0.01 
  h =0.25
  step_len=0.03
  # joint PD gains
  kp = np.array([150, 72, 72])
  kd = np.array([2, 1, 1])
  # Cartesian PD gains
  kpCartesian = np.diag([2500]*3)
  kdCartesian = np.diag([40]*3)

  # Bound ----------------------------------------
  # omg_stance = 5*np.pi
  # omg_swing = 3*omg_stance
  # couple_strength = 1
  # gc =0.03
  # gp =0.005
  # h =0.25
  # step_len=0.03
  # # joint PD gains
  # kp=np.array([150,70,70])
  # kd=np.array([5,1,1])
  # # Cartesian PD gains
  # kpCartesian = np.diag([2500]*3)
  # kdCartesian = np.diag([40]*3)

  # initialize Hopf Network, supply gait
  Hopf = HopfNetwork(mu=1**2,                          # converge to sqrt(mu)
                omega_swing = omg_swing,  
                omega_stance = omg_stance, 
                gait="WALK",                          # change depending on desired gait
                coupling_strength=couple_strength,    # coefficient to multiply coupling matrix
                couple=True,                          # should couple
                time_step=TIME_STEP,                  # time step 
                ground_clearance=gc,                  # foot swing height 
                ground_penetration=gp,                # foot stance penetration into ground 
                robot_height=h,                       # in nominal case (standing) 
                des_step_len=step_len,                # desired step length 
                )
  

  TEST_STEPS = int(10 / (TIME_STEP))
  t = np.arange(TEST_STEPS)*TIME_STEP

  # [TODO] initialize data structures to save CPG and robot states
  cpg_states = []
  desired_joint_angles = []
  desired_joint_angles_FR = []
  desired_foot_pos = []
  desired_foot_pos_FR = []
  joint_angles = []
  joint_angles_FR = []
  foot_pos = []
  foot_pos_FR = []
  base_speed = []
  base_pos = []
  motor_torques = []
  motor_vel = []


  for j in range(TEST_STEPS):
    # initialize torque array to send to motors
    action = np.zeros(12)
    # get desired foot positions from CPG
    xs, zs = Hopf.update()

    # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    q = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()
    joint_angles.append(q)

    base_speed_2 = env.robot.GetBaseLinearVelocity()
    base_speed.append(base_speed_2)
    base_pos2 = env.robot.GetBasePosition()
    base_pos.append(base_pos2)

    motor_torques = env.robot.GetMotorTorques()
    motor_vel = env.robot.GetMotorVelocities()

    # print(base_pos,base_speed)
    # loop through desired foot positions and calculate torques
    for i in range(4):
      # initialize torques for leg i
      tau = np.zeros(3)
      # get desired foot i pos (xi, yi, zi) in leg frame
      leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])
      desired_foot_pos.append(leg_xyz)

      # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
      leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)  # [TODO]
      desired_joint_angles.append(leg_q)
      # Add joint PD contribution to tau for leg i (Equation 4) # [TODO]
      tau = np.matmul(kp, (leg_q - q[3 * i:3 * i + 3])) - np.matmul(kd, dq[3 * i:3 * i + 3])

      if i == 0:
        desired_foot_pos_FR.append(leg_xyz)
        desired_joint_angles_FR.append(leg_q)
        joint_angles_FR.append(q[0:3])

      # add Cartesian PD contribution
      if ADD_CARTESIAN_PD:
        # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
        J, pos = env.robot.ComputeJacobianAndPosition(i)  # [TODO]
        foot_pos.append(pos)
        # Get current foot velocity in leg frame (Equation 2)
        v = np.dot(J, q[3 * i:3 * i + 3])  # [TODO]
        # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
        tau = np.matmul(np.transpose(J),
                        (np.matmul(kpCartesian, (leg_xyz - pos)) - np.matmul(kdCartesian, v)))  # [TODO]
      if i == 0:
        J, pos = env.robot.ComputeJacobianAndPosition(i)
        foot_pos_FR.append(pos)

      # Set tau for legi in action vector
      action[3 * i:3 * i + 3] = tau

    # send torques to robot and simulate TIME_STEP seconds
    env.step(action)

    # [TODO] save any CPG or robot states
    cpg_states.append([Hopf.X, Hopf.X_dot])

  #### Extract CPG States, foot & joint angles #########
  cpg_states = np.array(cpg_states)
  r_theta = cpg_states[:, 0]
  r_theta_dot = cpg_states[:, 1]
  r = r_theta[:, 0]
  r_dot = r_theta_dot[:, 0]
  theta = r_theta[:, 1]
  theta_dot = r_theta_dot[:, 1]

  joint_angles_FR = np.array(joint_angles_FR)
  desired_joint_angles_FR = np.array(desired_joint_angles_FR)
  foot_pos_FR = np.array(foot_pos_FR)
  desired_foot_pos_FR = np.array(desired_foot_pos_FR)

  base_speed = np.array(base_speed)
  base_pos = np.array(base_pos)

  ######################### COT Calculations ######################

  motor_torques = np.array(motor_torques)
  motor_vel = np.array(motor_vel)
  P = np.sum(np.abs(motor_torques * motor_vel))
  # base_speed_all = np.sqrt(base_speed)
  avg_vel_x = np.mean(base_speed[:, 0])
  avg_vel_y = np.mean(base_speed[:, 1])
  avg_vel_vect = np.sqrt(base_speed[:, 0]**2 + base_speed[:, 1]**2)
  avg_vel = np.sqrt(avg_vel_x**2 + avg_vel_y**2)
  print("Avg vel: ", avg_vel)
  CoT = P/(12*9.81*avg_vel)

  avg_dist_vect = np.sqrt(base_pos[:, 0]**2 + base_pos[:, 1]**2)

  print("CoT is : ", CoT)


  #####################################################
  #                        PLOTS                      #
  #####################################################

  ###### plot CPG states ###########

  fig = plt.figure()

  # ax = fig.add_subplot(111)  # The big subplot
  ax1 = fig.add_subplot(221)
  ax2 = fig.add_subplot(222)
  ax3 = fig.add_subplot(223)
  ax4 = fig.add_subplot(224)

  ax1.plot(t, r[:, 0], t, r_dot[:, 0], t, theta[:, 0], t, theta_dot[:, 0])
  ax1.legend(('r', 'r_dot', 'theta', 'theta_dot'))
  print('---------- r:', r[-1, 0])
  ax2.plot(t, r[:, 1], t, r_dot[:, 1], t, theta[:, 1], t, theta_dot[:, 1])
  ax2.legend(('r', 'r_dot', 'theta', 'theta_dot'))

  ax3.plot(t, r[:, 2], t, r_dot[:, 2], t, theta[:, 2], t, theta_dot[:, 2])
  ax3.legend(('r', 'r_dot', 'theta', 'theta_dot'))

  ax4.plot(t, r[:, 3], t, r_dot[:, 3], t, theta[:, 3], t, theta_dot[:, 3])
  ax4.legend(('r', 'r_dot', 'theta', 'theta_dot'))

  # ax1.set_xlabel('Time')
  ax1.set_ylabel('Arbitrary')
  # ax2.set_xlabel('Time')
  ax2.set_ylabel('Arbitrary')
  ax3.set_xlabel('Time [s]')
  ax3.set_ylabel('Arbitrary')
  ax4.set_xlabel('Time [s]')
  ax4.set_ylabel('Arbitrary')


  ax1.set_title('FR')
  ax2.set_title('FL')
  ax3.set_title('RR')
  ax4.set_title('RL')
  st = fig.suptitle("CPG States for each leg", fontsize="x-large")
  fig.subplots_adjust(top=0.85)

  fig.legend()
  plt.show()


  ########## plot feet ###########

  fig2 = plt.figure()

  # ax = fig.add_subplot(111)  # The big subplot
  ax1 = fig2.add_subplot(211)
  ax2 = fig2.add_subplot(212)

  ax1.plot(t, foot_pos_FR[:, 0], t, desired_foot_pos_FR[:, 0])
  ax1.legend(('x', 'x desired'))

  ax2.plot(t, foot_pos_FR[:, 2], t, desired_foot_pos_FR[:, 2])
  ax2.legend(('z', 'z desired'))

  ax1.set_xlabel('Time [s]')
  ax1.set_ylabel('Distance [m]')
  ax2.set_ylabel('Distance [m]')
  ax2.set_xlabel('Time [s]')

  ax1.set_title('X foot positions')
  ax2.set_title('Z foot positions')

  st = fig2.suptitle("Current vs Desired foot position (FR)", fontsize="x-large")
  fig2.subplots_adjust(top=0.85)

  fig2.legend()
  plt.show()

  ######## plot joint angles ####################


  fig3 = plt.figure()

  # ax = fig.add_subplot(111)  # The big subplot
  ax1 = fig3.add_subplot(311)
  ax2 = fig3.add_subplot(312)
  ax3 = fig3.add_subplot(313)

  ax1.plot(t, joint_angles_FR[:, 0], t, desired_joint_angles_FR[:, 0])
  ax1.legend(('q1', 'q1 desired'))

  ax2.plot(t, joint_angles_FR[:, 1], t, desired_joint_angles_FR[:, 1])
  ax2.legend(('q2', 'q2 desired'))

  ax3.plot(t, joint_angles_FR[:, 2], t, desired_joint_angles_FR[:, 2])
  ax3.legend(('q3', 'q3 desired'))

  ax1.set_xlabel('Time [s]')
  ax1.set_ylabel('Radians')
  ax2.set_ylabel('Radians')
  ax2.set_xlabel('Time [s]')
  ax2.set_ylabel('Radians')
  ax2.set_xlabel('Time [s]')

  ax1.set_title('q1 angles')
  ax2.set_title('q2 angles')
  ax3.set_title('q3 angles')

  st = fig3.suptitle("Current vs Desired Joint Angles (FR)", fontsize="x-large")
  fig3.subplots_adjust(top=0.85)

  fig3.legend()
  plt.show()

  ######### Period phases ####################

  fig4 = plt.figure()

  # ax = fig.add_subplot(111)  # The big subplot
  ax1 = fig4.add_subplot(221)
  ax2 = fig4.add_subplot(222)
  ax3 = fig4.add_subplot(223)
  ax4 = fig4.add_subplot(224)

  ax1.plot(theta[:, 0], theta_dot[:, 0], theta[:, 3], theta_dot[:, 3])
  ax1.legend(('theta_dot FR', 'theta_dot RL'))

  ax2.plot(theta[:, 1], theta_dot[:, 1], theta[:, 2], theta_dot[:, 2])
  ax2.legend(('theta_dot FL', 'theta_dot RR'))

  ax3.plot(theta[:, 0], theta_dot[:, 0],theta[:, 2], theta_dot[:, 2])
  ax3.legend(('theta_dot FR', 'theta_dot RR'))

  ax4.plot(theta[:, 0], theta_dot[:, 0],theta[:, 1], theta_dot[:, 1], theta[:, 2], theta_dot[:, 2], theta[:, 3],  theta_dot[:, 3])
  ax4.legend(('theta_dot FR','theta_dot FL', 'theta_dot RR', 'theta_dot RL'))

  # ax1.set_xlabel('Time')
  ax1.set_ylabel('Theta dot [rad/s]')
  # ax2.set_xlabel('Time')
  ax2.set_ylabel('Theta dot [rad/s]')
  ax3.set_xlabel('Theta [rad]')
  ax3.set_ylabel('Theta dot [rad/s]')
  ax4.set_xlabel('Theta [rad]')
  ax4.set_ylabel('Theta dot [rad/s]')

  ax1.set_title('FR & RL')
  ax2.set_title('FL & RR')
  ax3.set_title('FR & RR')
  ax4.set_title('All')
  st = fig4.suptitle("Phase limit cycle", fontsize="x-large")
  fig4.subplots_adjust(top=0.85)

  fig4.legend()
  plt.show()

 ##################### plot position and speed ########################

  fig5 = plt.figure()

  # ax = fig.add_subplot(111)  # The big subplot
  ax1 = fig5.add_subplot(211)
  ax2 = fig5.add_subplot(212)

  ax1.plot(t, base_pos[:, 0], t, base_pos[:, 1], t, avg_dist_vect)
  ax1.legend(('X Distance', 'Y Distance', 'XY Avg Distance'))

  ax2.plot(t, base_speed[:, 0], t, base_speed[:, 1], t, avg_vel_vect)
  ax2.legend(('X Velocity', 'Y Velocity', 'XY Avg Velocity'))
  # ax2.legend(('z', 'z desired'))

  ax1.set_xlabel('Time [s]')
  ax1.set_ylabel('Distance [m]')
  ax2.set_ylabel('Speed [m/s]')
  ax2.set_xlabel('Time [s]')

  ax1.set_title('Achieved X Distance')
  ax2.set_title('Achieved X Body Velocity')

  st = fig5.suptitle("Achieved Distance & Velocity", fontsize="x-large")
  fig5.subplots_adjust(top=0.85)

  fig5.legend()
  plt.show()