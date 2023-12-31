import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg")
else: # linux
  matplotlib.use('TkAgg')

# stable baselines
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.cmd_util import make_vec_env

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "PPO"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '111121133812'
log_dir = interm_dir + '122021012525'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config['render'] = False
env_config['record_video'] = False
env_config['add_noise'] = True
env_config['competition_env'] = False

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show()

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

base_pos = []
base_vel = []
episodes = []
energy = 0
for i in range(2000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    base_pos.append(env.envs[0].env.robot.GetBasePosition()[0])
    base_vel.append(env.envs[0].env.robot.GetBaseLinearVelocity()[0])
    energy += 0.001 * abs(np.dot(env.envs[0].env.robot.GetMotorTorques(), env.envs[0].env.robot.GetMotorVelocities()))
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episodes.append(info[0]['episode'])
        episode_reward = 0

num_episodes = len(episodes)
i = 0
start = 0
figure, axis = plt.subplots(2)
while i < num_episodes:
    index = episodes[i]['l']
    axis[0].plot(range(index-1), base_pos[start:(start+index-1)])
    axis[0].set_title("Distance Travelled per Episode")
    axis[1].plot(range(index-2), base_vel[start:(start + index-2)])
    axis[1].set_title("Velocity per Episode")
    start = start + index
    i += 1

plt.show()
cot = energy / (12 * 9.81 * 50)
print(cot)