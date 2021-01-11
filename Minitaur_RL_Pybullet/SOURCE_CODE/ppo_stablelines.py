import gym
import pybullet_envs

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1
# This is a testing code with Stable Baselines
# It is required to install openmpi which is not in conda
# and Stable Baselines

# https://stable-baselines.readthedocs.io/en/master/guide/install.html
# 
env = gym.make('MinitaurBulletEnv-v0')

model = PPO1(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000000)


obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()