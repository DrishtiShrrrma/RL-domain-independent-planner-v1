import torch
from stable_baselines3 import PPO
from stable_baselines3.common.envs import gym

class MyRDDLAgent:
    def __init__(self, action_space, num_actions=1, seed=None):
        self.action_space = action_space
        self.num_actions = num_actions
        if seed is not None:
            self.action_space.seed(seed)
        
        # Create a PPO model
        self.model = PPO("MlpPolicy", self.action_space, verbose=1)

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)    #check this (HF)

    def sample_action(self, state=None):    #check for state bit
        # Instead of sampling random actions, we will now get actions from our trained model
        action, _states = self.model.predict(state)
        return action




"""import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

class MyRDDLAgent:
    def __init__(self, env, model_path=None):
        self.env = DummyVecEnv([lambda: env])  # Stable Baselines3 requires vectorized environments
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        if model_path is not None:
            self.model.load(model_path)
            
    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)
    
    def save(self, model_path):
        self.model.save(model_path)

    def sample_action(self, state=None):
        action, _ = self.model.predict(state, deterministic=True)
        return action
"""
