import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import ActionWrapper
from gym.spaces import Box, Discrete, Dict
import numpy as np

class MyActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(MyActionWrapper, self).__init__(env)

        if isinstance(self.env.action_space, gym.spaces.Dict):
            self.num_actions = len(env.action_space.spaces)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            # For Box space, you might need to consider the shape of the space for determining num_actions
            self.num_actions = env.action_space.shape[0]
        else:
            self.num_actions = 1


    def action(self, action):
        # Ensure action is a dictionary
        assert isinstance(action, dict)

        transformed_action = {}
        for action_name, action_value in action.items():
            space = self.env.action_space.spaces[action_name]
            if isinstance(space, Box):
                transformed_action[action_name] = float(action_value) if space.shape[0] == 1 else list(map(float, action_value))
            elif isinstance(space, Discrete):
                transformed_action[action_name] = int(action_value)

        return transformed_action



class MyRDDLAgent:
    def __init__(self, action_space, num_actions=1, seed=None, env=None):
        self.env = env
        self.action_space = action_space
        self.num_actions = num_actions
        self.env = DummyVecEnv([lambda: env])  # PPO requires a vectorized environment
        if seed is not None:
            self.action_space.seed(seed)
        
        # Create a PPO model with additional parameters
        self.model = PPO(
            "MlpPolicy",
            self.env,
            MyActionWrapper(env),
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./ppo_tensorboard/"
        )

    def train(self, total_timesteps=10000, save_path="ppo_model.pkl"):
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(save_path)
        
    def sample_action(self, state=None):  
        action, _states = self.model.predict(state, deterministic=True)
        return action.tolist()

