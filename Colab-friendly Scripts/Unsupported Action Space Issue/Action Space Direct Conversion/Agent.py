
import torch
import gym
from gym.spaces import Box, Discrete, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from numpy import float32

class MyRDDLAgent:
    def __init__(self, action_space, num_actions=1, seed=None, env=None):
        self.env = env
        self.action_space = action_space
        self.num_actions = num_actions
        self.env = DummyVecEnv([lambda: env])  # PPO requires a vectorized environment
        if seed is not None:
            self.action_space.seed(seed)

        # Convert unsupported action spaces to supported ones
        if isinstance(self.action_space, Box):
            self.action_space = self.convert_box_action_space(self.action_space)
        elif isinstance(self.action_space, Dict):
            self.action_space = self.convert_dict_action_space(self.action_space)

        # Create a PPO model with additional parameters
        self.model = PPO(
            "MlpPolicy",
            self.env,
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

        # Convert the action back to the original action space format
        if isinstance(self.action_space, Box):
            action = self.convert_box_action(action)
        elif isinstance(self.action_space, Dict):
            action = self.convert_dict_action(action)

        return action

    def convert_box_action_space(self, action_space):
        return Discrete(2)  # Replace 2 with the appropriate number of discrete actions

    def convert_dict_action_space(self, action_space):
        if action_space == {'fx': Box(-1.0, 1.0, (1,), float32), 'fy': Box(-1.0, 1.0, (1,), float32)}:
            return Discrete(2)
        elif action_space == {'set-acc___a1': Discrete(3, start=-1), 'set-phi___a1': Box(-1.0, 1.0, (1,), float32), 'set-theta___a1': Box(-1.0, 1.0, (1,), float32)}:
            return Discrete(2)
        elif action_space == {'set-acc___a1': Discrete(3, start=-1), 'set-phi___a1': Discrete(3, start=-1), 'set-theta___a1': Discrete(3, start=-1)}:
            return Discrete(3)
        elif action_space == {'set-acc___a1': Box(-1.0, 1.0, (1,), float32), 'set-phi___a1': Box(-1.0, 1.0, (1,), float32), 'set-theta___a1': Box(-1.0, 1.0, (1,), float32)}:
            return Discrete(2)
        elif action_space == {'release___t1': Box(0.0, 100.0, (1,), float32), 'release___t2': Box(0.0, 100.0, (1,), float32), 'release___t3': Box(0.0, 100.0, (1,), float32)}:
            return Discrete(2)
        elif action_space == {'fan-in___z1': Box(0.05, float('inf'), (1,), float32), 'fan-in___z2': Box(0.05, float('inf'), (1,), float32), 'heat-input___h1': Box(-float('inf'), float('inf'), (1,), float32), 'heat-input___h2': Box(-float('inf'), float('inf'), (1,), float32)}:
            return Box(low=np.array([0.05, 0.05, -np.inf, -np.inf]), high=np.array([1.0, 1.0, np.inf, np.inf]), dtype=float32)
            return Discrete(2)
        elif action_space == {'recommend___c1__i1': Discrete(2), 'recommend___c1__i2': Discrete(2), 'recommend___c1__i3': Discrete(2), 'recommend___c1__i4': Discrete(2), 'recommend___c1__i5': Discrete(2), 'recommend___c2__i1': Discrete(2), 'recommend___c2__i2': Discrete(2), 'recommend___c2__i3': Discrete(2), 'recommend___c2__i4': Discrete(2), 'recommend___c2__i5': Discrete(2), 'recommend___c3__i1': Discrete(2), 'recommend___c3__i2': Discrete(2), 'recommend___c3__i3': Discrete(2), 'recommend___c3__i4': Discrete(2), 'recommend___c3__i5': Discrete(2), 'recommend___c4__i1': Discrete(2), 'recommend___c4__i2': Discrete(2), 'recommend___c4__i3': Discrete(2), 'recommend___c4__i4': Discrete(2), 'recommend___c4__i5': Discrete(2), 'recommend___c5__i1': Discrete(2), 'recommend___c5__i2': Discrete(2), 'recommend___c5__i3': Discrete(2), 'recommend___c5__i4': Discrete(2), 'recommend___c5__i5': Discrete(2)}:
            return Discrete(2)
        elif action_space == {'curProd___p1': Discrete(11), 'curProd___p2': Discrete(11), 'curProd___p3': Discrete(11)}:
            return Discrete(11)
        elif action_space == {'curProd___p1': Box(0.0, 10.0, (1,), float32), 'curProd___p2': Box(0.0, 10.0, (1,), float32), 'curProd___p3': Box(0.0, 10.0, (1,), float32)}:
            return Discrete(2)
        elif action_space == {'action': Box(-1.0, 1.0, (1,), float32)}:
            return Discrete(2)
        elif action_space == {'power-x___d1': Box(-0.1, 0.1, (1,), float32), 'power-x___d2': Box(-0.1, 0.1, (1,), float32), 'power-y___d1': Box(-0.1, 0.1, (1,), float32), 'power-y___d2': Box(-0.1, 0.1, (1,), float32), 'harvest___d1': Discrete(2), 'harvest___d2': Discrete(2)}:
            return Discrete(2)
        else:
            raise ValueError(f"Unsupported action space: {action_space}")

    def convert_box_action(self, action):
        if self.action_space == Discrete(2):
            if action < 0:
                return {'fx': -1.0, 'fy': 0.0}
            else:
                return {'fx': 1.0, 'fy': 0.0}
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")

    def convert_dict_action(self, action):
        if self.action_space == Discrete(2):
            if action == 0:
                return {'set-acc___a1': -1, 'set-phi___a1': 0.0, 'set-theta___a1': 0.0}
            else:
                return {'set-acc___a1': 1, 'set-phi___a1': 0.0, 'set-theta___a1': 0.0}
        elif self.action_space == Discrete(3):
            if action == 0:
                return {'set-acc___a1': -1, 'set-phi___a1': -1, 'set-theta___a1': -1}
            elif action == 1:
                return {'set-acc___a1': 0, 'set-phi___a1': 0, 'set-theta___a1': 0}
            else:
                return {'set-acc___a1': 1, 'set-phi___a1': 1, 'set-theta___a1': 1}
        elif self.action_space == Discrete(11):
            return {'curProd___p1': action, 'curProd___p2': action, 'curProd___p3': action}
        elif self.action_space == Discrete(2):
            if action == 0:
                return {'harvest___d1': 0, 'harvest___d2': 0}
            else:
                return {'harvest___d1': 1, 'harvest___d2': 1}
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")
