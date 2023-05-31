import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class MyRDDLAgent:
    def __init__(self, action_space, num_actions=1, seed=None, env=None):
        self.action_space = action_space
        self.num_actions = num_actions
        self.env = DummyVecEnv([lambda: env])  # PPO requires a vectorized environment
        if seed is not None:
            self.action_space.seed(seed)
            self.env.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPO('MlpPolicy', self.env, verbose=1, device=self.device)

    def learn(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)

    def sample_action(self, state=None):
        obs = self.env.reset()
        action, _states = self.model.predict(obs, deterministic=True)
        return action
