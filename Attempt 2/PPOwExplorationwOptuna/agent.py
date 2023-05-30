import torch
import torch.optim as optim
import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

class MyRDDLAgent:
    def __init__(self, action_space, num_actions=1, seed=None, env=None, epsilon=0.1):
        self.action_space = action_space
        self.num_actions = num_actions
        self.env = DummyVecEnv([lambda: env])  # PPO requires a vectorized environment
        if seed is not None:
            self.action_space.seed(seed)
            self.env.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPO('MlpPolicy', self.env, verbose=1, device=self.device)
        
        self.epsilon = epsilon  # New attribute for the exploration rate
        
        self.trial = None  # Placeholder for Optuna trial object

    def optimize_agent(self, trial):
        """Defines and optimizes PPO hyperparameters within the given trial."""

        self.trial = trial
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        n_steps = trial.suggest_categorical('n_steps', [16, 32, 64, 128, 256, 512, 1024, 2048])
        gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
        ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
        clip_range = trial.suggest_uniform('clip_range', 0.1, 0.4)
        n_epochs = trial.suggest_categorical('n_epochs', [1, 5, 10, 20])

        self.model.learning_rate = learning_rate
        self.model.n_steps = n_steps
        self.model.gamma = gamma
        self.model.ent_coef = ent_coef
        self.model.clip_range = clip_range
        self.model.n_epochs = n_epochs

        self.model.learn(total_timesteps=10000)
        
    def sample_action(self, state=None):
        obs = self.env.reset()
        # Use epsilon-greedy exploration
        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()
        else:
            action, _states = self.model.predict(obs, deterministic=True)
        return action
    


