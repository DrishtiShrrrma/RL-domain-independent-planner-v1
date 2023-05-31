import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import optuna

class MyRDDLAgent:
    def __init__(self, action_space, observation_space, num_actions=1, seed=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_actions = num_actions
        if seed is not None:
            np.random.seed(seed)

        # Create a trial study to optimize hyperparameters
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self._objective, n_trials=100)
        self.best_params = self.study.best_params

        # Create a model with optimized hyperparameters
        self.model = self._create_ppo_model(self.best_params)

    def _create_ppo_model(self, params):
        env = DummyVecEnv([lambda: self])
        model = PPO("MlpPolicy", env, verbose=1, 
                    learning_rate=params['learning_rate'], 
                    n_steps=params['n_steps'], 
                    batch_size=params['batch_size'],
                    gamma=params['gamma'])
        return model

    def sample_action(self, state=None):
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def _objective(self, trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        n_steps = trial.suggest_int('n_steps', 16, 2048)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
        gamma = trial.suggest_float('gamma', 0.9, 0.9999)

        env = DummyVecEnv([lambda: self])
        model = PPO("MlpPolicy", env, verbose=0, 
                    learning_rate=learning_rate, 
                    n_steps=n_steps, 
                    batch_size=batch_size,
                    gamma=gamma)

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        return mean_reward
