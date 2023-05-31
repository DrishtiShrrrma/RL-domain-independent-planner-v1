import numpy as np
import gym
from stable_baselines3 import A2C
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
        self.model = self._create_a2c_model(self.best_params)

    def _create_a2c_model(self, params):
        env = DummyVecEnv([lambda: self])
        model = A2C("MlpPolicy", env, verbose=1, 
                    learning_rate=params['learning_rate'], 
                    n_steps=params['n_steps'],
                    gamma=params['gamma'],
                    gae_lambda=params['gae_lambda'],
                    ent_coef=params['ent_coef'],
                    vf_coef=params['vf_coef'])
        return model

    def sample_action(self, state=None):
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def _objective(self, trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        n_steps = trial.suggest_int('n_steps', 16, 2048)
        gamma = trial.suggest_float('gamma', 0.9, 0.9999)
        gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.9999)
        ent_coef = trial.suggest_float('ent_coef', 0.0, 1.0)
        vf_coef = trial.suggest_float('vf_coef', 0.0, 1.0)

        env = DummyVecEnv([lambda: self])
        model = A2C("MlpPolicy", env, verbose=0, 
                    learning_rate=learning_rate, 
                    n_steps=n_steps, 
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef)

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
        return mean_reward

