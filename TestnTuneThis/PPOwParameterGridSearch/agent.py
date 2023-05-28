from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from sklearn.model_selection import ParameterGrid
import numpy as np

class MyRDDLAgent:
    def __init__(self, action_space, num_actions=1, initial_lr=0.01):
        self.action_space = action_space
        self.num_actions = num_actions
        self.learning_rate = initial_lr
        self.model = PPO("MlpPolicy", self.action_space, learning_rate=self.learning_rate, verbose=1)

    def train(self, env, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps, env=env)

    def sample_action(self, observation):
        action, _ = self.model.predict(observation)
        return action

    def tune_hyperparameters(self, env, num_episodes=10):
        # Define the hyperparameter space
        hyperparameters = {
            'learning_rate': [0.001, 0.01, 0.1]
        }

        # Create a grid over the hyperparameter space
        param_grid = ParameterGrid(hyperparameters)

        # Loop over each set of hyperparameters
        best_reward = -np.inf
        best_params = None
        for params in param_grid:
            # Update the agent's hyperparameters
            self.model.learning_rate = params['learning_rate']
            self.learning_rate = params['learning_rate']

            # Evaluate the agent's performance
            total_reward = 0
            for i in range(num_episodes):
                observation = env.reset()
                done = False
                while not done:
                    action = self.sample_action(observation)
                    observation, reward, done, _ = env.step(action)
                    total_reward += reward

            # Update the best parameters if this performance was better
            if total_reward > best_reward:
                best_reward = total_reward
                best_params = params

        # Update the agent to the best found parameters
        self.model.learning_rate = best_params['learning_rate']
        self.learning_rate = best_params['learning_rate']
        print('Best learning rate:', self.learning_rate)

