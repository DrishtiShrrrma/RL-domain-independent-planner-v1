
#### Significance of State = None

1. If the state parameter is not provided when calling the sample_action method, it will remain as None => agent will not use any specific state information when sampling an action. Instead, it will directly reset the environment and sample an action based on the current observation.

2. The state parameter is useful when we want to provide a specific state to the agent for action selection. This can be beneficial in scenarios where we want the agent to make decisions based on a specific state representation or when we want to continue the agent's behavior from a specific state in the environment.

3. If we have specific requirements for the state parameter or if we want the agent to utilize a specific state for action selection, we can pass the desired state value when calling the sample_action method. Otherwise, leaving it as None will result in the agent making decisions based solely on the current observation.



#### Hybrid Action Spaces

Stable Baselines3 requires the action space to be either continuous (gym.spaces.Box) or discrete (gym.spaces.Discrete), but does not natively support hybrid action spaces. If we have hybrid action spaces, we will need to create a custom wrapper around your environments that translates hybrid action spaces into a form that Stable Baselines3 can handle.



Let's say, we don't want to train and want to just utilize an already tuned model:

Method:

app.py:
class MyRDDLAgent:
    def __init__(self, env, model_path=None):
        self.env = DummyVecEnv([lambda: env])  # Stable Baselines3 requires vectorized environments
        if model_path is not None:
            self.model = PPO.load(model_path)
        else:
            self.model = PPO("MlpPolicy", self.env, verbose=1)

    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)
    
    def save(self, model_path):
        self.model.save(model_path)

    def sample_action(self, state=None):
        action, _ = self.model.predict(state, deterministic=True)
        return action

main.py:

try:
    ################################################################
    # Initialize your agent here:
    agent = MyRDDLAgent(myEnv, model_path="path_to_your_model_weights.pkl")
    ################################################################
except:
    ...



In a separate script train it using - 
env = # Your environment initialization here
agent = MyRDDLAgent(env)
agent.train(total_timesteps=10000)  # Or however many timesteps you want to train for
agent.save("path_to_your_model_weights.pkl")
 
