
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
    
    agent = MyRDDLAgent(myEnv, model_path="path_to_model_weights.pkl")
    
    ################################################################

except:
    ...



In a separate script train it using - 

env = # Your environment initialization here

agent = MyRDDLAgent(env)

agent.train(total_timesteps=10000)  # Or however many timesteps you want to train for

agent.save("path_to_your_model_weights.pkl")

 
### PPO

The PPO (Proximal Policy Optimization) algorithm inherently uses an Actor-Critic architecture. It employs two separate networks - the actor, which suggests the next action to take given a state, and the critic, which estimates the value function of being in a state. The critic helps to reduce the variance of the expected return, making the learning process more stable.



### PPO-GAE
To compare different settings of gae_lambda to see how much GAE affects the performance - create two agents, one with gae_lambda set to 1 (which is equivalent to not using GAE, or a high bias towards immediate rewards) and the other with gae_lambda set to a value less than 1 (indicating the use of GAE).



Stable-baselines3 PPO

![image](https://github.com/DrishtiShrrrma/domain-independent-planner-v1/assets/129742046/4b8652a5-ad69-459c-a01f-bb053c822475)

Note: To load the saved model - use the load method provided by Stable Baselines:

model = PPO.load("ppo_model.pkl")

gym.spaces.dict.Dict ---> observation space is a dictionary - useful in situations where the observation is comprised of several different types of data that might not fit neatly into a traditional array or tensor format - one can handle a Dict space by processing each component of the dict individually.

Error Message: 
/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py:121: DeprecationWarning: pkg_resources is deprecated as an API
  warnings.warn("pkg_resources is deprecated as an API", DeprecationWarning)
/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py:2870: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('google')`.
Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
  declare_namespace(pkg)
/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py:2870: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('google.cloud')`.
Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
  declare_namespace(pkg)
/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py:2870: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('google.logging')`.
Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
  declare_namespace(pkg)
/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py:2870: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('mpl_toolkits')`.
Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
  declare_namespace(pkg)
/usr/local/lib/python3.10/dist-packages/pkg_resources/__init__.py:2870: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('sphinxcontrib')`.
Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
  declare_namespace(pkg)
/usr/local/lib/python3.10/dist-packages/torch/utils/tensorboard/__init__.py:4: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if not hasattr(tensorboard, "__version__") or LooseVersion(
2023-06-01 23:40:21.908911: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-06-01 23:40:23.259385: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
['main.py', 'HVAC', '1', 'None', '5']
preparing to launch instance 1 of domain HVAC...
/usr/local/lib/python3.10/dist-packages/gym/core.py:317: DeprecationWarning: WARN: Initializing wrapper in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.
  deprecation(
/usr/local/lib/python3.10/dist-packages/stable_baselines3/common/vec_env/patch_gym.py:49: UserWarning: You provided an OpenAI Gym environment. We strongly recommend transitioning to Gymnasium environments. Stable-Baselines3 is automatically wrapping your environments in a compatibility layer, which could potentially cause issues.
  warnings.warn(
Using cpu device
Timed out! ( 0.0029287338256835938  seconds)
This domain will continue exclusively with default actions!
Error during agent initialization and training: The algorithm only supports (<class 'gymnasium.spaces.box.Box'>, <class 'gymnasium.spaces.discrete.Discrete'>, <class 'gymnasium.spaces.multi_discrete.MultiDiscrete'>, <class 'gymnasium.spaces.multi_binary.MultiBinary'>) as action spaces but Dict('fan-in___z1': Box(0.05, inf, (1,), float32), 'fan-in___z2': Box(0.05, inf, (1,), float32), 'heat-input___h1': Box(-inf, inf, (1,), float32), 'heat-input___h2': Box(-inf, inf, (1,), float32)) was provided
Timed out! ( 2.86102294921875e-06  seconds)
This episode will continue with default actions!

![image](https://github.com/DrishtiShrrrma/domain-independent-planner-v1/assets/129742046/ba2e4b3c-0398-4907-a495-7e0fb69f128f)




Action Spaces that domain-independent planner needs to handdle:

RaceCar:

Action space: Dict('fx': Box(-1.0, 1.0, (1,), float32), 'fy': Box(-1.0, 1.0, (1,), float32))
Shape of action space: None

UAV_mixed:

Action space: Dict('set-acc___a1': Discrete(3, start=-1), 'set-phi___a1': Box(-1.0, 1.0, (1,), float32), 'set-theta___a1': Box(-1.0, 1.0, (1,), float32))
Shape of action space: None

HVAC:

Action space: Dict('set-acc___a1': Discrete(3, start=-1), 'set-phi___a1': Box(-1.0, 1.0, (1,), float32), 'set-theta___a1': Box(-1.0, 1.0, (1,), float32))
Shape of action space: None

UAV_contnuous:

Action space: Dict('set-acc___a1': Box(-1.0, 1.0, (1,), float32), 'set-phi___a1': Box(-1.0, 1.0, (1,), float32), 'set-theta___a1': Box(-1.0, 1.0, (1,), float32))
Shape of action space: None

Elevators:

Action space: Dict('move-current-dir___e0': Discrete(2), 'move-current-dir___e1': Discrete(2), 'open-door___e0': Discrete(2), 'open-door___e1': Discrete(2), 'close-door___e0': Discrete(2), 'close-door___e1': Discrete(2))
Shape of action space: None

Reservoir_continuous:

Action space: Dict('release___t1': Box(0.0, 100.0, (1,), float32), 'release___t2': Box(0.0, 100.0, (1,), float32), 'release___t3': Box(0.0, 100.0, (1,), float32))
Shape of action space: None

PropDBN:

Action space: Dict('a': Discrete(2))
Shape of action space: None

HVAC: 

Action space: Dict('fan-in___z1': Box(0.05, inf, (1,), float32), 'heat-input___h1': Box(-inf, inf, (1,), float32))
Shape of action space: None




