Note: stable-baselines3 has its own way of handling hyperparameters through the use of policy_kwargs and learning_rate as an argument during the creation of the model. 

Alternative code for agent.py:



class MyRDDLAgent:
    def __init__(self, action_space, num_actions=1, seed=None, env=None):
        self.action_space = action_space
        self.num_actions = num_actions
        self.env = DummyVecEnv([lambda: env])  # PPO requires a vectorized environment
        if seed is not None:
            self.action_space.seed(seed)
            self.env.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # We'll initialize the model in optimize_agent() function.

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

        policy_kwargs = dict(
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=0.95,  # fixed value, add it as another hyperparameter if you like
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            vf_coef=0.5,  # fixed value, add it as another hyperparameter if you like
            max_grad_norm=0.5,  # fixed value, add it as another hyperparameter if you like
            n_epochs=n_epochs,
            clip_range=clip_range,
            clip_range_vf=None,
        )

        # Here we create the PPO model with the suggested hyperparameters
        self.model = PPO('MlpPolicy', self.env, verbose=1, device=self.device, policy_kwargs=policy_kwargs)

        self.model.learn(total_timesteps=10000)

    def sample_action(self, state=None):
        obs = self.env.reset()
        action, _states = self.model.predict(obs, deterministic=True)
        return action
