
Core Idea = PPO + some model-based algo + auto-tuning


# 1. PPO
- Proposed by OpenAI
- An on-policy algorithm - improvement of policy gradient methods
- Does not require a value function approximation, instead, it optimizes the policy directly.
- Model-free RL algo - does not need a model of the environment to learn a policy.
- PPO can be used in environments with discrete or continuous action spaces, and it's generally easier to apply because it doesn't require designing a cost function like LQR or MPC.



