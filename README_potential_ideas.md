
Core Idea = PPO + some model-based algo + auto-tuning


# 1. PPO
- Proposed by OpenAI
- An on-policy algorithm - improvement of policy gradient methods
- Does not require a value function approximation, instead, it optimizes the policy directly.
- Model-free RL algo - does not need a model of the environment to learn a policy.
- PPO can be used in environments with discrete or continuous action spaces, and it's generally easier to apply because it doesn't require designing a cost function like LQR or MPC.


# Why not LQR and MPC? (Background of Research Paper)
- LQR (Linear Quadratic Regulator) and MPC (Model Predictive Control) are model-based methods that require a good model of the environment. 
- LQR assumes that the dynamics of the environment are linear and the cost function is quadratic - also assumes that the system is fully observable, which might not be the case always.
- MPC is more flexible than LQR and can handle non-linear dynamics and non-quadratic cost functions. However, MPC needs to solve an optimization problem at each step, which can be computationally expensive. Also, the performance of MPC depends heavily on the quality of the model and the cost function.
