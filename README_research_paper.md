
Core Idea = PPO + some model-based algo + auto-tuning?


# 1. PPO
- Proposed by OpenAI
- Proximal Policy Optimization (PPO) is an Actor-Critic method. Actor-Critic system has two models: the Actor and the Critic. The Actor corresponds to the policy π and is used to choose the action for the agent and update the policy network.
- An on-policy algorithm - improvement of policy gradient methods - combines the benefits of both on-policy and off-policy algorithms (CONTRADICTORY - check!)
- Does not require a value function approximation, instead, it optimizes the policy directly.
- Model-free RL algo - does not need a model of the environment to learn a policy.
- PPO can be used in environments with discrete or continuous action spaces, and it's generally easier to apply because it doesn't require designing a cost function like LQR or MPC.
- 


# Why not LQR and MPC? (Can easily go in Background Section)
- LQR (Linear Quadratic Regulator) and MPC (Model Predictive Control) are model-based methods that require a good model of the environment. 
- LQR assumes that the dynamics of the environment are linear and the cost function is quadratic - also assumes that the system is fully observable, which might not be the case always.
- MPC is more flexible than LQR and can handle non-linear dynamics and non-quadratic cost functions. However, MPC needs to solve an optimization problem at each step, which can be computationally expensive. Also, the performance of MPC depends heavily on the quality of the model and the cost function.

Given the diverse set of environments we have, a model-free RL algorithm like PPO might be more suitable because it doesn't require a model of the environment and can handle both discrete and continuous actions. 


# 2. PPO Vs A3C

**PPO :**
- Relatively simple to implement and tune, and has fewer hyperparameters compared to A3C.
- It is sample-efficient and often produces stable results without the need for finely tuned hyperparameters.
- It typically requires less parallel computation resources than A3C.

**A3C :**
- A3C can potentially learn faster than PPO due to its parallel architecture, as it learns from multiple different environments simultaneously.
- It often requires careful tuning of hyperparameters and the balancing of multiple parallel environments to get optimal results.
- **A3C often results in less stable training than PPO, as updates from different environments can interfere with each other - hyperparameter tuning could make A3C a more viable choice in some cases


# 3. PPO-GAE
- Proximal Policy Optimization with Generalized Advantage Estimation (PPO-GAE)
- PPO-GAE combines the advantages of policy optimization (PPO) with value estimation (Actor-Critic) to achieve more stable and efficient training.
- In PPO-GAE, the value function is used to estimate the advantage values, which are then used to update the policy. This combination helps in reducing variance and improving the learning process.
- In GAE, there's a hyperparameter named lambda (λ) which is used for trading-off bias vs variance in advantage estimation. This hyperparameter typically varies between 0 and 1. A value of 0 leads to high bias/low variance (one-step TD) estimates, and a value of 1 leads to low bias/high variance (Monte Carlo) estimates.
- gae_lambda represents the trade-off between bias and variance for Generalized Advantage Estimation.

Note : stable_baselines3's PPO implementation already includes GAE. So unless one needs to modify the GAE calculation, one should be fine with the basic implementation of PPO. If we do need to modify GAE, then we will have to write your own custom environment and agent that allows for the necessary modifications.

# 4. PPO-GAE with Exploration 😍
