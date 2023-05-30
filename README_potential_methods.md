
Core Idea = PPO + some model-based algo + auto-tuning (incorporate TPEPruner/SuccessiveHalvingPruner to prevent early stopping?)



1. Hybrid: PPO for the high-level decision-making, but incorporate elements of model-based planning for lower-level tasks.
2. Automatic Hyperparameter Tuning: Optuna/ray-tuning?
3. Transfer Learning and Meta-Learning: Rather than training a new agent from scratch for each domain- transfer knowledge from one domain to another. Example: an agent might learn a general strategy in one domain that could be applied or fine-tuned in a different domain. Research in this area could be quite novel and impactful. (Muy interesante!)
4. Learning Domain-Independent Features: learn a set of features or representations that capture the essential dynamics of a wide range of domains. This might involve developing novel network architectures or learning algorithms. (Might be future work!)
5. Understanding and Improving Exploration: Studying the role of exploration in domain-independent planning and developing methods to improve it.



Stable-baselines3 PPO
![image](https://github.com/DrishtiShrrrma/domain-independent-planner-v1/assets/129742046/4b8652a5-ad69-459c-a01f-bb053c822475)
