# AI learns to play Snake by itself

# Summary

Created a self-learning AI for Snake using deep reinforcement learning, particularly Proximal Policy Optimization (PPO). This project served as a valuable learning experience, involving critical research in model selection, reward shaping, hyperparameter tuning, and developing a reusable environment with parallel computing capabilities.

# Objective

This project aims to explore deep-reinforcement learning and apply the knowledge to create my own self-learning AI agent. I chose the [Snake](https://en.wikipedia.org/wiki/Snake_(video_game_genre)) game because it is ["Easy to Learn, Difficult to Master"](https://en.wikipedia.org/wiki/Bushnell%27s_Law). I was curious if it's true for AIs as well. 

# Methodology

## Model Selection

<img src="readme_image/rl_algorithms.png" width="600">

[Image source](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

There are a variety of reinforcement learning (RL) algorithms. Above diagram shows some examples of RL taxonomy. 

### What is model in RL? Model-based vs. Model-free 

In reinforcement learning, the term "model" typically doesn't refer to a learned neural network but to the design of the environment used for simulation and planning. A model in RL a representation of how the RL agent perceives and interacts with the environment. It includes information about the states, actions, rewards, and transition dynamics.

There are two main types of RL - Model-based and Model-free. Here are some differences:

|       Aspect       |           Model-based            |                   Model-free                   |
|:------------------:|:--------------------------------:|:----------------------------------------------:|
|   Learning type    |       Planning using model       |           Learning from exploration            |
|   Learning speed   |              Faster              |                     Slower                     |
| Sample efficiency  |              Higher              |                     Lower                      |
|     Robustness     | Performs well in accurate models | Performs well in a less-understood environment |

Here is an example from [this blog](https://medium.com/the-official-integrate-ai-blog/understanding-reinforcement-learning-93d4e34e5698):

Let's say an RL AI wants to go visit somewhere, but they don't have a map. A model-based AI will keep track of routes it took before and use them to plan the future trip, whereas a model-free AI will just try anything and find the general direction of the trip.

I decided to go with the model-free algorithms for its robustness. 

### Model Options

These are some popular RL algorithm choices when it comes to video games:

1. Deep Q-Network (DQN):
    * DQN is a powerful algorithm, especially when you have a large state space or complex game dynamics. It can handle high-dimensional input data effectively, making it a strong candidate for success in the Snake game.

2. A3C (Asynchronous Advantage Actor-Critic):
    * A3C is particularly useful when you have access to multiple computational resources and can parallelize the training process. This can significantly speed up learning and potentially lead to better results.

3. Proximal Policy Optimization (PPO):
    * PPO is known for its stability and good sample efficiency, making it a strong choice for training agents in various environments. It can also be effective in the Snake game.

I'm running this program in a laptop with a not too powerful CPU (Ryzen 3 3200U). Therefore, I went with PPO for its stability and sample efficiency. I used [Stable Baseline3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)'s PPO implementation because it was easy to use.

## Reward Shaping

The reward system defines the goals for the agent and guides it in learning the desired behavior. Well-balanced rewards are critical in finding the desired outcome. There are a few things to consider when it comes to reward shaping:

### Reward Density (frequency)

Check the below table for the characteristics of sparse and dense rewards:   

|                     |         Sparse Rewards          |    Dense Rewards    |
|:-------------------:|:-------------------------------:|:-------------------:|
| Feedback Frequency  |             Delayed             |      Immediate      |
|   Learning Speed    |             Slower              |       Faster        |
| Encouraged Behavior |           Exploration           |    Exploitation     |
|       Example       | Rewarded at the end of the game | Rewarded every move |
| Potential Downside  |         Uncertain agent         |    Biased agent     |

It's important to find the balance between them. For example, the survival reward should be substantial enough to encourage safe movements, but not so high that the agent does not find apples rewarding. Similarly, the time penalties should be applied in a way that encourages efficient gameplay without overly punishing the snake for exploring and making smart moves.

### Reward Types

I balanced the reward density using immediate and intermediate feedback. Immediate rewards are feedback that require immediate attention, and intermediate rewards are feedback that are meant to give the agent a more general guideline.

Immediate Feedback (in descending priority):
1. Collision (-)
2. Eating apple (+)
3. Survival (+)
4. Exploration (+)

Intermediate Feedback:
1. Distance to apple (+ for closer)
2. Distance to tail (+ for further)

## Agent's Observations

To enable the agent to make connections between rewards and relevant information in the Snake game, it needs access to essential game state information (called 'observation' in stable baseline3). Below is a list some of my considerations. More information can yield more sophisticated behavior, but it can also cause longer training time.

1. Snake head position
2. Snake body positions / tail position
3. Apple position / distance to apple
4. Past moves
5. Map boundaries
6. Movement direction
7. Score (snake length)

<img src="readme_image/memory_comparison_length.png" height="200"> <img src="readme_image/memory_comparison_reward.png" height="200">

I had to experiment how adding or removing each information affects the agent's behavior. For example, above is the result from playing around with the number of past moves an agent can remember. The left graphs is the average game runtime, and the right graph is the average reward in a game with 20x20 grid. 

The agent starts the game with a lot of exploration, illustrated by the big spike in the game length graph. However, the agent does not gain much reward from this stage because I did not reward exploration during this test. Eventually, around 1M steps in, the exploration is over and the average steps per game converges to around 130, while the reward starts to slowly increase, converging to 1700 rewards per game (score of 15). 

I tested 5, 20, 70 and 200 recent moves, and ran each of them for about 1.5M steps (I ran 200-moves one 6M steps just in case). Surprisingly, the number of recent moves did not change the agent's learning behavior too much. Possible explanations are:

1. I didn't give the agents enough time to learn despite the change in agent complexity.
2. The 'past moves' information was not important in learning.
3. The reward structure did not represent the goal of the game.

More testing is required to understand this behavior. 

## Hyperparameter Tuning

PPO has quite a few hyperparameters, and each of them affect the agent's behavior differently. Check [stable baseline3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)'s implementation, too. Below table explains some important hyperparameters:

| Hyperparameter                  | Description                                                     | Effect of Change                                                                        | Common Range/Value                |
|---------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------|-----------------------------------|
| **Learning Rate (lr)**          | Step size for policy optimization.                              | Influences the speed of learning.                                                       | 1e-5 to 1e-3                      |
| **Clip Parameter (clip_epsilon)** | Maximum allowed policy change ratio.                            | Stabilizes training by limiting policy updates.                                         | 0.2                               |
| **Number of Epochs (n_epochs)** | Number of times data is reused for updates.                     | More epochs can lead to more stable policy updates.                                     | 1 to 10                           |
| **Batch Size (batch_size)**     | Number of samples in each policy update.                        | Larger batch sizes may yield more accurate gradient estimates.                          | Varies                            |
| **Value Function Coefficient (vf_coef)** | Weight of the value function loss.                              | Controls the balance between policy and value updates.                                  | 0.5                               |
| **Entropy Coefficient (ent_coef)** | Weight of the entropy term.                                     | Influences the level of exploration in the policy.                                      | 0.01                              |
| **Discount Factor (gamma)**     | Trade-off between immediate and future rewards.                 | Higher value favors long-term rewards.                                                  | 0.99                              |
| **GAE Lambda (gae_lambda)**     | Controls weight of accumulated rewards in advantage estimation. | Adjusts the impact of the GAE calculation on the advantage.                             | 0.95                              |
| **Number of Parallel Environments** | Number of parallel workers (if used).                           | Impacts training speed and stability.                                                   | Varies. I went with 4.            |
| **Network Architecture**        | Design of policy and value function networks.                   | Influences the complexity of the model and its ability to capture patterns in the data. | Varies. I went with MlpPolicy.    |
| **Optimization Algorithm**      | Choice of optimization algorithm (ex., Adam, RMSprop).          | Affects convergence speed and stability.                                                | Varies. Adam in stable baseline3. |

### Learning Rate

<img src="readme_image/lr_comparison.png" height="200">

I played around with learning rate to observe its effect on agent's behavior. Green is lr=0.00003 and Red is lr=0.003. As expected, higher learning rate yielded a faster result. If Green was given enough time, it might have yielded a better result than Red. However, since time and performance is a trade-off, Green is too slow. 

### Entropy Coefficient

<img src="readme_image/ent_coeff_comparison.png" height="200">

I played around with entropy coefficient to observe its effect on agent's behavior. Orange is ent_coeff=0.001 and Gray is ent_coeff=0.1. I thought higher entropy coefficient would encourage exploration, but I was quite the opposite for this test. Maybe 0.1 was too much encouragement for randomness, that the Gray agent's decisions were contaminated by noises.

# Result

## Evaluation Criteria

There is an infallible non-RL solution for Snake game. The snake can simply follow a Hamiltonian Cycle, a cyclic path that passes every square on the grid without crossing itself. This is a trivial solution, but it can be a good reference to measure my snake agent's performance. 

I'll calculate the expected average steps per completed game. I'll assume that the average steps required to eat apple at any point is half of the number of remaining unoccupied cells.

Let's define some variables first:
* N = the grid size
* L = the snake's length
* m = the snake's starting length, where m < N<sup>2</sup> 

Then, the average total steps per game would be:
> $$\displaystyle \lim_{L \to \infty} \int_{m}^{N^2} \frac{N^2-L}{2} dL$$

Solving it gives:
> $$\frac{N^4}{4}$$

I double-checked this approach by calculating them manually (step_test.py). I found that this equation a bad estimator for smaller maps (N < 7) where it gives accuracy under 0.90. However, the accuracy is over 0.95 for N > 10 and over 0.99 for N > 23, which is good to use.

## Evaluation

# Discussion

* What I achieved
* Limitations
* Future works
