# AI learns to play Snake by itself

# Summary

Created a self-learning AI for Snake using deep reinforcement learning, particularly Proximal Policy Optimization (PPO). This project served as a valuable learning experience, involving critical research in model selection, reward shaping, hyperparameter tuning, and developing a reusable environment with parallel computing capabilities.

# Objective

This project aims to explore deep-reinforcement learning and apply the knowledge to create my own self-learning AI agent. I chose the [Snake](https://en.wikipedia.org/wiki/Snake_(video_game_genre)) game because it is "Easy to Learn, Difficult to Master" (a.k.a [Bushnell's Law](https://en.wikipedia.org/wiki/Bushnell%27s_Law)). I was curious if it's true for AIs as well. 

# Methodology

## Model Selection

<img src="readme_image/rl_algorithms.png" width="600">
[Image source](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

There are a variety of reinforcement learning (RL) algorithms. Above diagram shows some examples of RL taxonomy. Here is my understanding of each category:

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

I'm running this program in a laptop with a not too powerful CPU (Ryzen 3 3200U). Therefore, I went with PPO for its stability and sample efficiency. 

## Reward Shaping

The reward system defines the goals for the agent and guides it in learning the desired behavior. Well-balanced rewards are critical in finding the desired outcome. There are a few things to consider when it comes to reward shaping:

### Reward Density (frequency)

Check the below table for the characteristics of sparse and dense rewards:   

|                     |             Sparse              |         Dense          |
|:-------------------:|:-------------------------------:|:----------------------:|
| Feedback Frequency  |             Delayed             |       Immediate        |
|   Learning Speed    |             Slower              |         Faster         |
| Encouraged Behavior |           Exploration           |      Exploitation      |
|       Example       | Rewarded at the end of the game |  Rewarded every move   |
| Potential Downside  |    Uncertain decision-making    | Biased decision-making |

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

## hyperparameter tuning

## libraries

developing a reusable environment with parallel computing capabilities.


# Result

* Evaluation

# Discussion

* What i achieved
* Limitations
* Future works
