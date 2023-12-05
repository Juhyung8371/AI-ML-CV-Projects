# AI learns to play Snake by itself

# <ins>Summary</ins>

Created a self-learning AI for Snake using deep reinforcement learning, particularly Proximal Policy Optimization (PPO). This project served as a valuable learning experience, involving critical research in model selection, reward shaping, hyperparameter tuning, and developing a reusable environment with parallel computing capabilities.

# <ins>Objective</ins>

This project aims to explore deep-reinforcement learning and apply the knowledge to create my own self-learning AI agent. I chose the [Snake](https://en.wikipedia.org/wiki/Snake_(video_game_genre)) game because it is easy to learn, but difficult to master. I was curious if it's true for AIs as well. 

# <ins>Methodology</ins>

## Model Selection

<img src="readme_image/rl_algorithms.png" width="600">

[Image source](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

There are a variety of reinforcement learning (RL) algorithms. Above diagram shows some examples of RL taxonomy. 

### *What is model in RL? Model-based vs. Model-free*

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

### *Model Options*

These are some popular RL algorithm choices when it comes to video games:

1. Deep Q-Network (DQN):
    * DQN is a powerful algorithm, especially when you have a large state space or complex game dynamics. It can handle high-dimensional input data effectively, making it a strong candidate for success in the Snake game.

2. A3C (Asynchronous Advantage Actor-Critic):
    * A3C is particularly useful when you have access to multiple computational resources and can parallelize the training process. This can significantly speed up learning and potentially lead to better results.

3. Proximal Policy Optimization (PPO):
    * PPO is known for its stability and good sample efficiency, making it a strong choice for training agents in various environments. It can also be effective in the Snake game.

I'm running this program in a laptop with a not too powerful CPU (Ryzen 3 3200U). Therefore, I went with PPO for its stability and sample efficiency. I used [Stable Baseline3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)'s PPO implementation because it was easy to use.

## Reward Shaping

The reward system defines the goals for the agent and guides it in learning the desired behavior. Well-designed rewards are critical in finding the desired outcome. There are quite a few things to consider here, which makes reward shaping a challenging task.

### *Reward Density*

|                     |             Sparse Rewards             |    Dense Rewards    |
|:-------------------:|:--------------------------------------:|:-------------------:|
| Feedback Frequency  |                Delayed                 |      Immediate      |
|   Learning Speed    |                 Slower                 |       Faster        |
| Encouraged Behavior |              Exploration               |    Exploitation     |
|       Example       | -1 for game over and +1 for game clear | Rewarded every move |
| Potential Downside  |            Uncertain agent             |    Biased agent     |

The reward density is one of the ways to classify the reward types. It's important to balance the sparse and dense rewards for their unique characteristics. Sparse rewards are intuitive and easy to balance, but it's often not enough feedback for the agent to learn. While dense rewards can fill the gap with intermediate rewards and yield a faster learning outcome, they are prone to human bias and overfitting [[2]](#references). Paper [[1]](#references) suggests that one should "*specify how to measure outcomes, not how to achieve them*" to produce more robust outcome. For example, let's say I'm training an RL agent to drive a car safely from point A to B, and I gave it a reward for driving straight since it's one of the safe driving method. Unfortunately, this agent might hit a jaywalking person since it's trained to only go straight. In this case, a safe driving measure will be a better feedback than encouraging explicit behavior that the programmer thought was correct. 

### *Reward Function Sanity Check*

|     | Sanity check failures                                           | Brief explanation                                                                                                                                                             | Potential intervention(s)                                                                                                                                                                   |
|-----|:----------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1   | Unsafe reward shaping                                           | If reward includes guidance on behavior that deviates from only measuring desired outcomes, reward shaping exists.                                                            | Separately define the true reward function and any shaping reward. Report both true return and shaped return. Change it to an applicable safe reward shaping method. Remove reward shaping. | 
| 2   | Mismatch in people’s and reward function’s preference orderings | If there is human consensus that one trajectory is better than another, the reward function should agree.                                                                     | Change the reward function to align its preferences with human consensus.                                                                                                                   |
| 3   | Undesired risk tolerance via indifference points                | Assess a reward function's risk tolerance via indifference points and compare to a human-derived acceptable risk tolerance.                                                   | Change reward function to align its risk tolerance with human-derived level.                                                                                                                |
| 4   | Learnable loophole(s)                                           | If learned policies show a pattern of undesirable behavior, consider whether it is explicitly encouraged by reward.                                                           | Remove encouragement of the loophole(s) from the reward function.                                                                                                                           |
| 5   | Missing attribute(s)                                            | If desired outcomes are not part of reward function, it is indifferent to them.                                                                                               | Add missing attribute(s).                                                                                                                                                                   |
| 6   | Redundant attribute(s)                                          | Two or more reward function attributes include measurements of the same outcome.                                                                                              | Eliminate redundancy.                                                                                                                                                                       |
| 7   | Trial-and-error reward design                                   | Tuning the reward function to improve RL agents' performances has unexamined consequences.                                                                                    | Only use observations of behavior to improve the reward function's measurement of task outcomes or to tune separately defined shaping reward.                                               |
| 8   | Incomplete description of problem specification                 | Missing descriptions of reward function, termination conditions, discount factor, or time step duration may indicate insufficient consideration of the problem specification. | In research publications, write the full problem specification and why it was chosen. The process might reveal issues.                                                                      |

Above table is the reward function sanity check table from paper [[1]](#references). It says, "Each sanity check is described by what problematic characteristic to look for. Failure of the first 5 sanity checks identifies problems with the reward function; failure of the last 3 checks should be considered a warning." Using this sanity check can help avoiding [premature convergence](https://machinelearningmastery.com/premature-convergence/), which refers to the situation where the agent is stuck at a suboptimal solution. It was easy for me to go down the rabbit hole of keep adding more heuristics to the agent in attempt to improve it, so I referenced this sanity check to keep my reward function in check. 

### *Exploration vs. Exploitation*

We want our agent to make the most informed decisions, and that information can be gained from exploration. To encourage exploration, we have to make it rewarding. One way to achieve that is implementing intrinsic rewards, also known as curiosity rewards, such as rewards for new discoveries and improved knowledge. See [OpenAI page](https://openai.com/research/reinforcement-learning-with-prediction-based-rewards) and [blog page](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/) for much more in-depth explanation of exploration in RL. I implemented curiosity reward by adding a memory to the agent - rewarding the agent for finding different states of the game. 

Exploitation is not always a bad thing. It means the agent found a good plan to maximize the reward. However, it becomes a problem when the reward structure is mis-designed. One of the example of this is 'the noisy-tv problem.' It happens when the agent finds a source of randomness in the environment and continuously harvest intrinsic rewards, without getting closer to the ultimate goal. This problem is mentioned in the 4th point of the sanity check table. In my curiosity reward system, the snake may end up circling at a corner for this reason. So, I adjusted the state difference threshold to balance what is considered a new game state.

<img src='readme_image/noisy_tv.gif'>

[The noisy-tv problem](https://openai.com/research/reinforcement-learning-with-prediction-based-rewards)

### *Reward Types*

[//]: # (priority --------------------------------)

[//]: # (1. Survival)

[//]: # ()
[//]: # (2. Safe moves)

[//]: # (   1. Able to reach tail )

[//]: # (   2. Not make void )

[//]: # (   3. Fill void seeds &#40;1 box surrounded on 3 sides&#41;)

[//]: # (3. Eat apple)

[//]: # (4. Exploration)


Immediate Feedback:
* Collision penalty
* Eating apple reward

Intermediate Feedback:
* Exploration reward
* Creating void (usuable spots in the grid) penalty

In reward design, I've balanced feedback density with immediate rewards tied to game outcomes (ex: eating the apple, colliding with the wall) and intermediate feedback for general guidance, like exploration and inefficient move feedback. While I considered additional rewards (ex: survival, proximity to the apple, distance from the tail), I limited frequent feedback to avoid premature convergence. For instance, frequent survival rewards can discourage exploration despite encouraging safe movements.

### *Dynamic Rewards*

|                         Dynamic Reward Test                          |                                                                      | 
|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|
| <img src="readme_image/dynamic_reward_comparison1.png" height="200"> | <img src="readme_image/dynamic_reward_comparison2.png" height="200"> |
|               Red: dynamic reward / Blue: fixed reward               |       Left: Game length / Middle: Total reward /  Right: Score       |

Like many other video games, Snake game progressively gets more challenging as you play, so adjusting the feedback becomes crucial in reflecting the real game play experience. I noticed that the priority for making safe move gets increasingly important as the snake gets longer. Based on that, I implemented a dynamic reward and penalty system.

It starts with a small safe-move reward in the early game to encourage/greedy behavior while the risk of collision is lower. As the game progresses, the gradual increment of safe-move reward should teach the agent that it's better to be safe than sorry. Also, this increment of overall reward should positively affect the learning behavior as demonstrated in paper [[3]](#references). It says, "when defining subgoal rewards, it helps to gradually increase rewards as the agent gets closer to the goal state. This design helps counteract the effect of discounting, but also continually spurs the agent forward, much like an annual salary raise is considered to be a good motivator in the commercial sector."

Above charts show this system in action. Although agent with fixed reward got more reward, but the agent with dynamix reward survived longer and scored higher.

### *Human Bias*

|                Inefficient Move Penalty Test                 |
|:------------------------------------------------------------:|
| <img src="readme_image/penalty_comparison.png" height="200"> |
|          Green: without penalty / Red: with penalty          |
|           Left: Game length / Right: Total reward            |

Giving meaningful feedback can be hard because it often requires a deep understanding of the environment, which is not always possible. For instance, I tested how penalizing the agent for making 'bad' moves (getting trapped, repeating meaningless moves) affects the agent. Interestingly, as shown the charts above, agents performed better without penalties in many cases. This result is also in line with the finding from [[3]](#references), "penalizing each step taken induces faster learning than rewarding the goal" - excess penalties encouraged a hasty learning behavior. 

I learned that reward shaping is particularly prone to human bias and difficult to balance. After all, reinforcement learning is an optimization algorithm, not a behaviour-copying algorithm. Therefore, The RL agent will do the best in the given environment, but if the environment does not reflect the goal properly, then the agent is bound to fail. That's why I also focused on providing more and better quality information to the agent, so it can make more intelligent decision. 

## Agent's Observations (Input)

For the agent to make connections between rewards and actions, it needs access to essential game state information (called 'observation' in Stable Baseline3). More information can yield more sophisticated behavior, but it can also cause longer training time. 

### *Quality over Quantity*

<img src="readme_image/memory_comparison_length.png" height="200"> <img src="readme_image/memory_comparison_reward.png" height="200">

I experimented how quality and quantity of information affect the agent's behavior. For example, above is the result from playing around with the number of past moves from the current round an agent can reference. The left graphs is the average steps per game, and the right graph is the average reward per game. I tested 5, 20, 70 and 200 recent moves, and ran each of them for about 1.5M steps. I let 200-moves one learn longer to compensate the decreased rate from more complex information. Surprisingly, the number of recent moves did not change the agent's learning behavior and performance too much. They all have similar trends overall. The reward converged to 1700 per game, which is only around 15 out of 400 points in this environment. 

Possible explanations for this behavior are:

1. The 'past moves' information was not important in learning due to its poor information quality.
2. The reward structure did not represent the goal of the game.

Since I confirmed that I can't just chuck everything at the agent and expect it to do better, I needed to feed it quality information. 

| <img src="readme_image/square_view.png" height="200"> | <img src="readme_image/line_view.png" height="200"> |
|:-----------------------------------------------------:|-----------------------------------------------------|
|                      Square View                      | Line View                                           |

I started by giving view information to the agent. I tried two types: square view and line view around the snake's head. In the images above, green tiles are the snake, black tiles are the wall, red tile is the apple, and the gray tiles are the tile that the agent can see. The agent is given the type of the tile (wall, empty, snake, and apple) and distance to that tile. Also, like straight, left, and right actions, the view is also rotated based on the direction of the head, so it's more consistent (ex. the view is rotated clockwise if the snake is looking left). Unfortunately, adding this kind of view information wasn't effective as the agent failed to improve beyond score 10 because it couldn't figure out which move can lead to a dead end. It might need more sophisticated processing like CNN to truly take advantage of view information. However, I was able to use this view information to implement curiosity reward. I took the view information and compared it with others in the past. If it's different enough, then the agent gets an exploration reward. This will encourage the agent to explore the grid. 

|   <img src="readme_image/future_observations.png" height="200">   |
|:-----------------------------------------------------------------:|
|                          Score over time                          |
| Orange: can only see current board, Gray: can see on 1 step ahead |

Then, instead of waiting for the agent to figure out rule from scratch visually, I gave it a nudge by allowing the agent to look 1 step into the future. I told it which actions will lead to collision, lead to the largest open area, create voids, eliminate gaps that can become a void, cut access to the tail, etc. I could add more complex heuristics to produce better outcome, however, it is like an endless wack-a-mole game at that point. As shown in the graph above, this implementation increased the score from 10 to 120.










### *Improving Data Quality with Data Preprocessing*

Like any machine learning process, reinforcement learning benefits from data preprocessing. In my case, data normalization and one-hot encoding was particularly useful.
* Normalization is a fundamental preprocessing step in machine learning that contributes to the stability, efficiency, and generalization capabilities of models, making them more robust and suitable for a wide range of applications.
* One-hot encoding is a technique used in data preprocessing to represent categorical variables as binary vectors. It can prevent model bias from ordinal assumptions (ex. thinking that 'red' is greater than 'blue' because it comes first) and improve performance.

A list of information I fed to the agent:

1. Normalized snake head position
2. Normalized Apple position
3. One-hot encoded snake head direction
4. One-hot encoded direction to apple 
5. Normalized current score
6. Evaluation of future actions
   * A boolean 'collision' to check which future actions can cause a collision.
   * A boolean 'visited' to check which future actions lead the snake to a cell that's been visited already. Hopefully, the agent realizes that re-visiting a cell is a not too efficient move.
   * A boolean 'void created' to check which future actions can create voids (unreachable spots) in the grid. Creating unreachable voids is a fatal move because it restricts snake's movement.
7. Evaluation of current body positions
   * A boolean 'trapped' info to check if the snake head is trapped in a void. 

## Hyperparameter Tuning

| Hyperparameter                           | Description                                                     | Effect of Change                                                                        | Common Range/Value                               |
|------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------|--------------------------------------------------|
| **Learning Rate (lr)**                   | Step size for policy optimization.                              | Influences the speed of learning.                                                       | 1e-5 to 1e-3. I went with 3e-4.                  |
| **Clip Parameter (clip_epsilon)**        | Maximum allowed policy change ratio.                            | Stabilizes training by limiting policy updates.                                         | 0.2                                              |
| **Number of Epochs (n_epochs)**          | Number of times data is reused for updates.                     | More epochs can lead to more stable policy updates.                                     | 1 to 10                                          |
| **Batch Size (batch_size)**              | Number of samples in each policy update.                        | Larger batch sizes may yield more accurate gradient estimates.                          | Varies. I went with 256.                         |
| **Value Function Coefficient (vf_coef)** | Weight of the value function loss.                              | Controls the balance between policy and value updates.                                  | 0.5                                              |
| **Entropy Coefficient (ent_coef)**       | Weight of the entropy term.                                     | Influences the level of exploration in the policy.                                      | 0.01. I went with 0.001.                         |
| **Discount Factor (gamma)**              | Trade-off between immediate and future rewards.                 | Higher value favors long-term rewards.                                                  | 0.99                                             |
| **GAE Lambda (gae_lambda)**              | Controls weight of accumulated rewards in advantage estimation. | Adjusts the impact of the GAE calculation on the advantage.                             | 0.95                                             |
| **Number of Parallel Environments**      | Number of parallel workers (if used).                           | Impacts training speed and stability.                                                   | Varies. I went with 4 due to the hardware limit. |
| **Network Architecture**                 | Design of policy and value function networks.                   | Influences the complexity of the model and its ability to capture patterns in the data. | Varies. I went with MlpPolicy.                   |
| **Optimization Algorithm**               | Choice of optimization algorithm (ex., Adam, RMSprop).          | Affects convergence speed and stability.                                                | Varies. Adam in Stable Baseline3's PPO.          |

PPO has quite a few hyperparameters, and each of them affect the agent's behavior differently. Check [Stable Baseline3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)'s implementation, too. Above table explains some important hyperparameters. I played with hyperparameters and here are some experiment results:

### *Learning Rate*

<img src="readme_image/lr_comparison.png" height="200">

I played around with learning rate to observe its effect on agent's behavior. Green is lr=0.00003 and Red is lr=0.003. As expected, higher learning rate yielded a faster result. If Green was given enough time, it might have yielded a better result than Red. However, since time and performance is a trade-off, Green can be less practical. 

### *Entropy Coefficient*

<img src="readme_image/ent_coeff_comparison.png" height="200">

I played around with entropy coefficient to observe its effect on agent's behavior. Orange is ent_coeff=0.001 and Gray is ent_coeff=0.1. I thought higher entropy coefficient would encourage exploration, but it was quite the opposite for this test. Maybe 0.1 was too much randomness that the Gray agent's decisions were contaminated by noises.

# <ins>Result</ins>

## Evaluation Criteria

There is an infallible non-RL solution for Snake game. The snake can simply follow a Hamiltonian Cycle, a cyclic path that passes every square on the grid without crossing itself. This is a trivial solution, but it can be a good reference to measure my snake agent's performance. If the snake travel the entire grid for each apple, then the total steps for a completed game can be the grid-area squared (160000 steps for 20x20 grid). I can make some assumptions to make this number more realistic. I'll assume that the average steps required to eat apple at any point is half of the number of remaining unoccupied cells.

Let's define some variables first:
* N = the grid size
* L = the snake's length
* m = the snake's starting length, where 0 < m < N<sup>2</sup> 

Then, the average total steps per game would be:
> $$\displaystyle \lim_{L \to \infty} \int_{m}^{N^2} \frac{N^2-L}{2} dL$$

Solving it gives:
> $$\frac{N^4}{4}$$

The expected average game length to finish the 20x20 game is 40000 steps.

## Evaluation






# <ins>Discussion</ins>

* What I achieved
* Limitations 

## Future works

* I can provide the entire game screen information to the agent using a convolutional neural network. The learning time can get longer due to the increased information, but it might save me from manually making complex strategy for the agent. Then, I can compare the learning outcome from manual information and visual information.
* I can try more complex memory system. For instance, a system like LSTM can allow the agent to learn fine techniques from every game and general strategy from multiple runs. 
* I can implement auto-hyperparameter tuning feature to find better options faster.


# <ins>References</ins>


1. Knox, W. B., Allievi, A., Banzhaf, H., Schmitt, F., & Stone, P. (2023). Reward (Mis)design for autonomous driving. Artificial Intelligence, 316, 103829. https://doi.org/10.1016/j.artint.2022.103829
2. Booth, S., Knox, W. B., Shah, J., Niekum, S., Stone, P., & Allievi, A. (2023). The Perils of Trial-and-Error Reward Design: Misdesign through Overfitting and Invalid Task Specifications. Proceedings of the AAAI Conference on Artificial Intelligence, 37(5), 5920-5929. https://doi.org/10.1609/aaai.v37i5.25733
3. Sowerby, H., Zhou, Z., & Littman, M. L. (2022). Designing rewards for fast learning. 
https://doi.org/10.48550/arXiv.2205.15400
