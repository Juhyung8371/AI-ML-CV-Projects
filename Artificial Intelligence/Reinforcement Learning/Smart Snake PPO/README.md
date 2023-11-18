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

The reward system defines the goals for the agent and guides it in learning the desired behavior. Well-balanced rewards are critical in finding the desired outcome. There are quite a few things to consider here.

### *Reward Density (Feedback Frequency)*

|                     |         Sparse Rewards          |    Dense Rewards    |
|:-------------------:|:-------------------------------:|:-------------------:|
| Feedback Frequency  |             Delayed             |      Immediate      |
|   Learning Speed    |             Slower              |       Faster        |
| Encouraged Behavior |           Exploration           |    Exploitation     |
|       Example       | Rewarded at the end of the game | Rewarded every move |
| Potential Downside  |         Uncertain agent         |    Biased agent     |

The reward density is one of the ways to classify the reward types. It's important to balance the sparse and dense rewards because they have unique pros and cons. For example, if the agent is rewarded only when it finishes the snake game (completely fill up the grid with the snake), then the agent might never realize the ultimate goal because finishing the game requires very calculated series of moves. On the other hand, if the human micromanages the agent by doing all the thinking for the agent, then not only it can introduce human bias, but it also diminishes the purpose of using reinforcement learning. 

### *Premature Convergence*

We need to consider is avoiding [premature convergence](https://machinelearningmastery.com/premature-convergence/). It refers to the situation where the agent is stuck at a suboptimal solution and fail to improve further. One of its main cause is the lack of exploration. The agent may refuse to explore and learn new things if it's not worth it - either the exploration is too risky or the exploitation is too good. 

### *Exploration vs. Exploitation*

We want our agent to make the most informed decisions, and that information is gained from exploration. To encourage exploration, we have to make it rewarding. One way to achieve that is implementing intrinsic rewards, also known as curiosity rewards, such as rewards for new discoveries and improved knowledge. See these [OpenAI page](https://openai.com/research/reinforcement-learning-with-prediction-based-rewards) and [blog page](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/) for much more in-depth explanation of exploration in RL. 

Exploitation is not always a bad thing. It means the agent found a good plan to maximize the reward. However, it becomes a problem when the reward structure is exploited. One of the example of this is 'the noisy-tv problem.' It happens when the agent finds a source of randomness in the environment and continuously harvest intrinsic rewards, without getting closer to the ultimate goal. 

<img src='readme_image/noisy_tv.gif'>

[The noisy-tv problem](https://openai.com/research/reinforcement-learning-with-prediction-based-rewards)

### *Reward Types*

Immediate Feedback:
* Collision penalty
* Eating apple reward

Intermediate Feedback:
* Exploration reward
* Creating void (usuable spots in the grid) penalty

In reward design, I've balanced feedback density with immediate rewards tied to game outcomes (ex: eating the apple, colliding with the wall) and intermediate feedback for general guidance, like exploration and inefficient move feedback. While I considered additional rewards (ex: survival, proximity to the apple, distance from the tail), I limited frequent feedback to avoid premature convergence. For instance, frequent survival rewards can discourage exploration despite encouraging safe movements.

|                         Dynamic Reward Test                          |                                                                      | 
|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|
| <img src="readme_image/dynamic_reward_comparison1.png" height="200"> | <img src="readme_image/dynamic_reward_comparison2.png" height="200"> |
|               Red: dynamic reward / Blue: fixed reward               |       Left: Game length / Middle: Total reward /  Right: Score       |

Like many other video games, Snake game progressively gets more challenging as you play, so adjusting the feedback amount becomes crucial in reflecting the real game play experience. I implemented a dynamic reward and penalty system, starting with small collision penalties and apple rewards in the early game. This emphasizes exploration behavior early on, with the importance of apple and collision feedback gradually increasing as the game progresses. Above charts show this system in action. Although agent with fixed reward got more reward, but the agent with dynamix reward survived longer and scored higher.

|                Inefficient Move Penalty Test                 |
|:------------------------------------------------------------:|
| <img src="readme_image/penalty_comparison.png" height="200"> |
|          Green: without penalty / Red: with penalty          |
|           Left: Game length / Right: Total reward            |

I found that penalty can often lead to less optimal outcome. For instance, I tested how penalizing the agent for making efficient moves (getting trapped, repeating meaningless moves) affects the agent. Interestingly, as shown the charts above, agents performed better without penalties in many cases. I learned that reward shaping is particularly prone to human bias. That's why I focused on providing more and better quality information to the agent instead of feedback, so the agent can make more intelligent decision. 


## Agent's Observations (Input)

For the agent to make connections between rewards and actions, it needs access to essential game state information (called 'observation' in Stable Baseline3). More information can yield more sophisticated behavior, but it can also cause longer training time. 

### *Quality over Quantity*

<img src="readme_image/memory_comparison_length.png" height="200"> <img src="readme_image/memory_comparison_reward.png" height="200">

I experimented how quality and quantity of information affect the agent's behavior. For example, above is the result from playing around with the number of past moves from the current round an agent can reference. The left graphs is the average game runtime, and the right graph is the average reward.

The agent starts the game with a lot of exploration, illustrated by the big spike in the game length graph. However, the agent does not gain much reward from this stage because I did not reward exploration during this test. Eventually, around 1M steps in, the exploration is over and the average steps per game converges to around 130, while the reward starts to slowly increase, converging to 1700 rewards per game (score of 15 in this reward system). 

I tested 5, 20, 70 and 200 recent moves, and ran each of them for about 1.5M steps . I let 200-moves one learn longer to compensate the decreased rate from more complex information. Surprisingly, the number of recent moves did not change the agent's learning behavior and performance too much. Possible explanations are:

1. The 'past moves' information was not important in learning due to its poor information quality.
2. The reward structure did not represent the goal of the game.

Since I confirmed that can't just chuck everything at the agent and expect it to be better, I needed to feed it quality information. 

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

| Hyperparameter                  | Description                                                     | Effect of Change                                                                        | Common Range/Value                               |
|---------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------|--------------------------------------------------|
| **Learning Rate (lr)**          | Step size for policy optimization.                              | Influences the speed of learning.                                                       | 1e-5 to 1e-3. I went with 3e-4.                  |
| **Clip Parameter (clip_epsilon)** | Maximum allowed policy change ratio.                            | Stabilizes training by limiting policy updates.                                         | 0.2                                              |
| **Number of Epochs (n_epochs)** | Number of times data is reused for updates.                     | More epochs can lead to more stable policy updates.                                     | 1 to 10                                          |
| **Batch Size (batch_size)**     | Number of samples in each policy update.                        | Larger batch sizes may yield more accurate gradient estimates.                          | Varies. I went with 256.                         |
| **Value Function Coefficient (vf_coef)** | Weight of the value function loss.                              | Controls the balance between policy and value updates.                                  | 0.5                                              |
| **Entropy Coefficient (ent_coef)** | Weight of the entropy term.                                     | Influences the level of exploration in the policy.                                      | 0.01. I went with 0.001.                         |
| **Discount Factor (gamma)**     | Trade-off between immediate and future rewards.                 | Higher value favors long-term rewards.                                                  | 0.99                                             |
| **GAE Lambda (gae_lambda)**     | Controls weight of accumulated rewards in advantage estimation. | Adjusts the impact of the GAE calculation on the advantage.                             | 0.95                                             |
| **Number of Parallel Environments** | Number of parallel workers (if used).                           | Impacts training speed and stability.                                                   | Varies. I went with 4 due to the hardware limit. |
| **Network Architecture**        | Design of policy and value function networks.                   | Influences the complexity of the model and its ability to capture patterns in the data. | Varies. I went with MlpPolicy.                   |
| **Optimization Algorithm**      | Choice of optimization algorithm (ex., Adam, RMSprop).          | Affects convergence speed and stability.                                                | Varies. Adam in Stable Baseline3's PPO.          |

PPO has quite a few hyperparameters, and each of them affect the agent's behavior differently. Check [Stable Baseline3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)'s implementation, too. Above table explains some important hyperparameters. I played with hyperparameters and here are some experiment results:

### *Learning Rate*

<img src="readme_image/lr_comparison.png" height="200">

I played around with learning rate to observe its effect on agent's behavior. Green is lr=0.00003 and Red is lr=0.003. As expected, higher learning rate yielded a faster result. If Green was given enough time, it might have yielded a better result than Red. However, since time and performance is a trade-off, Green can be less practical. 

### *Entropy Coefficient*

<img src="readme_image/ent_coeff_comparison.png" height="200">

I played around with entropy coefficient to observe its effect on agent's behavior. Orange is ent_coeff=0.001 and Gray is ent_coeff=0.1. I thought higher entropy coefficient would encourage exploration, but it was quite the opposite for this test. Maybe 0.1 was too much randomness that the Gray agent's decisions were contaminated by noises.

# <ins>Result</ins>

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

I double-checked the math by adding moves manually (step_test.py). I found that this equation a bad estimator for smaller maps (N < 7) where it gives accuracy under 0.90. However, the accuracy is over 0.95 for N > 10 and over 0.99 for N > 23, which is good to use. Since I used 20x20 grid for training, the expected average game length to finish the game is 40000 steps.

## Evaluation






# <ins>Discussion</ins>

* What I achieved
* Limitations 

## Future works

Add more observation for agent:
* I can provide the entire game screen information using a convolutional neural network. It will be a lot more information for the agenet will need to interpret so the learning time can get longer, but it will be a lot less work for me since I won't have to do more work to give the agent quality information. Then, I can compare the learning outcome from manual information and visual information.
* I can try memory system. For instance, a system like LSTM can allow the agent to learn fine techniques from every game and general strategy from multiple runs. 

