# Pathfinding using Q-learning method

In this scenario, I will use the q-learning method to teach warehouse robots to find the best path between a location in the warehouse and the drop-off location. I will also discuss congestion problems that may arise from every robot taking the same optimal path. 
This project is inspired by [this tutorial](https://youtu.be/iKdlKYG78j4). 

## What is Q-Learning?
- Q-learning is a type of reinforcement learning method. It involves states(map), actions (input), and rewards (output).
- Q-learning does not involve probabilistic models. Instead, it creates the optimal policy via trial and error. 

## Q-value
- The Q-value indicates the quality of the action in a given state.
- A higher Q-value means more reward.
- Every combination of state and action has a Q-value associated with it. This information is stored in a Q-table (policy).
- The states and the actions must be finite.

## Temporal Difference
<img src='https://github.com/Juhyung8371/AI-ML-CV-Projects/blob/main/Artificial%20Intelligence/Reinforcement%20Learning/Q-Learning/Warehouse%20Navigation/Temporal%20Difference.png?raw=true' width=600> 

[image source](https://youtu.be/__t2XRxXGxI)

- The temporal difference is a method of considering current rewards in evaluating past actions.

<img src='https://github.com/Juhyung8371/AI-ML-CV-Projects/blob/main/Artificial%20Intelligence/Reinforcement%20Learning/Q-Learning/Warehouse%20Navigation/Bellman%20Equation.png?raw=true' width=600> 

[image source](https://youtu.be/__t2XRxXGxI)

- Then, we can use the Bellman Equation to update the Q-value

## Define the environment 

States: 11 x 11 grid

Actions: up, right, down, and left

Each cell in the grid is either a wall, a road, or the goal. The environmental feedback is the following:

1. a penalty for traveling through the wall
2. a penalty for not taking the best path
3. a reward for reaching the goal

## Training

As described in the first section, training is done using a Q-table, temporal difference, and the Bellman Equation. Check the code for more details.

## Result

The pathfinding algorithm works as intended.

<img src='https://github.com/Juhyung8371/AI-ML-CV-Projects/blob/main/Artificial%20Intelligence/Reinforcement%20Learning/Q-Learning/Warehouse%20Navigation/before_path.png?raw=true'>

## Congestion Problem

There is more than one robot working in a warehouse. If every robot takes the same optimal path, then that will cause a congestion problem. I will alleviate that issue by analyzing the road usage rate and adjusting the reward of each road based on its usage rate. For example, robots will be rewarded for using longer but less used roads.

After adjusting the rewards, the right path's usage decreases up to 8, and that is transferred to the left path. This can alleviate some congestion problems in the right path.

Usage rate before and after:

<img src='https://github.com/Juhyung8371/AI-ML-CV-Projects/blob/main/Artificial%20Intelligence/Reinforcement%20Learning/Q-Learning/Warehouse%20Navigation/before_usage.png?raw=true'> <img src='https://github.com/Juhyung8371/AI-ML-CV-Projects/blob/main/Artificial%20Intelligence/Reinforcement%20Learning/Q-Learning/Warehouse%20Navigation/after_usage.png?raw=true'>

Path change before and after:

<img src='https://github.com/Juhyung8371/AI-ML-CV-Projects/blob/main/Artificial%20Intelligence/Reinforcement%20Learning/Q-Learning/Warehouse%20Navigation/before_path.png?raw=true'> <img src='https://github.com/Juhyung8371/AI-ML-CV-Projects/blob/main/Artificial%20Intelligence/Reinforcement%20Learning/Q-Learning/Warehouse%20Navigation/after_path.png?raw=true'>
