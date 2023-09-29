# Flappy Bird with a genetic algorithm

## Intoduction

Flappy Bird is a game where the player navigates the bird through pairs of pipes with equally sized gaps placed at random heights. The bird automatically descends and only ascends when the player taps the touchscreen. Each successful pass through a pair of pipes awards the player one point. Colliding with a pipe, ground, or the ceiling ends the gameplay.

This project aims to make an AI to learn how to play Flappy Bird using a genetic algorithm.

## Genetic Algorithm

The genetic algorithm is a method for solving optimization problems based on natural selection. 

<img src='https://www.mathworks.com/help/gads/gaflowchart.png'>

This flowchart from [MathWorks](https://www.mathworks.com/help/gads/what-is-the-genetic-algorithm.html) summarizes the idea of a genetic algorithm.

1. The initial population is created. Their DNAs are all random.
2. Score population.
3. Choose which entities will pass down their DNAs. Usually, the entities that performed well will be chosen.
4. Produce new batches of DNAs based on the ones from Step 3.
5. Introduce some mutations in DNA for variance. This helps the algorithm to escape the local maxima.
6. Combine the populations from Step 4 and 5, and go back to Step 2.

## Model

The model consists of the following:

1. 3 inputs
   - Horizontal distance traveled, vertical distance to the pipe gap, and vertical distance to the center of the map
   - Inputs are normalized within the (-1, 1) range.
2. 1 hidden layer with 6 fully-connected perceptron
   - Each perceptron takes inputs from the previous layer, multiplies them with its weights, sums up the result, and adds its bias.
3. 1 output
   - The result > 0.5 means perform 'jump'.

## Result



## Discussion
  
