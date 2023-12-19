import numpy as np


def calc(N):
    """
    To evaluate the Snake's performance with expected outcome:
    It's a bad estimator for smaller maps (N < 7) where it gives 0.5-0.9 accuracy.
    However, the accuracy is over 0.95 for N > 10 and over 0.99 for N > 23.

    :param N: Grid size (N x N)
    """

    length = 3  # My snake is 3 units long
    unoccupied_cells = N * N - length
    steps = 0

    while True:
        move_per_cycle = np.round(unoccupied_cells / 2)
        steps += move_per_cycle
        unoccupied_cells -= 1
        if unoccupied_cells == 0:
            steps += 1
            break
    print(f'Size: {N}x{N}, Steps: {steps}, Estimation Ratio: {steps / (N ** 4 / 4)}')


# testing
for i in range(3, 23):
    calc(i)
