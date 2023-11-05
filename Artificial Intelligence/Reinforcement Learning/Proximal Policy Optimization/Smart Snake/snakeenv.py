import random
from collections import deque

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from skimage import measure

MAP_SIZE = 20
SNAKE_SPEED = 1
DISPLAY_SCALE = 12
DEATH_PENALTY = 10

SNAKE_MOVE_MEMORY_SIZE = int((MAP_SIZE**2)/2)  # the snake memorizes the last n moves (maybe it adds complexity?)

N_DISCRETE_ACTIONS = 3  # straight left right
OBS_SPACE_SHAPE = SNAKE_MOVE_MEMORY_SIZE + 11 + 3  # prev actions + move check + other observations

MOVE_STRAIGHT = 0
MOVE_LEFT = 1
MOVE_RIGHT = 2

# this clockwise order is important when updating direction
DIR_UP = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_LEFT = 3


class SnekEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ['human', 'none'], "render_fps": 15}

    def __init__(self, render_mode=None):

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-MAP_SIZE, high=MAP_SIZE,
                                            shape=(OBS_SPACE_SHAPE,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window_size = MAP_SIZE * DISPLAY_SCALE
        self.window = None
        self.clock = None

        # these will be updated in reset()
        self.done = None
        self.reward = None
        self.snake_position = None
        self.apple_position = None
        self.score = None
        self.reward = None
        self.snake_head = None
        self.info = None
        self.direction = None

    def step(self, action):

        grew = self._update_snake_pos_n_score()

        self.__update_direction(action)

        # TODO testing still... ------------------------------------------------
        # only keep the relevant moves depending on the length
        # ex: 3 moves if length is 3
        self.prev_actions.popleft()
        insert_index = len(self.snake_position) - 1

        if insert_index > (SNAKE_MOVE_MEMORY_SIZE - 1):
            insert_index = SNAKE_MOVE_MEMORY_SIZE - 1

        self.prev_actions.insert(insert_index, action)

        # game over on collision
        self.done = self.is_game_over()
        self.__update_observation()

        # initialize the reward
        self.reward = 0

        # give higher reward for growing when longer
        if grew:
            self.reward += self.score * 10
            # self.reward += (self.score ** 2) + MAP_SIZE  #+map size to cancel out walking? TODO remove?

        # punish for creating unusable voids
        # TODO change the hard-coding
        if self.observation[2] == 1:
            self.reward -= DEATH_PENALTY

        if self.done:
            self.reward -= DEATH_PENALTY

        # if truncated:
        #     self.reward += GAME_CLEAR_REWARD

        self._render_frame()

        # obs, reward, terminated, truncated, info
        return self.observation, self.reward, self.done, False, {}

    def reset(self, seed=None, options=None):

        center = int(MAP_SIZE / 2)

        self.done = False
        # Initial Snake and Apple position
        self.snake_position = [[center, center], [center - 1, center], [center - 2, center]]
        self.apple_position = [random.randrange(0, MAP_SIZE), random.randrange(0, MAP_SIZE)]
        self.score = 0
        self.reward = 0
        self.snake_head = [center, center]
        self.direction = DIR_RIGHT

        # re-spawn apple if it's spawned on top of the snake
        while self.apple_position in self.snake_position:
            self.apple_position = [random.randrange(0, MAP_SIZE), random.randrange(0, MAP_SIZE)]

        # observation
        # head xy, apple delta_xy, snake length, previous moves
        self.__update_observation(True)

        ## TODO return info if needed #########################
        # self.info = {}

        self._render_frame()

        return self.observation, {}

    def render(self):
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def __update_direction(self, action):

        # no need to update direction for move_straight
        # move and deal with the out-of-direction case

        if action == MOVE_LEFT:

            self.direction -= 1

            if self.direction < 0:
                self.direction = DIR_LEFT

        elif action == MOVE_RIGHT:

            self.direction += 1

            if self.direction > DIR_LEFT:
                self.direction = DIR_UP

    def __update_observation(self, is_reset=False):

        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        apple_x = self.apple_position[0]
        apple_y = self.apple_position[1]

        dir_l = self.direction == DIR_LEFT
        dir_r = self.direction == DIR_RIGHT
        dir_u = self.direction == DIR_UP
        dir_d = self.direction == DIR_DOWN

        # 11 info
        state = [
            # Danger straight
            (dir_r and self.is_game_over(offsetx=1)) or
            (dir_l and self.is_game_over(offsetx=-1)) or
            (dir_u and self.is_game_over(offsety=-1)) or
            (dir_d and self.is_game_over(offsety=1)),

            # Danger right
            (dir_u and self.is_game_over(offsetx=1)) or
            (dir_d and self.is_game_over(offsetx=-1)) or
            (dir_l and self.is_game_over(offsety=-1)) or
            (dir_r and self.is_game_over(offsety=1)),

            # Danger left
            (dir_d and self.is_game_over(offsetx=1)) or
            (dir_u and self.is_game_over(offsetx=-1)) or
            (dir_r and self.is_game_over(offsety=-1)) or
            (dir_l and self.is_game_over(offsety=1)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            apple_x < head_x,  # food left
            apple_x > head_x,  # food right
            apple_y < head_y,  # food up
            apple_y > head_y  # food down
        ]

        state_list = np.array(state, dtype='int16')
        state_list = list(state_list)

        if is_reset:
            self.prev_actions = deque(maxlen=SNAKE_MOVE_MEMORY_SIZE)
            # populate with default values
            for _ in range(SNAKE_MOVE_MEMORY_SIZE):
                self.prev_actions.append(-1)

        # connectedness Map TODO testcode----------------------------------------
        temp_map = np.zeros(shape=(MAP_SIZE, MAP_SIZE), dtype=np.uint8)
        for i, body in enumerate(self.snake_position):
            if i == 0:  # skip the head since it might be out of map
                continue
            temp_map[body[1]][body[0]] = 1

        connectedness = measure.label(temp_map, connectivity=1)

        trapped = 0

        # another penalty for trapping yourself
        if 2 in connectedness:
            trapped = 1

        self.observation = [head_x, head_y, trapped] \
                           + state_list \
                           + list(self.prev_actions)

        self.observation = np.array(self.observation, dtype='int16')  # TODO watch out for overflow

    def _update_snake_pos_n_score(self):
        # Change the head position based on the button direction

        if self.direction == DIR_UP:
            self.snake_head[1] -= SNAKE_SPEED
        elif self.direction == DIR_DOWN:
            self.snake_head[1] += SNAKE_SPEED
        elif self.direction == DIR_LEFT:
            self.snake_head[0] -= SNAKE_SPEED
        elif self.direction == DIR_RIGHT:
            self.snake_head[0] += SNAKE_SPEED

        # add the new snake head position to the position list
        self.snake_position.insert(0, list(self.snake_head))

        # update the score
        grew = self.update_score()

        # respaen apple
        if grew:
            # re-spawn apple if it's spawned on top of the snake
            while self.apple_position in self.snake_position:
                self.apple_position = [random.randrange(0, MAP_SIZE), random.randrange(0, MAP_SIZE)]
        # remove the oldest tail if it didn't grow
        else:
            self.snake_position.pop()



        return grew

    def update_score(self):

        grew = False

        if self.snake_head == self.apple_position:
            grew = True
            self.score += 1

        return grew

    def collision_with_boundaries(self, head):
        return head[0] >= MAP_SIZE or head[0] < 0 or head[1] >= MAP_SIZE or head[1] < 0

    def collision_with_self(self, snake_head):
        return snake_head in self.snake_position[1:]

    def is_game_over(self, offsetx=0, offsety=0):

        temp_head = self.snake_head.copy()

        if offsetx != 0:
            temp_head[0] += offsetx
        if offsety != 0:
            temp_head[1] += offsety

        return self.collision_with_boundaries(temp_head) or self.collision_with_self(temp_head)

    def _render_frame(self):

        if self.render_mode != "human":
            return

        # initialize pygame for human render mode
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Display Apple
        pygame.draw.rect(canvas,
                         (255, 0, 0),
                         pygame.Rect(
                             self.apple_position[0] * DISPLAY_SCALE,
                             self.apple_position[1] * DISPLAY_SCALE,
                             DISPLAY_SCALE,
                             DISPLAY_SCALE
                         ))

        # Display Snake
        for position in self.snake_position:
            pygame.draw.rect(canvas,
                             (0, 255, 0),
                             pygame.Rect(
                                 position[0] * DISPLAY_SCALE,
                                 position[1] * DISPLAY_SCALE,
                                 DISPLAY_SCALE,
                                 DISPLAY_SCALE
                             ))

        if self.done:
            canvas.fill((100, 100, 100))

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
