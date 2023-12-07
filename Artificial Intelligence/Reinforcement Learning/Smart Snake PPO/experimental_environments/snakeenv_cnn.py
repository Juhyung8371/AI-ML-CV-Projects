import random
from timeit import default_timer as timer

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from skimage import measure

DISPLAY_SCALE = 20

N_DISCRETE_ACTIONS = 3  # straight left right

# action space
ACTION_STRAIGHT = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2

# this clockwise order is important when updating direction
DIR_UP = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_LEFT = 3

# these values are used for view normalization,
# so let out_of_grid and apple be min max
EMPTY_CELL = 0
SNAKE_BODY = 1
SNAKE_HEAD = 2
SNAKE_TAIL = 3
APPLE = 4


class SnekEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ['human', 'none', 'debug'], "render_fps": 40}

    def __init__(self, render_mode='none', map_size=20):

        self.map_size = map_size
        self.map_area = map_size ** 2

        # CnnPolicy's minimum resolution
        min_size = 36
        self.obs_scale = int(min_size / self.map_size) + 1  # will be 1 or higher

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.map_size * self.obs_scale, self.map_size * self.obs_scale, 1),
                                            dtype=np.uint8)
        self.observation = None

        # rendering
        self.render_mode = render_mode
        self.prev_render_time = None

        # following will be updated in reset()
        self.done = None
        self.collided = None
        self.early_stop = None
        self.score = 0
        self.reward_this_step = None

        self.snake_head = None
        self.snake_position = None
        self.apple_position = None
        self.direction = None
        self.game_map = None
        self.connection_map = None

        self.current_can_reach_tail = None
        self.past_can_reach_tail = None
        self.is_tail_lost_flag = None

        self.past_apple_distance = None
        self.current_apple_distance = None
        self.is_apple_closer = None

        self.steps_this_apple = None

        self.MAX_STEPS_PER_APPLE = self.map_area  # if you can't get an apple while running the entire map, stop

    def step(self, action):

        # reset the reward for this step
        self.reward_this_step = 0

        # update the snake properties. The order matters.
        self.direction = self._update_direction(action, self.direction)
        self.snake_head, grew = self._update_snake_head_position(self.snake_head, self.direction, self.apple_position)
        self.snake_position, removed_tail = self._update_snake_positions(self.snake_position, self.snake_head, grew)
        self.game_map = self._update_game_map(self.game_map, self.snake_position, removed_tail)

        if grew:
            self._update_apple_position(self.game_map)

        self.connection_map = self._update_connection_map(self.snake_head, self.snake_position[-1], self.game_map)

        self._update_game_over(grew)

        self._update_can_reach_tail(self.collided)
        self._update_apple_distance()

        self._update_observation()

        #################################################
        # calculate rewards
        #################################################

        length = len(self.snake_position)

        # if the snake is still alive
        if not self.done:
            # The snake ate the apple!
            if grew:
                self.score += 1  # update the score
                self.reward_this_step += length

            if self.is_apple_closer:
                self.reward_this_step += length / self.map_area  # small reward

            # penalty for not having an escape route
            if self.is_tail_lost_flag:
                self.reward_this_step -= length / 2

        # either a game over or game completed
        else:
            if self.collided or self.early_stop:
                self.reward_this_step -= length  # apply death penalty
            else:
                self.reward_this_step += length * 10  # game completion

        self._render_frame()

        # print(f'{self.steps_this_apple}, {self.reward_this_step}')

        # obs, reward, terminated, truncated, info
        return self.observation, self.reward_this_step, self.done, False, {'score': self.score}

    def reset(self, seed=None, options=None):
        """
        Reset/initialize all the game states to the default.
        """
        self.done = False
        self.collided = False
        self.early_stop = False
        self.score = 0
        self.reward_this_step = 0
        self.steps_this_apple = 0

        # Initial Snake position and direction (length 3 in the center, looking right)
        self.direction = DIR_RIGHT
        center = int(self.map_size / 2)
        self.snake_head = [center, center]
        self.snake_position = [[center, center], [center - 1, center], [center - 2, center]]

        # reset and initialize the body map
        self.game_map = np.zeros(shape=(self.map_size, self.map_size), dtype=np.uint8)
        snake_len = len(self.snake_position)
        for i, pos in enumerate(self.snake_position):
            x, y = pos
            if i == 0:
                self.game_map[y][x] = SNAKE_HEAD
            elif i == snake_len - 1:
                self.game_map[y][x] = SNAKE_TAIL
            else:
                self.game_map[y][x] = SNAKE_BODY

        # summon apple after the body map is established
        self._update_apple_position(self.game_map)

        # update connection map after the body map is established
        self.connection_map = self._update_connection_map(self.snake_head, self.snake_position[-1], self.game_map)

        # you can reach the tail at the start due to the starting position
        self.current_can_reach_tail = True
        self.past_can_reach_tail = True
        self.is_tail_lost_flag = False

        self.past_apple_distance = self._get_apple_distance()
        self.current_apple_distance = self.past_apple_distance
        self.is_apple_closer = 0

        # initialize observation
        self._update_observation()

        self._render_frame()

        return self.observation, {'score': self.score}

    def render(self):  # reset() renders instead
        pass

    def close(self):
        if self.prev_render_time is not None:
            cv2.destroyAllWindows()

    ########################################################
    # Game loop update utility functions
    ########################################################

    def _update_direction(self, action, old_direction):
        """
        Update the direction based on the action.
        :param action:
        :param old_direction:
        :return: new direction
        """
        if action == ACTION_LEFT:
            return DIR_LEFT if old_direction == DIR_UP else old_direction - 1
        elif action == ACTION_RIGHT:
            return DIR_UP if old_direction == DIR_LEFT else old_direction + 1
        else:
            return old_direction

    def _update_snake_head_position(self, old_head, new_direction, apple_position):
        """
        Get the new head positions based on the new direction.
        :param old_head:
        :param new_direction:
        :param apple_position:
        :return: [new head position [x, y], and whether the snake should grow]
        """
        new_head = old_head.copy()

        if new_direction == DIR_UP:
            new_head[1] -= 1
        elif new_direction == DIR_DOWN:
            new_head[1] += 1
        elif new_direction == DIR_LEFT:
            new_head[0] -= 1
        elif new_direction == DIR_RIGHT:
            new_head[0] += 1

        return [new_head, new_head == apple_position]

    def _update_snake_positions(self, old_positions, new_head, grew):
        """
        Update the snake positions and return the new positions and the removed tail's [x,y] if the snake didn't grow.
        :param grew:
        :return: [new snake positions, removed tail's [x,y] or None]
        """
        new_positions = old_positions.copy()

        new_positions.insert(0, new_head)
        removed_old_tail = new_positions.pop() if not grew else None

        return [new_positions, removed_old_tail]

    def _update_game_map(self, old_game_map, positions, removed_tail_xy=None):

        new_game_map = old_game_map.copy()
        head = positions[0]
        grew = removed_tail_xy is None

        # update tail and apple based on growth
        if not grew:
            new_game_map[removed_tail_xy[1]][removed_tail_xy[0]] = EMPTY_CELL
            new_tail = self.snake_position[-1]
            new_game_map[new_tail[1]][new_tail[0]] = SNAKE_TAIL

        # update the new head
        if not self._check_collision(head, positions):
            new_game_map[head[1]][head[0]] = SNAKE_HEAD

        # replace the past head with body
        past_head = positions[1]
        new_game_map[past_head[1]][past_head[0]] = SNAKE_BODY

        # update apple once the snake body is all updated
        if grew:
            new_game_map[self.apple_position[1]][self.apple_position[0]] = APPLE

        return new_game_map

    def _update_connection_map(self, headxy, tailxy, game_map):
        """
        Get the connection map without the head and the tail.
        It's easier to track the 'void created' and 'can reach tail' this way.
        :param headxy:
        :param tailxy:
        :param game_map:
        :return: connection map
        """
        game_map_copy = game_map.copy()
        head_out_of_map = self._is_boundary_collision(headxy)

        # Remove the tail and replace head
        # because they are constantly moving (and they use different code from SNAKE_BODY)
        if not head_out_of_map:
            game_map_copy[headxy[1]][headxy[0]] = SNAKE_BODY

        game_map_copy[tailxy[1]][tailxy[0]] = EMPTY_CELL

        # remove apple too
        game_map_copy[self.apple_position[1]][self.apple_position[0]] = EMPTY_CELL

        return measure.label(game_map_copy, connectivity=1, background=SNAKE_BODY)

    def _update_can_reach_tail(self, collided):
        """
        Update whether the head can reach the tail.
        """
        # can't reach tail when you are dead
        if collided:
            self.is_tail_lost_flag = True
            return

        self.past_can_reach_tail = self.current_can_reach_tail

        tailxy = self.snake_position[-1]

        # since head is SNAKE_BODY and tail is EMPTY_CELL in connection_map,
        # check the slr cells to see of they are connected to the tail
        s, l, r = self._get_slr_cells(self.snake_head, self.direction)

        # reset
        self.current_can_reach_tail = False

        # update can reach tail
        if not self._check_collision(s, self.snake_position):
            self.current_can_reach_tail = self.connection_map[s[1]][s[0]] == self.connection_map[tailxy[1]][tailxy[0]]
        if not self._check_collision(l, self.snake_position) and not self.current_can_reach_tail:
            self.current_can_reach_tail = self.connection_map[l[1]][l[0]] == self.connection_map[tailxy[1]][tailxy[0]]
        if not self._check_collision(r, self.snake_position) and not self.current_can_reach_tail:
            self.current_can_reach_tail = self.connection_map[r[1]][r[0]] == self.connection_map[tailxy[1]][tailxy[0]]

        # only true when the snake just lost its tail
        self.is_tail_lost_flag = self.past_can_reach_tail and not self.current_can_reach_tail

    def _update_apple_position(self, game_map):
        """
        Update the position variable and the game map.
        """
        # no need to continue if the game is over
        if self.done:
            return

        avail_rows = [i for i, row in enumerate(game_map) if (EMPTY_CELL in row)]

        # quit if there is no space to put the apple
        if len(avail_rows) == 0:
            self.apple_position = None
            return

        the_row = random.choice(avail_rows)
        avail_cols = [i for i, x in enumerate(game_map[the_row]) if (x == EMPTY_CELL)]
        the_col = random.choice(avail_cols)

        self.apple_position = [the_col, the_row]

        self.game_map[the_row][the_col] = APPLE

    def _update_apple_distance(self):

        self.past_apple_distance = self.current_apple_distance
        self.current_apple_distance = self._get_apple_distance()

        self.is_apple_closer = self.current_apple_distance < self.past_apple_distance

    def _update_game_over(self, grew):
        """
        Update collided, early_stop, and done.
        :param grew:
        :return:
        """
        # increase the step (for early stopping)
        self.steps_this_apple = self.steps_this_apple + 1

        if grew:
            self.steps_this_apple = 0

        # update some flags after the snake properties are updated
        if self.steps_this_apple < self.MAX_STEPS_PER_APPLE:
            self.collided = self._check_collision(cell_xy=self.snake_head, snake_position=self.snake_position)
        else:  # early-stop if the snake is too bad
            self.early_stop = True

        # game over or game completed
        self.done = self.collided or self.early_stop or len(self.snake_position) == self.map_area

    def _update_observation(self):
        """
        Update the observation.
        """
        # convert booleans to number
        # also, observations needs to be a np array
        self.observation = self._game_map_to_obs()

    ########################################################
    # Flag functions
    ########################################################

    def _is_boundary_collision(self, cell_xy):
        """
        Check if the cell is outside the game grid.
        :param cell_xy: to check
        :return: True if outside
        """
        return cell_xy[0] >= self.map_size or cell_xy[0] < 0 or cell_xy[1] >= self.map_size or cell_xy[1] < 0

    def _is_self_collision(self, cell_xy, snake_position, exclude_tail=False):
        """
        Check if the cell overlaps with the current snake positions.
        :param cell_xy: to check
        :param exclude_tail: For future collision check. Tail will move out of the way, so it can't be collided.
        :return: True if self-collision
        """
        # it's impossible to hit itself if the snake is too short
        if len(snake_position) < 5:
            return False

        # you can't collide with the first 4 body parts
        return cell_xy in snake_position[3:-1] if exclude_tail else cell_xy in snake_position[3:]

    def _check_collision(self, cell_xy, snake_position, offsetx=0, offsety=0, split_collision_flags=False):
        """
        :param cell_xy: to check
        :param split_collision_flags:
        :return: [boundary_collision, self_collision] if split_collision_flags,
                else (boundary_collision or self_collision)
        """

        cell_xy = cell_xy.copy()
        exclude_tail = False

        if offsetx != 0 or offsety != 0:
            cell_xy[0] += offsetx
            cell_xy[1] += offsety
            exclude_tail = True

        # check boundary and self collision flags
        boundary_collision = self._is_boundary_collision(cell_xy)
        self_collision = self._is_self_collision(cell_xy, snake_position, exclude_tail)

        return [boundary_collision, self_collision] if split_collision_flags else \
            boundary_collision or self_collision

    ########################################################
    # Utility functions
    ########################################################

    def _normalize(self, x, _min, _max):
        return (x - _min) / (_max - _min)

    def _get_one_hot_directions(self, direction):

        dir_l = direction == DIR_LEFT
        dir_r = direction == DIR_RIGHT
        dir_u = direction == DIR_UP
        dir_d = direction == DIR_DOWN

        return [dir_l, dir_r, dir_u, dir_d]

    def _get_apple_distance(self):

        x = self.snake_head[0] - self.apple_position[0] if self.snake_head[0] > self.apple_position[0] else \
        self.apple_position[0] - self.snake_head[0]
        y = self.snake_head[1] - self.apple_position[1] if self.snake_head[1] > self.apple_position[1] else \
        self.apple_position[1] - self.snake_head[1]

        return self._normalize(x + y, 0, (self.map_size - 1) * 2)

    def _get_slr_cells(self, cellxy, direction):
        """
        Get the coordinates of the straight, left, and right cells based on the
        given cell and direction.
        CAUTION: it may return out of bound cells
        :param cellxy: Anchor
        :param direction: The direction to reference straight, left, and right actions
        :return: [straight_xy, left_xy, right_xy]
        """

        s_cell = cellxy.copy()
        r_cell = cellxy.copy()
        l_cell = cellxy.copy()

        if direction == DIR_UP:
            s_cell[1] -= 1
            l_cell[0] -= 1
            r_cell[0] += 1
        elif direction == DIR_DOWN:
            s_cell[1] += 1
            l_cell[0] += 1
            r_cell[0] -= 1
        elif direction == DIR_LEFT:
            s_cell[0] -= 1
            l_cell[1] += 1
            r_cell[1] -= 1
        elif direction == DIR_RIGHT:
            s_cell[0] += 1
            l_cell[1] -= 1
            r_cell[1] += 1

        return [s_cell, l_cell, r_cell]

    def _get_void_cells_type_n_count(self, connection_map):
        """
        Get the void cells' type and counts.

        :return: A list of void cells in format [[cell_types], [counts]].
                    'type' is an arbitrary ID for each of them, starting from 1.
                    May return and empty list [[],[]] if there are no voids.
        """

        # find occurrence count to determine which one is a smaller blob (aka void)
        cell_types, counts = np.unique(connection_map, return_counts=True)

        # The type 0 is always the snake body.
        # Since the snake body is never a void, remove it
        cell_types = cell_types[1:]
        counts = counts[1:]

        return [cell_types, counts]

    ########################################################
    # Game render
    ########################################################

    def _game_map_to_obs(self):

        # white image background
        img = np.full(shape=(self.map_size, self.map_size, 3), fill_value=255, dtype=np.uint8)

        # draw body map
        for y, row in enumerate(self.game_map):
            for x, cell in enumerate(row):
                if cell == SNAKE_BODY:
                    img[y][x] = [0, 255, 0]
                elif cell == SNAKE_HEAD:
                    img[y][x] = [0, 0, 255]
                elif cell == SNAKE_TAIL:
                    img[y][x] = [100, 150, 100]
                elif cell == APPLE:
                    img[y][x] = [255, 0, 0]

        img = cv2.resize(img, (0, 0), fx=self.obs_scale, fy=self.obs_scale, interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = np.reshape(img, (img.shape[0], img.shape[1], 1))

        return img

    def _obs_to_img(self, scale=10):

        img = self.observation.copy()
        img = np.reshape(img, (img.shape[0], img.shape[1]))

        return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def _render_frame(self):

        if self.render_mode == "none":
            return

        if self.prev_render_time is None:
            self.prev_render_time = timer()

        cv2.imshow('game', self._obs_to_img(scale=10))

        if cv2.waitKey(1) & 0xff == ord('q'):
            self.done = True

        # control FPS via pooling method
        while (timer() - self.prev_render_time) < (1 / self.metadata["render_fps"]):
            continue

    def get_screen_img(self):
        """
        Export the screen image to bytes.
        :return:
        """
        if self.prev_render_time is None:
            return None

        return None  # TODO -----------------------------------------------------------
