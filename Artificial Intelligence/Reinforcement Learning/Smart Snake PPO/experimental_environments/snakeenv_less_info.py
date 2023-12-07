import random
from collections import deque

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from skimage import measure

SNAKE_SPEED = 1
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
OUT_OF_GRID = -1
EMPTY_CELL = 0
SNAKE_BODY = 1
SNAKE_HEAD = 2

MEMORY_LENGTH = 100
MEMORY_DIFFERENCE_THRESHOLD = 1  # inclusive
VISION_RANGE = 2


class SnekEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ['human', 'none', 'debug'], "render_fps": 40}

    def __init__(self, render_mode='none', map_size=20):

        self.map_size = map_size
        self.map_area = map_size ** 2

        # OBS_SPACE_SHAPE = 21 + 8*4 #(VISION_RANGE * 2 + 1) ** 2
        OBS_SPACE_SHAPE = 3 + 11 + 2

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(OBS_SPACE_SHAPE,), dtype=np.float32)
        # rendering
        self.render_mode = render_mode
        self.window_size = self.map_size * DISPLAY_SCALE
        self.window = None
        self.clock = None

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
        self.body_map = None
        self.connection_map = None

        self.current_void_num = None
        self.past_void_num = None
        self.void_created_flag = None

        self.current_can_reach_tail = None
        self.past_can_reach_tail = None
        self.is_tail_lost_flag = None

        self.past_is_hugging = None
        self.current_is_hugging = None
        self.is_not_hugging_twice_in_a_row_flag = None

        self.past_apple_distance = None
        self.current_apple_distance = None
        self.is_apple_closer = None

        self.memory = None
        self.snake_view = None
        self.is_view_novel_flag = None

        self.steps_this_apple = None

        self.MAX_STEPS_PER_APPLE = self.map_area  # if you can't get an apple while running the entire map, stop

    def _update_snake(self, action, old_direction, old_head, old_apple, old_positions, old_body_map):
        """
        Update the old snake properties based on the action.
        :param action:
        :param old_direction:
        :param old_head:
        :param old_apple:
        :param old_positions:
        :param old_body_map:
        :return: [new_direction, new_head, new_position, new_body_map, new_connection_map, grew]
        """

        new_direction = self._update_direction(action, old_direction)
        new_head, grew = self._update_snake_head_position(old_head, new_direction, old_apple)
        new_position, removed_tail = self._update_snake_positions(old_positions, new_head, grew)
        new_body_map = self._update_body_map(old_body_map, new_position, removed_tail)
        new_connection_map = self._update_connection_map(new_head, new_position[-1], new_body_map)

        return [new_direction, new_head, new_position, new_body_map, new_connection_map, grew]

    def step(self, action):

        # reset the reward for this step
        self.reward_this_step = 0

        # update the snake properties
        self.direction, self.snake_head, self.snake_position, self.body_map, self.connection_map, grew = self._update_snake(
            action, self.direction, self.snake_head, self.apple_position, self.snake_position, self.body_map)

        # increase the step (for early stopping)
        self.steps_this_apple = self.steps_this_apple + 1

        if grew:
            self.steps_this_apple = 0

        # update some flags after the snake properties are updated
        if self.steps_this_apple < self.MAX_STEPS_PER_APPLE:
            self.collided = self._check_collision(cell_xy=self.snake_head, snake_position=self.snake_position)
        else:  # early-stop if the snake is too bad
            self.early_stop = True

        self._update_void_created(self.collided)
        self._update_can_reach_tail(self.collided)
        self._update_hugging(self.snake_position, self.collided)
        self._update_apple_distance()

        # update the apple position before observation update,
        # so that the snake knows where the apple is as soon as it eats one
        if grew:
            self._update_apple_position(self.body_map)

        # anything new in terms of game state should be all updated by now, so update the observation
        self._update_observation()

        #################################################
        # calculate rewards
        #################################################

        # Feedback size:
        # Collision > Tail access lost > Void creation > Eating apples > Hugging >= Exploration
        # 1. Collision needs to be avoided at all cost. This penalty must be bigger than any rewards combined.
        # 2. Tail access lost means no escape route, which can be an immediate danger.
        # 3. Void creation means less spaces are available for traversal, which can be a future danger.
        # 4. Apple is important to win the game, but not losing the game comes first.
        # 5. Hugging and Exploration are to encourage safe but new actions. They enourage more efficient apple-eating behavior.

        # length = len(self.snake_position)

        # if the snake is still alive
        if not self.done:
            # The snake ate the apple!
            if grew:
                self.score += 1  # update the score
                self.reward_this_step += self.map_size / 2

            # exploration reward
            if self.is_view_novel_flag:
                self.reward_this_step += 0.5

            if self.is_apple_closer:
                self.reward_this_step += 0.5

            # reward for hugging a collide-able cell to encourage not to move too wildly and make a void
            if self.score > 1:
                if not self.is_not_hugging_twice_in_a_row_flag:
                    self.reward_this_step += 1  # encourage hugging a lot
                else:
                    self.reward_this_step -= 2

            # penalty for not having an escape route
            if self.is_tail_lost_flag:
                self.reward_this_step -= self.map_size / 2 - 1

            # void creation penalty
            if self.void_created_flag:
                self.reward_this_step -= self.map_size / 2 - 1

        # either a game over or game completed
        else:
            if self.collided or self.early_stop:
                self.reward_this_step -= self.map_size / 2 + 2  # apply death penalty
            else:
                self.reward_this_step += self.map_size * 2  # game completion

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
        self.body_map = np.zeros(shape=(self.map_size, self.map_size), dtype=np.uint8)
        for x, y in self.snake_position:
            self.body_map[y][x] = 1

        # summon apple after the body map is established
        self._update_apple_position(self.body_map)

        # update connection map after the body map is established
        self.connection_map = self._update_connection_map(self.snake_head, self.snake_position[-1], self.body_map)

        # there is only 1 'void' at the start
        self.current_void_num = 1
        self.past_void_num = 1
        self.void_created_flag = False

        # you can't hug anything at the start due to the starting position
        self.current_is_hugging = False
        self.past_is_hugging = False
        self.is_not_hugging_twice_in_a_row_flag = False

        # you can reach the tail at the start due to the starting position
        self.current_can_reach_tail = True
        self.past_can_reach_tail = True
        self.is_tail_lost_flag = False

        self.past_apple_distance = self._get_apple_distance()
        self.current_apple_distance = self.past_apple_distance
        self.is_apple_closer = 0

        # update memory after the body map is established
        self.memory = deque([], maxlen=MEMORY_LENGTH)
        self.snake_view = self._get_snake_view(self.snake_head,
                                               self.direction)  # self._get_snake_view_info_from_lines(self.snake_head, self.body_map, collided=False)
        self.is_view_novel_flag = self._update_memory(self.snake_view)  # Populate the memory. This will be True

        # initialize observation
        self._update_observation()

        self._render_frame()

        return self.observation, {'score': self.score}

    def render(self):  # reset() renders instead
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

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
            new_head[1] -= SNAKE_SPEED
        elif new_direction == DIR_DOWN:
            new_head[1] += SNAKE_SPEED
        elif new_direction == DIR_LEFT:
            new_head[0] -= SNAKE_SPEED
        elif new_direction == DIR_RIGHT:
            new_head[0] += SNAKE_SPEED

        return [new_head, new_head == apple_position]

    def _update_snake_positions(self, old_positions, new_head, grew):
        """
        Update the snake positions and return the new positions and the removed tail's [x,y] if the snake didn't grow.
        :param grew:
        :return: [new snake positions, removed tail's [x,y] or None]
        """
        new_positions = old_positions.copy()

        # add the new snake head position to the positions list (the tail end is still not updated yet)
        new_positions.insert(0, new_head)

        removed_old_tail = new_positions.pop() if not grew else None

        # remove the oldest tail if it didn't grow
        return [new_positions, removed_old_tail]

    def _update_body_map(self, old_body_map, positions, removed_tail_xy=None):

        new_body_map = old_body_map.copy()
        head = positions[0]

        if removed_tail_xy is not None:
            new_body_map[removed_tail_xy[1]][removed_tail_xy[0]] = EMPTY_CELL

        # update the new head if it's not collided
        if not self._check_collision(head, positions):
            new_body_map[head[1]][head[0]] = SNAKE_HEAD

        # replace the past head with body
        past_head = positions[1]
        new_body_map[past_head[1]][past_head[0]] = SNAKE_BODY

        return new_body_map

    def _update_connection_map(self, headxy, tailxy, body_map):
        """
        Get the connection map without the head and the tail.
        It's easier to track the 'void created' and 'can reach tail' this way.
        :param headxy:
        :param tailxy:
        :param body_map:
        :return: connection map
        """
        body_map_copy = body_map.copy()
        head_out_of_map = self._is_boundary_collision(headxy)

        # Remove the tail and replace head
        # because they are constantly moving (and they use different code from SNAKE_BODY)
        if not head_out_of_map:
            body_map_copy[headxy[1]][headxy[0]] = SNAKE_BODY

        body_map_copy[tailxy[1]][tailxy[0]] = EMPTY_CELL

        return measure.label(body_map_copy, connectivity=1, background=SNAKE_BODY)

    def _update_void_created(self, collided):
        """
        Update the number of voids in the grid - past and current - and check if voids increased.
        """
        # can't create void when you are dead
        if collided:
            self.void_created_flag = False
            return

        void_types, void_counts = self._get_void_cells_type_n_count(self.connection_map)
        self.past_void_num = self.current_void_num
        self.current_void_num = len(void_types)

        # void is created when the current void number is more than the past's
        self.void_created_flag = self.current_void_num > self.past_void_num

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

    def _update_hugging(self, positions, collided):
        """
        Update the hugging status (adjacent collide-able cells).
        :return:
        """
        # can't hug anything if you are dead
        if collided:
            self.current_is_hugging = False
            self.is_not_hugging_twice_in_a_row_flag = True
            return

        s, l, r = self._get_slr_cells(self.snake_head, self.direction)

        self.past_is_hugging = self.current_is_hugging
        self.current_is_hugging = self._check_collision(s, positions) or \
                                  self._check_collision(l, positions) or \
                                  self._check_collision(r, positions)

        self.is_not_hugging_twice_in_a_row_flag = not self.current_is_hugging and not self.past_is_hugging

    def _update_memory(self, view):
        """
        Update the memory queue. The queue populates from the left.
        It saves the view only when the difference >= MEMORY_DIFFERENCE_THRESHOLD
        It throws away old memory if queue length > MEMORY_LENGTH.

        :param view:
        :return: is_novel - True if this view is not in the memory.
        """

        is_novel = True

        # check if the view exists in the memory
        for piece in self.memory:

            difference_score = np.count_nonzero(view != piece)

            # at least N tiles must be different
            if difference_score < MEMORY_DIFFERENCE_THRESHOLD:
                is_novel = False
                break

        if is_novel:
            # append left, so we can check the recent ones first in the memory later
            self.memory.appendleft(view)

        return is_novel

    def _update_apple_position(self, body_map):
        """
        Summon it where it can be summoned.
        """
        # no need to continue if the game is over
        if self.done:
            return

        avail_rows = [i for i, row in enumerate(body_map) if (EMPTY_CELL in row)]

        # quit if there is no space to put the apple
        if len(avail_rows) == 0:
            return

        the_row = random.choice(avail_rows)
        avail_cols = [i for i, x in enumerate(body_map[the_row]) if (x == EMPTY_CELL)]
        the_col = random.choice(avail_cols)

        self.apple_position = [the_col, the_row]

    def _update_apple_distance(self):

        self.past_apple_distance = self.current_apple_distance
        self.current_apple_distance = self._get_apple_distance()

        self.is_apple_closer = self.current_apple_distance < self.past_apple_distance

    def _update_observation(self):
        """
        Update the observation. This needs to be called at the end of the step(),
        once everything game-wise if over.
        """

        # update the view
        self.snake_view = self._get_snake_view(self.snake_head,
                                               self.direction)  # _get_snake_view_info_from_lines(self.snake_head, self.body_map, collided=self.collided)

        # check whether the current view is new
        self.is_view_novel_flag = self._update_memory(self.snake_view)

        # game over or game completed
        self.done = self.collided or self.early_stop or len(self.snake_position) == self.map_area

        # directions = self._get_one_hot_directions(self.direction)

        apple_l = self.apple_position[0] < self.snake_head[0]
        apple_r = self.apple_position[0] > self.snake_head[0]
        apple_u = self.apple_position[1] < self.snake_head[1]
        apple_d = self.apple_position[1] > self.snake_head[1]

        # which actions lead to the apple
        apple_slr = [0, 0, 0]

        if self.direction == DIR_UP:
            if apple_u:
                apple_slr[0] = 1
            if apple_l:
                apple_slr[1] = 1
            elif apple_r:
                apple_slr[2] = 1
        elif self.direction == DIR_RIGHT:
            if apple_r:
                apple_slr[0] = 1
            if apple_u:
                apple_slr[1] = 1
            elif apple_d:
                apple_slr[2] = 1
        elif self.direction == DIR_DOWN:
            if apple_d:
                apple_slr[0] = 1
            if apple_r:
                apple_slr[1] = 1
            elif apple_l:
                apple_slr[2] = 1
        elif self.direction == DIR_LEFT:
            if apple_l:
                apple_slr[0] = 1
            if apple_d:
                apple_slr[1] = 1
            elif apple_u:
                apple_slr[2] = 1

        apple_distance_diff = self.current_apple_distance - self.past_apple_distance

        view = self.snake_view.tolist()

        steps = self._normalize(self.steps_this_apple, 0, self.map_area)

        # concat flags
        states = apple_slr + view + [steps, apple_distance_diff]

        # convert booleans to number
        # also, observations needs to be a np array
        self.observation = np.array(states, dtype=np.float32)

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

    def _check_apple_edible_flag(self, snake_position):
        """
        Check whether the apple is safe to eat by checking if it's surrounded by wall or bodies from 3 sides.
        Snake tail end don't count as collide-able.
        :return:
        """
        is_up_danger = self._check_collision(offsety=-1, cell_xy=self.apple_position, snake_position=snake_position)
        is_down_danger = self._check_collision(offsety=1, cell_xy=self.apple_position, snake_position=snake_position)
        is_left_danger = self._check_collision(offsetx=-1, cell_xy=self.apple_position, snake_position=snake_position)
        is_right_danger = self._check_collision(offsetx=1, cell_xy=self.apple_position, snake_position=snake_position)

        num_surrounding = [is_up_danger, is_right_danger, is_left_danger, is_down_danger].count(True)

        # is 3 or more things are surrounding the apple,
        # it might be a trapped one.
        return num_surrounding < 3

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

    def _get_snake_view(self, headxy, direction, is_reduced=True):
        """
        Get the NxN square view around the head, where N is vision_range*2+1
        and where the head is in center of that square.
        The view will be in the snake's perspective, where it's aligned in the way that the head is pointing up.
        :param headxy:
        :param vision_range: # of block to look
        :return: A flattened, 1D numpy array of the view for the agent to learn
        """
        hx = headxy[0]
        hy = headxy[1]

        vision_range = VISION_RANGE

        vision_size = vision_range * 2 + 1

        # the vision range is a square around the head
        row_start = hy - vision_range
        row_end = row_start + vision_size
        col_start = hx - vision_range
        col_end = col_start + vision_size

        # Crop out the visible map part
        # out-of-bound cases when the head is collided to the boundary of the grid
        if col_start < 0:
            col_start = 0
        elif col_end >= self.map_size:
            col_end = self.map_size  # the end index is non-inclusive so added 1
        if row_start < 0:
            row_start = 0
        elif row_end >= self.map_size:
            row_end = self.map_size

        # have a blank canvas and then put the cropped image on it
        bg = np.full(shape=(vision_size, vision_size), fill_value=OUT_OF_GRID, dtype=np.int8)

        body_map = self.body_map.copy()
        body_map[self.apple_position[1]][self.apple_position[0]] = EMPTY_CELL
        vision_cropped = body_map[row_start:row_end, col_start:col_end]

        c_rows, c_cols = vision_cropped.shape

        # To shift the cropped map on the background
        cropped_row_shift = 0
        cropped_col_shift = 0
        if hx < vision_range:
            cropped_col_shift = vision_range - hx
        if hy < vision_range:
            cropped_row_shift = vision_range - hy

        # apply the image on the background
        bg[cropped_row_shift:c_rows + cropped_row_shift, cropped_col_shift:c_cols + cropped_col_shift] = vision_cropped

        # now, rotate the view based on the snake's perspective (direction)
        # The view will be re-aligned so that the head is always pointed up.
        # Hopefully this will help the snake get more consistent views.
        if direction == DIR_LEFT:
            bg = np.rot90(bg, k=-1)  # clockwise 90
        elif direction == DIR_RIGHT:
            bg = np.rot90(bg, k=1)  # counter-clockwise 90
        elif direction == DIR_DOWN:
            bg = np.rot90(bg, k=2)  # counter-clockwise 180

        # normalize the view
        bg = self._normalize(bg, OUT_OF_GRID, SNAKE_HEAD)

        # flattened image
        bg = np.reshape(bg, (vision_size ** 2,))

        # reduce the view to 11 instead of 25
        if is_reduced:
            mask = np.array([False, False, True, False, False,
                             False, True, True, True, False,
                             True, True, True, True, True,
                             False, True, False, True, False,
                             False, False, False, False, False])
            return bg[mask]
        else:
            return bg

    def _get_snake_view_info_from_lines(self, headxy, body_map, collided=None, positions=None):
        """
        :return: A flattened, 1D numpy array of the view for the agent to learn
        """

        # Total 8 lines in radially outward direction from the head (head excluded).
        # Each line deliver 4 following info:
        # is head - 0 or 1  True or False, basically
        # distance to wall - 0 if collided to wall ~ 1
        # distance to self - 0 if not there ~ 1
        # distance to apple - 0 if not there ~ 1
        num_flags = 3
        flags = np.zeros(shape=(8 * num_flags,), dtype=np.float32)  # default case

        # deal with the collision case so there are no array-out-of-bound cases later.
        # View from the collision is not too useful either, since the snake can't see its head position.
        if collided is None:
            collided = self._check_collision(headxy, positions)

        # return default if collided to wall
        if collided:
            return flags

        hx = headxy[0]
        hy = headxy[1]

        # Get the whole line, and split if from where the head is
        # each split line starts from the head and outward (all include head)

        # vertical
        ver_view = body_map[:, hx:hx + 1]
        ver_view = ver_view.reshape((self.map_size,))
        top = np.flip(ver_view[:hy + 1])
        bot = ver_view[hy:]

        # horizontal
        hor_view = body_map[hy]
        hor_view = np.reshape(hor_view, (self.map_size,))
        left = np.flip(hor_view[:hx + 1])
        right = hor_view[hx:]

        # diagonal TL to BR
        m = min(hx, hy)
        TLBR_view = np.diagonal(a=body_map, offset=hx - hy)
        tl = np.flip(TLBR_view[:m + 1])
        br = TLBR_view[m:]

        # diagonal TR to BL
        fx = self.map_size - 1 - hx
        n = min(fx, hy)
        TRBL_view = np.diagonal(a=np.fliplr(body_map), offset=fx - hy)
        tr = np.flip(TRBL_view[:n + 1])
        bl = TRBL_view[n:]

        every_line = None

        # clockwise
        # rotate the view so that teh snake faces up
        if self.direction == DIR_UP:
            every_line = [top, tr, right, br, bot, bl, left, tl]
        elif self.direction == DIR_LEFT:
            every_line = [left, tl, top, tr, right, br, bot, bl]
        elif self.direction == DIR_DOWN:
            every_line = [bot, bl, left, tl, top, tr, right, br]
        elif self.direction == DIR_RIGHT:
            every_line = [right, br, bot, bl, left, tl, top, tr]

        # Now, check each line for the distance to wall, self, tail, and apple.
        # 1 tile away from the head is distance 1
        for line_index, line in enumerate(every_line):

            body_found = False

            for cell_index, cell_type in enumerate(line):
                # Determine which cell type it is and record the distance
                # Find the body only once (closest)
                if cell_type == SNAKE_HEAD:
                    flags[line_index * num_flags] = 1
                elif cell_type == OUT_OF_GRID:
                    flags[line_index * num_flags + 1] = cell_index
                elif cell_type == SNAKE_BODY and not body_found:
                    flags[line_index * num_flags + 2] = cell_index
                    body_found = True

        # return a 1D normalized distance ndarray
        return self._normalize(flags, 0, self.map_size)

    ########################################################
    # Game render
    ########################################################

    def _render_frame(self):

        if self.render_mode == "none":
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
        pygame.draw.rect(canvas,
                         (100, 200, 100),
                         pygame.Rect(
                             self.snake_head[0] * DISPLAY_SCALE,
                             self.snake_head[1] * DISPLAY_SCALE,
                             DISPLAY_SCALE,
                             DISPLAY_SCALE
                         ))
        # DEBUG ########################

        if self.render_mode == 'debug':

            headx = self.snake_head[0] * DISPLAY_SCALE
            heady = self.snake_head[1] * DISPLAY_SCALE
            half_head_size = int(DISPLAY_SCALE / 2)

            apple_s, apple_l, apple_r, *_, = self.observation

            s_xy = [headx, heady]
            r_xy = [headx, heady]
            l_xy = [headx, heady]

            if self.direction == DIR_UP:
                s_xy[0] += half_head_size
                r_xy[0] += DISPLAY_SCALE
                r_xy[1] += half_head_size
                l_xy[1] += half_head_size
            elif self.direction == DIR_DOWN:
                s_xy[0] += half_head_size
                s_xy[1] += DISPLAY_SCALE
                r_xy[1] += half_head_size
                l_xy[0] += DISPLAY_SCALE
                l_xy[1] += half_head_size
            elif self.direction == DIR_LEFT:
                s_xy[1] += half_head_size
                r_xy[0] += half_head_size
                l_xy[0] += half_head_size
                l_xy[1] += DISPLAY_SCALE
            elif self.direction == DIR_RIGHT:
                s_xy[0] += DISPLAY_SCALE
                s_xy[1] += half_head_size
                r_xy[0] += half_head_size
                r_xy[1] += DISPLAY_SCALE
                l_xy[0] += half_head_size

            red = (255, 0, 0)
            blue = (0, 0, 255)

            red_size = half_head_size + 1  # 11

            if self.void_created_flag:
                pygame.draw.circle(canvas, blue, [l_xy[0], l_xy[1]], 4, 0)

            if self.is_tail_lost_flag:
                pygame.draw.circle(canvas, red, [r_xy[0], r_xy[1]], 4, 0)

            # draw body map
            for y, row in enumerate(self.body_map):
                for x, cell in enumerate(row):
                    if cell == SNAKE_BODY:
                        pygame.draw.circle(canvas,
                                           (0, 0, 0),
                                           [x * DISPLAY_SCALE + half_head_size,
                                            y * DISPLAY_SCALE + half_head_size],
                                           2, 0)

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined frame rate.
        # The following line will automatically add a delay to keep the frame rate stable.
        self.clock.tick(self.metadata["render_fps"])

    def get_screen_img(self):
        """
        Export the screen image to bytes.
        :return:
        """
        if self.window is None:
            return None

        canvas = pygame.display.get_surface()
        img = pygame.image.tobytes(canvas, 'RGB')

        return img
