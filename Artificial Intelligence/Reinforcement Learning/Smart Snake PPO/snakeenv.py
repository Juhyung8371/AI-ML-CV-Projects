import random

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

EMPTY_CELL = 0
SNAKE_BODY = 1


class SnekEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ['human', 'none', 'debug'], "render_fps": 40}

    def __init__(self, render_mode='none', map_size=20):

        self.map_size = map_size
        self.map_area = map_size ** 2

        OBS_SPACE_SHAPE = 40

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=-self.map_size, high=self.map_size,
                                            shape=(OBS_SPACE_SHAPE,), dtype=np.float32)
        self.render_mode = render_mode
        self.window_size = self.map_size * DISPLAY_SCALE
        self.window = None
        self.clock = None

        # these will be updated in reset()
        self.done = None
        self.reward_this_step = None
        self.snake_head = None
        self.snake_position = None
        self.apple_position = None
        self.score = 0
        self.direction = None
        self.visited_cells = None
        self.body_map = None
        self.connection_map = None
        self.future_void_moves = None
        self.future_must_fill_moves = None
        self.future_can_reach_moves = None

        self.steps = 0

    def _update_snake(self, action, old_direction, old_head, old_apple, old_positions, old_body_map,
                      force_growth=False):
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

        if force_growth:
            grew = True

        new_position, removed_tail = self._update_snake_positions(old_positions, new_head, grew)
        new_body_map = self._get_next_body_map(old_body_map, new_head, removed_tail)
        new_connection_map = self._to_connection_map(new_body_map)

        return new_direction, new_head, new_position, new_body_map, new_connection_map, grew

    def step(self, action):

        # reset the reward for this step
        self.reward_this_step = 0

        # update the snake properties
        self.direction, self.snake_head, self.snake_position, self.body_map, self.connection_map, grew = self._update_snake(
            action, self.direction, self.snake_head, self.apple_position, self.snake_position, self.body_map)

        length = len(self.snake_position)

        # if apple is eaten, no need to check collision
        if grew:

            self.score += 1  # update the score
            self.reward_this_step += 2 * self.map_size / length  # apply apple reward

            if len(self.snake_position) == self.map_area:
                self.done = True
                self.reward_this_step += self.map_area
                print('beat the game~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!!')
            else:
                # Otherwise, the game must continue. Try summoning the apple.
                self._summon_apple()

            # reset visited cells
            self.visited_cells = [[False for _x in range(self.map_size)] for _y in range(self.map_size)]

        # if the snake didn't grow
        else:

            # TODO the current 'visited' design might not be a good way to encourage exploration
            # check if tail covered the past move
            the_tail = self.snake_position[-1]
            tail_redundant_move_flag = self._check_visited(the_tail)
            # reset redundant move to refresh the board a bit
            if tail_redundant_move_flag:
                self.visited_cells[the_tail[1]][the_tail[0]] = False

            # check for collision and redundant move
            wall_coll, self_coll = self._check_collision(self.snake_head, split_collision_flags=True)
            self.done = wall_coll or self_coll

            redundant_move_flag = self._check_visited(self.snake_head, boundary_collision=wall_coll)

            # mark the visited cell if the snake successfully explored a new place
            if not self.done:

                # survival reward
                self.reward_this_step += length / self.map_area

                if not redundant_move_flag:
                    self.visited_cells[self.snake_head[1]][self.snake_head[0]] = True
                    self.reward_this_step += (length * 0.1) / self.map_area
            else:
                self.reward_this_step -= (length * 10) / self.map_area  # apply death penalty

        # 0 snake_length_normalized,
        # 1 2 3 s_bcoll, r_bcoll, l_bcoll,
        # 4 5 6 s_scoll, r_scoll, l_scoll,
        # 7 8 9 straight_visited, right_visited, left_visited,
        # 10 11 12 13 apple_l, apple_r, apple_u, apple_d,
        # 14 apple_safety,
        # 15 is_stuck_in_void
        # 16 17 18 19 directions
        # 20 21 22 safe_actions_flags
        # 23 24 25 SLR will_create_void_flags
        # 26 27 28 29 positions
        # 30 31 32 SLR fillup_flags
        # 33 34 35 SLR most hug flags
        # 36 will_hug
        # 37 38 39 SLR can reach tail flags

        # finally, update the observation, and then render if wanted
        self._update_observation()

        # # void creation penalty
        if (self.future_void_moves[0] == 1 and action == ACTION_STRAIGHT) or \
                (self.future_void_moves[1] == 1 and action == ACTION_LEFT) or \
                (self.future_void_moves[2] == 1 and action == ACTION_RIGHT):
            self.reward_this_step -= (length * 2.5) / self.map_area

        self.future_void_moves[0] = self.observation[23]
        self.future_void_moves[1] = self.observation[24]
        self.future_void_moves[2] = self.observation[25]

        # # can reach tail bonus
        if (self.future_can_reach_moves[0] == 1 and action == ACTION_STRAIGHT) or \
                (self.future_can_reach_moves[1] == 1 and action == ACTION_LEFT) or \
                (self.future_can_reach_moves[2] == 1 and action == ACTION_RIGHT):
            self.reward_this_step += (length * 1.5) / self.map_area

        self.future_can_reach_moves[0] = self.observation[37]
        self.future_can_reach_moves[1] = self.observation[38]
        self.future_can_reach_moves[2] = self.observation[39]

        # must-fill follow reward
        if (self.future_must_fill_moves[0] == 1 and action == ACTION_STRAIGHT) or \
                (self.future_must_fill_moves[1] == 1 and action == ACTION_LEFT) or \
                (self.future_must_fill_moves[2] == 1 and action == ACTION_RIGHT):
            self.reward_this_step += (length * 0.4) / self.map_area

        self.future_must_fill_moves[0] = self.observation[30]
        self.future_must_fill_moves[1] = self.observation[31]
        self.future_must_fill_moves[2] = self.observation[32]

        # this is the reward for having some huggin wall in the future
        if self.observation[36]:
            self.reward_this_step += (length * 0.3) / self.map_area

        # this is the reward for hugging the body or wall
        # prefer body hug
        if self.observation[4] == 1 or self.observation[5] == 1 or self.observation[6] == 1:
            self.reward_this_step += (length * 0.2) / self.map_area
        # but better if you can also hug the wall
        if self.observation[1] == 1 or self.observation[2] == 1 or self.observation[3] == 1:
            self.reward_this_step += (length * 0.1) / self.map_area

        # Stuck in void penalty
        if self.observation[15] == 1:
            self.reward_this_step -= (length * 5) / self.map_area

        self._render_frame()

        # obs, reward, terminated, truncated, info
        return self.observation, self.reward_this_step, self.done, False, {'score': self.score}

    def reset(self, seed=None, options=None):

        center = int(self.map_size / 2)

        self.done = False
        # Initial Snake and Apple position

        self.snake_head = [center, center]
        self.snake_position = [[center, center], [center - 1, center], [center - 2, center]]

        self.score = 0
        self.reward_this_step = 0
        self.direction = DIR_RIGHT
        self.visited_cells = [[False for _x in range(self.map_size)] for _y in range(self.map_size)]

        self.future_void_moves = [0, 0, 0]
        self.future_must_fill_moves = [0, 0, 0]
        self.future_can_reach_moves = [0, 0, 0]

        # reset and initialize with initial positions
        self.body_map = np.zeros(shape=(self.map_size, self.map_size), dtype=np.uint8)
        for x, y in self.snake_position:
            self.body_map[y][x] = 1

        # summon apple after the body map is established
        self._summon_apple()

        # update connection map body map is established
        self.connection_map = self._to_connection_map(self.body_map)

        # initialize observation
        self._update_observation()

        self._render_frame()

        return self.observation, {'score': self.score}

    def render(self):
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # strategically summon it where it can be summoned
    def _summon_apple(self):

        # no need to continue if the game is over
        if self.done:
            return

        avail_rows = [i for i, row in enumerate(self.body_map) if (EMPTY_CELL in row)]

        # quit if there is no space to put the apple
        if len(avail_rows) == 0:
            return

        the_row = random.choice(avail_rows)
        avail_cols = [i for i, x in enumerate(self.body_map[the_row]) if (x == EMPTY_CELL)]
        the_col = random.choice(avail_cols)

        self.apple_position = [the_col, the_row]

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
        Get the new head positions basedon the new direction.
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

    def _update_observation(self):
        """
        Update the observation. This needs to be called at the end of the step(),
        once everything game-wise if over.
        """

        directions = self._get_one_hot_directions(self.direction)

        s_cell, l_cell, r_cell = self._get_slr_cells(self.snake_head, self.direction)

        s_bcoll, s_scoll = self._check_collision(cell_xy=s_cell, split_collision_flags=True,
                                                 is_checking_future_move=True)
        straight_visited = self._check_visited(cell_xy=s_cell, boundary_collision=s_bcoll)

        l_bcoll, l_scoll = self._check_collision(cell_xy=l_cell, split_collision_flags=True,
                                                 is_checking_future_move=True)
        left_visited = self._check_visited(cell_xy=l_cell, boundary_collision=l_bcoll)

        r_bcoll, r_scoll = self._check_collision(cell_xy=r_cell, split_collision_flags=True,
                                                 is_checking_future_move=True)
        right_visited = self._check_visited(cell_xy=r_cell, boundary_collision=r_bcoll)

        safe_actions_flags = self._get_largest_void_actions_flags(self.snake_head)

        apple_l = self.apple_position[0] < self.snake_head[0]
        apple_r = self.apple_position[0] > self.snake_head[0]
        apple_u = self.apple_position[1] < self.snake_head[1]
        apple_d = self.apple_position[1] > self.snake_head[1]

        apple_safety = self._check_is_apple_safety_flag()
        will_create_void_flags, can_reach_tail_flags = self._get_will_create_void_n_can_reach_tail_flags()

        head_x = self._normalize(self.snake_head[0], 0, self.map_size)
        head_y = self._normalize(self.snake_head[1], 0, self.map_size)
        apple_x = self._normalize(self.apple_position[0], 0, self.map_size)
        apple_y = self._normalize(self.apple_position[1], 0, self.map_size)

        positions_normalized = [head_x, head_y, apple_x, apple_y]

        is_stuck_in_void = self._is_stuck_in_void(self.snake_head, self.body_map)

        snake_length_normalized = self._normalize(self.score + 3, 3, self.map_area)

        must_fill, hugging_flags, will_hug_flag = self._get_must_fill_n_most_hugging_flags(self.snake_head)

        # concat flags
        states = [snake_length_normalized,
                  s_bcoll, r_bcoll, l_bcoll,
                  s_scoll, r_scoll, l_scoll,
                  straight_visited, right_visited, left_visited,
                  apple_l, apple_r, apple_u, apple_d,
                  apple_safety,
                  is_stuck_in_void
                  ] + directions \
                 + safe_actions_flags \
                 + will_create_void_flags \
                 + positions_normalized + must_fill + hugging_flags + [will_hug_flag] + can_reach_tail_flags

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

    def _is_self_collision(self, cell_xy, is_checking_future_move=False, is_for_hugging=False):
        """
        Check if the cell overlaps with the current snake positions.
        :param cell_xy: to check
        :param is_checking_future_move: True if you are checking future moves
                                        (then, tail is excluded from the collision check)
        :return: True if self-collision
        """
        # it's impossible to hit itself if the snake is too short
        # But, if you just want to check the hugging, then i need to check length 4 as well
        if len(self.snake_position) < 5 and not is_for_hugging:
            return False

        starting_index = 3

        # hugging needs to check the collision-ness of 2nd body part
        if is_for_hugging:
            starting_index = 2

        if is_checking_future_move:
            # If we're checking future actions, the tail will move out of the way, always.
            # The only time it won't is when it's 1 move before eating the final apple.
            return cell_xy in self.snake_position[starting_index:-1]
        else:
            return cell_xy in self.snake_position[starting_index:]

    def _check_collision(self, cell_xy, offsetx=0, offsety=0, is_checking_future_move=False,
                         split_collision_flags=False,
                         is_for_hugging=False):
        """

        :param cell_xy:
        :param offsetx: setting this will flag future move to True
        :param offsety: setting this will flag future move to True
        :param is_checking_future_move: self-collision needs to know this because the tail moves
        :param split_collision_flags:
        :return: [boundary_collision, self_collision] if split_collision_flags, else (boundary_collision or self_collision)

        """
        temp_head = cell_xy.copy()

        if offsetx != 0:
            temp_head[0] += offsetx
            is_checking_future_move = True
        if offsety != 0:
            temp_head[1] += offsety
            is_checking_future_move = True

        # check boundary and self collision flags
        boundary_collision = self._is_boundary_collision(temp_head)
        self_collision = self._is_self_collision(temp_head, is_checking_future_move, is_for_hugging)

        return [boundary_collision, self_collision] if split_collision_flags else \
            boundary_collision or self_collision

    def _check_visited(self, cell_xy, offsetx=0, offsety=0, boundary_collision=None):
        """
        Check if this cell can cause collision or visited.
        :param cell_xy: to check
        :param offsetx:
        :param offsety:
        :return: [collision, visited]
        """
        temp_cell = cell_xy.copy()

        if offsetx != 0:
            temp_cell[0] += offsetx
        if offsety != 0:
            temp_cell[1] += offsety

        # check boundary and self collision flags
        if boundary_collision is None:
            boundary_collision = self._is_boundary_collision(temp_cell)

        # consider out-of-bound not visited
        visited = False if boundary_collision else self.visited_cells[temp_cell[1]][temp_cell[0]]

        return visited

    def _check_is_apple_safety_flag(self):
        """
        Check whether the apple is safe to eat by checking if it's surrounded by wall or bodies from 3 sides.
        Snake head and tail end don't count as collidable.
        :return:
        """
        is_up_danger = self._check_collision(offsety=-1, cell_xy=self.apple_position, is_checking_future_move=True)
        is_down_danger = self._check_collision(offsety=1, cell_xy=self.apple_position, is_checking_future_move=True)
        is_left_danger = self._check_collision(offsetx=-1, cell_xy=self.apple_position, is_checking_future_move=True)
        is_right_danger = self._check_collision(offsetx=1, cell_xy=self.apple_position, is_checking_future_move=True)

        num_surrounding = [is_up_danger, is_right_danger, is_left_danger, is_down_danger].count(True)

        # is 3 or more things are surrounding the apple,
        # it might be a trapped one.
        return num_surrounding < 3

    def _is_stuck_in_void(self, cellxy, body_map):

        if self._is_boundary_collision(cellxy):
            return False

        body_map = body_map.copy()

        # I can check whether the head is stuck in a void by checking the connection map after removing the head
        body_map[cellxy[1]][cellxy[0]] = EMPTY_CELL

        connection_map = self._to_connection_map(body_map)

        void_types, void_counts = self._get_void_cells_type_n_count(connection_map)

        void_exists = len(void_counts) > 1

        # there is no void to be stuck
        if not void_exists:
            return False

        max_index = np.argmax(void_counts)

        # check if the current head position is not in the largest void
        return connection_map[cellxy[1]][cellxy[0]] != void_types[max_index]

    def _get_largest_void_actions_flags(self, cellxy):
        """
        Get the actions that leads to the largest available void from cellxy.
        :param cellxy: Anchor
        :return: [straight_safe, left_safe, right_safe]
        """

        # get the direction that leads to the largest available void

        # straight, left, right
        safe_void_flags = [False, False, False]

        # since we're looking at the future move, we can remove the tail from the body map
        temp_body_map = self.body_map.copy()
        tail = self.snake_position[-1]
        temp_body_map[tail[1]][tail[0]] = EMPTY_CELL
        connection_map = self._to_connection_map(temp_body_map)

        # format: [types, ...], [counts, ...]
        void_types, void_counts = self._get_void_cells_type_n_count(connection_map)

        # get the future cells
        s_cell, l_cell, r_cell = self._get_slr_cells(cellxy, self.direction)

        # if it's collidable, it's not a void
        s_coll = self._check_collision(s_cell, is_checking_future_move=True)
        l_coll = self._check_collision(l_cell, is_checking_future_move=True)
        r_coll = self._check_collision(r_cell, is_checking_future_move=True)

        # 0 initially. -1 means it's not worth considering due to collision
        s_cell_count = -1 if s_coll else 0
        l_cell_count = -1 if l_coll else 0
        r_cell_count = -1 if r_coll else 0

        if not s_coll:
            # the void cell's ID
            s_cell_type = connection_map[s_cell[1]][s_cell[0]]
            # -1 because I removed 1 from the list (the body)
            s_cell_count = void_counts[s_cell_type - 1]

        if not l_coll:
            l_cell_type = connection_map[l_cell[1]][l_cell[0]]
            l_cell_count = void_counts[l_cell_type - 1]

        if not r_coll:
            r_cell_type = connection_map[r_cell[1]][r_cell[0]]
            r_cell_count = void_counts[r_cell_type - 1]

        slr_void_cell_counts = [s_cell_count, l_cell_count, r_cell_count]

        # find the largest voids
        max_count = max(slr_void_cell_counts)
        max_indexs = [i for i, x in enumerate(slr_void_cell_counts) if x == max_count]

        # set the action that leads to the largest voids to True
        for i in max_indexs:
            safe_void_flags[i] = True

        return safe_void_flags

    def _get_must_fill_n_most_hugging_flags(self, cellxy):

        # get the future cells
        slr_cells = self._get_slr_cells(cellxy, self.direction)

        must_fill_flags = [False, False, False]

        hug_count = [-1, -1, -1]
        hug_flags = [False, False, False]

        will_hug_flag = False

        for i, cell in enumerate(slr_cells):
            # check the immediate move
            coll = self._check_collision(cell, is_checking_future_move=True)
            # if the immediate move is valid, check the next moves
            if not coll:
                # look for the colidable cells with staircase looking structure
                ucoll = self._check_collision(cell, offsety=-1, is_for_hugging=True, is_checking_future_move=True)
                dcoll = self._check_collision(cell, offsety=1, is_for_hugging=True, is_checking_future_move=True)
                lcoll = self._check_collision(cell, offsetx=-1, is_for_hugging=True, is_checking_future_move=True)
                rcoll = self._check_collision(cell, offsetx=1, is_for_hugging=True, is_checking_future_move=True)

                must_fill_flags[i] = (ucoll and lcoll) or (lcoll and dcoll) or (dcoll and rcoll) or (rcoll and ucoll)

                count = [ucoll, dcoll, lcoll, rcoll].count(True)

                hug_count[i] = count

                # next moves have at least 1 option with hugging
                if not will_hug_flag and count > 0:
                    will_hug_flag = True

        # find the largest voids
        max_count = max(hug_count)
        max_indexs = [i for i, x in enumerate(hug_count) if x == max_count]

        # set the action that leads to the nist hugs to True
        for i in max_indexs:
            hug_flags[i] = True

        return [must_fill_flags, hug_flags, will_hug_flag]

    def _get_void_cells_type_n_count(self, connection_map):
        """
        Get the void cells' type and counts.

        :return: A list of void cells in format [[cell_types], [counts]].
                    'type' is an arbitrary ID for each of them, starting from 1.
                    May return and empty list [[],[]] if there are no voids.
        """

        # find occurrence count to determine which one is a smaller blob (aka void)
        (cell_types, counts) = np.unique(connection_map, return_counts=True)

        # The type 0 is always the snake body.
        # Since the snake body is never a void, remove it
        cell_types = cell_types[1:]
        counts = counts[1:]

        return [cell_types, counts]

    def _get_will_create_void_n_can_reach_tail_flags(self):

        # get the new connection maps based on future actions
        # and force growth to prevent the voids from decreasing in number
        _, s_head, s_pos, s_body_map, s_connection_map, _____ = self._update_snake(
            ACTION_STRAIGHT, self.direction, self.snake_head, self.apple_position, self.snake_position, self.body_map,
            True)
        _, l_head, l_pos, l_body_map, l_connection_map, ____ = self._update_snake(
            ACTION_LEFT, self.direction, self.snake_head, self.apple_position, self.snake_position, self.body_map, True)
        _, r_head, r_pos, r_body_map, r_connection_map, ____ = self._update_snake(
            ACTION_RIGHT, self.direction, self.snake_head, self.apple_position, self.snake_position, self.body_map,
            True)

        ori_void_types, _ = self._get_void_cells_type_n_count(self.connection_map)
        ori_void_count = len(ori_void_types)

        s_void_types, _ = self._get_void_cells_type_n_count(s_connection_map)
        s_void_count = len(s_void_types)

        l_void_types, _ = self._get_void_cells_type_n_count(l_connection_map)
        l_void_count = len(l_void_types)

        r_void_types, _ = self._get_void_cells_type_n_count(r_connection_map)
        r_void_count = len(r_void_types)

        s_void_created = s_void_count > ori_void_count
        l_void_created = l_void_count > ori_void_count
        r_void_created = r_void_count > ori_void_count

        s_can_reach_tail = self._can_reach_tail(s_head, s_pos, s_body_map)
        l_can_reach_tail = self._can_reach_tail(l_head, l_pos, l_body_map)
        r_can_reach_tail = self._can_reach_tail(r_head, r_pos, r_body_map)

        return [[s_void_created, l_void_created, r_void_created],
                [s_can_reach_tail, l_can_reach_tail, r_can_reach_tail]]

    # get head because the first thing in the positions is not always the head
    def _can_reach_tail(self, head, positions, body_map):

        if self._is_boundary_collision(head):
            return False

        tail = positions[-1]

        body_map_copy = body_map.copy()

        # remove the head and tail
        body_map_copy[head[1]][head[0]] = EMPTY_CELL
        body_map_copy[tail[1]][tail[0]] = EMPTY_CELL

        connection_map = self._to_connection_map(body_map_copy)

        # check if the head and tail exists in the same zone
        return connection_map[head[1]][head[0]] == connection_map[tail[1]][tail[0]]

    ########################################################
    # Uility functions
    ########################################################

    def _normalize(self, x, _min, _max):
        return (x - _min) / (_max - _min)

    def _get_next_body_map(self, old_body_map, added_head_xy=None, removed_tail_xy=None):

        head_out_of_map = True if added_head_xy is None else self._is_boundary_collision(added_head_xy)

        tail_removed = removed_tail_xy is not None
        new_body_map = old_body_map.copy()

        # remove the tail first
        if tail_removed:
            new_body_map[removed_tail_xy[1]][removed_tail_xy[0]] = EMPTY_CELL

        if not head_out_of_map:
            new_body_map[added_head_xy[1]][added_head_xy[0]] = SNAKE_BODY

        return new_body_map

    def _to_connection_map(self, body_map):
        return measure.label(body_map, connectivity=1, background=SNAKE_BODY)

    def _get_one_hot_directions(self, direction):

        dir_l = direction == DIR_LEFT
        dir_r = direction == DIR_RIGHT
        dir_u = direction == DIR_UP
        dir_d = direction == DIR_DOWN

        return [dir_l, dir_r, dir_u, dir_d]

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
        color = 255
        for position in self.snake_position:
            pygame.draw.rect(canvas,
                             (0, color, 0),
                             pygame.Rect(
                                 position[0] * DISPLAY_SCALE,
                                 position[1] * DISPLAY_SCALE,
                                 DISPLAY_SCALE,
                                 DISPLAY_SCALE
                             ))
            color -= 2

        # Display head
        pygame.draw.rect(canvas,
                         (100, 20, 200),
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

            snake_length_norm, \
                s_bcoll, r_bcoll, l_bcoll, \
                s_scoll, r_scoll, l_scoll, \
                s_visited, r_visited, l_visited, \
                apple_l, apple_r, apple_u, apple_d, \
                is_apple_safe, \
                is_stuck, \
                dir_l, dir_r, dir_u, dir_d, \
                s_safe, l_safe, r_safe, \
                s_void_created, l_void_created, r_void_created, \
                hx, hy, ax, ay, \
                s_fill, l_fill, r_fill, \
                s_hug, l_hug, r_hug, \
                will_hug, \
                s_can_reach, l_can_reach, r_can_reach = self.observation

            s_xy = [headx, heady]
            r_xy = [headx, heady]
            l_xy = [headx, heady]

            if dir_u:
                s_xy[0] += half_head_size
                r_xy[0] += DISPLAY_SCALE
                r_xy[1] += half_head_size
                l_xy[1] += half_head_size
            elif dir_d:
                s_xy[0] += half_head_size
                s_xy[1] += DISPLAY_SCALE
                r_xy[1] += half_head_size
                l_xy[0] += DISPLAY_SCALE
                l_xy[1] += half_head_size
            elif dir_l:
                s_xy[1] += half_head_size
                r_xy[0] += half_head_size
                l_xy[0] += half_head_size
                l_xy[1] += DISPLAY_SCALE
            elif dir_r:
                s_xy[0] += DISPLAY_SCALE
                s_xy[1] += half_head_size
                r_xy[0] += half_head_size
                r_xy[1] += DISPLAY_SCALE
                l_xy[0] += half_head_size

            red = (255, 0, 0)
            blue = (0, 0, 255)
            purple = (255, 0, 255)
            black = (0, 0, 0)

            red_size = half_head_size + 1  # 11
            blue_size = int(2 * red_size / 3)  # 8
            purple_size = int(red_size / 3)  # 4
            black_size = 2  # 2

            # draw must fill
            if s_fill:
                pygame.draw.circle(canvas, (100, 50, 150), [s_xy[0], s_xy[1]], 17, 0)
            if l_fill:
                pygame.draw.circle(canvas, (100, 50, 150), [l_xy[0], l_xy[1]], 17, 0)
            if r_fill:
                pygame.draw.circle(canvas, (100, 50, 150), [r_xy[0], r_xy[1]], 17, 0)

            # draw most hugged
            if s_hug:
                pygame.draw.circle(canvas, (50, 150, 100), [s_xy[0], s_xy[1]], 14, 0)
            if l_hug:
                pygame.draw.circle(canvas, (50, 150, 100), [l_xy[0], l_xy[1]], 14, 0)
            if r_hug:
                pygame.draw.circle(canvas, (50, 150, 100), [r_xy[0], r_xy[1]], 14, 0)

            # draw danger
            if s_bcoll or s_scoll:
                pygame.draw.circle(canvas, red, [s_xy[0], s_xy[1]], red_size, 0)
            if l_bcoll or l_scoll:
                pygame.draw.circle(canvas, red, [l_xy[0], l_xy[1]], red_size, 0)
            if r_bcoll or r_scoll:
                pygame.draw.circle(canvas, red, [r_xy[0], r_xy[1]], red_size, 0)

            # draw visited
            if s_visited:
                pygame.draw.circle(canvas, blue, [s_xy[0], s_xy[1]], blue_size, 0)
            if l_visited:
                pygame.draw.circle(canvas, blue, [l_xy[0], l_xy[1]], blue_size, 0)
            if r_visited:
                pygame.draw.circle(canvas, blue, [r_xy[0], r_xy[1]], blue_size, 0)

            # draw safe
            if s_safe:
                pygame.draw.circle(canvas, purple, [s_xy[0], s_xy[1]], purple_size, 0)
            if l_safe:
                pygame.draw.circle(canvas, purple, [l_xy[0], l_xy[1]], purple_size, 0)
            if r_safe:
                pygame.draw.circle(canvas, purple, [r_xy[0], r_xy[1]], purple_size, 0)

            # draw void created
            if s_void_created:
                pygame.draw.circle(canvas, black, [s_xy[0], s_xy[1]], black_size, 0)
            if l_void_created:
                pygame.draw.circle(canvas, black, [l_xy[0], l_xy[1]], black_size, 0)
            if r_void_created:
                pygame.draw.circle(canvas, black, [r_xy[0], r_xy[1]], black_size, 0)

            # Display is stuck
            if is_stuck:
                pygame.draw.circle(canvas,
                                   (0, 0, 0),
                                   [self.apple_position[0] * DISPLAY_SCALE,
                                    self.apple_position[1] * DISPLAY_SCALE],
                                   12, 5)

            # Display DANGEROUS Apple
            if not is_apple_safe:
                pygame.draw.rect(canvas,
                                 (255, 255, 0),
                                 pygame.Rect(
                                     self.apple_position[0] * DISPLAY_SCALE,
                                     self.apple_position[1] * DISPLAY_SCALE,
                                     DISPLAY_SCALE,
                                     DISPLAY_SCALE
                                 ))

            # draw visited path
            for y, row in enumerate(self.visited_cells):
                for x, visited in enumerate(row):
                    if visited:
                        pygame.draw.circle(canvas,
                                           (0, 0, 255),
                                           [x * DISPLAY_SCALE + half_head_size,
                                            y * DISPLAY_SCALE + half_head_size],
                                           4, 0)

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

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def get_screen_img(self):
        if self.window is None:
            return None

        canvas = pygame.display.get_surface()
        img = pygame.image.tobytes(canvas, 'RGB')

        return img
