"""
A module for the blink detection and click.
"""

from math import hypot  # for distance between points in eye region
from timeit import default_timer as timer  # for click timing

import pyautogui

import constants

# The ideal frames per seconds
FPS = 15

# 4 frames for 1 blink
# see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4043155/
# see https://www.researchgate.net/publication/262280711_Evaluating_Perceived_Trust_From_Procedurally_Animated_Gaze
# see https://www.pnas.org/doi/10.1073/pnas.1714220115
MAX_QUEUE_LENGTH = 4

# the time limit in seconds between each blink to be considered a double click
DOUBLE_CLICK_TIME_LIMIT = 1


class BlinkProcessor:

    def __init__(self):

        # keeps track of time between each click
        self.double_click_timing_prev = 0

        # where the recent blink ratios get stored for blink detection
        self.blink_ratio_queue = []

        self.landmarks = None

        self.double_click_ready_flag = False

    def update(self, landmarks_):
        """
        Update the blink processing.

        :param landmarks_: The landmarks from face predictor.
        """

        self.landmarks = landmarks_

        # the order is important
        self.__update_blink_ratio_queue()
        blink_flag = self.__detect_blink()
        self.__update_double_click_timer()

        return blink_flag

    def is_eye_closed(self):
        return self.__get_average_blink_ratio() > constants.EYE_CLOSED_THRESH

    def __update_blink_ratio_queue(self):
        """Update the frames in the ratio queue with FIFO policy."""

        # add the newest frame at the back
        self.blink_ratio_queue.append(self.__get_average_blink_ratio())

        # if the queue is full, remove the first one
        if len(self.blink_ratio_queue) > MAX_QUEUE_LENGTH:
            self.blink_ratio_queue.pop(0)

    def __detect_blink(self):
        """
        Detect blink and execute a click or double click.
        """
        length = len(self.blink_ratio_queue)

        # the queue only has 4 frames, which is just enough for a blink
        # so wait for it to fully fill up
        if length == MAX_QUEUE_LENGTH:

            # 4 phases of blink
            closing = self.blink_ratio_queue[0]
            closed = self.blink_ratio_queue[1]
            early_opening = self.blink_ratio_queue[2]
            late_opening = self.blink_ratio_queue[3]

            # check the peak
            # the closed or early_opening phases should have high blink ratio
            if closed >= constants.BLINK_THRESHOLD or early_opening >= constants.BLINK_THRESHOLD:

                # the closing and late_opening phases should have low blink ratio
                if closing < constants.BLINK_THRESHOLD and late_opening < constants.BLINK_THRESHOLD:

                    # since a blink is found, the stored ratios are not needed anymore
                    self.blink_ratio_queue.clear()

                    # a consecutive blink is detected
                    if self.double_click_ready_flag:

                        self.double_click_ready_flag = False

                        pyautogui.click(pyautogui.position().x, pyautogui.position().y)
                        pyautogui.click(pyautogui.position().x, pyautogui.position().y)

                        return constants.BLINK_FLAG_DOUBLE

                    # a single blink is detected otherwise
                    else:
                        self.__raise_double_click_flag()

                        pyautogui.click(pyautogui.position().x, pyautogui.position().y)

                        return constants.BLINK_FLAG_SINGLE

        return constants.BLINK_FLAG_NONE

    def get_eye_dimensions(self, left_or_right):
        """
        :param left_or_right:
        :return: (horizontal_eye_length, vertical_eye_length)
        """

        # The eyes landmark points from the face detector
        if left_or_right == constants.LEFT:
            eye_points = constants.LEFT_EYE_LANDMARKS
        else:
            eye_points = constants.RIGHT_EYE_LANDMARKS

        right_point = self.landmarks[eye_points[0]]
        bottom_point = self.landmarks[eye_points[4]]
        left_point = self.landmarks[eye_points[8]]
        top_point = self.landmarks[eye_points[12]]

        # get the length
        # hypot is based on origin so we subtract two for the difference
        ver_line_length = hypot((top_point[0] - bottom_point[0]),
                                (top_point[1] - bottom_point[1]))

        # get the length
        hor_line_length = hypot((left_point[0] - right_point[0]),
                                (left_point[1] - right_point[1]))

        return hor_line_length, ver_line_length

    def __get_blink_ratio(self, left_or_right):
        """
        Get the blinking ratio of one eye.

        :param left_or_right: "left" or "right" to choose the eye.
        :type left_or_right: string
        :return: The ratio of the width and height of the eye.
        """

        # get the length
        # hypot is based on origin so we subtract two for the difference
        hor_line_length, ver_line_length = self.get_eye_dimensions(left_or_right)

        # since the vertical is smaller, hor/ver will give a bigger ratio than ver/hor
        # and eye height (ver Length) will never be 0
        ratio = (hor_line_length / (ver_line_length + 1))

        return ratio

    def __get_average_blink_ratio(self):
        """Get the average blinking ratio between both eyes."""
        return (self.__get_blink_ratio("left") + self.__get_blink_ratio("right")) / 2

    def __raise_double_click_flag(self):
        """Raise the double_click_ready_flag and start the double click timer."""
        self.double_click_ready_flag = True
        self.double_click_timing_prev = timer()

    def __update_double_click_timer(self):
        """If there is no blink detected within DOUBLE_CLICK_TIME_LIMIT,
        lower the double click flag."""
        # if it's ready to take in a double click
        if self.double_click_ready_flag:
            # and this blink is done consecutively within the timeframe
            if (timer() - self.double_click_timing_prev) >= DOUBLE_CLICK_TIME_LIMIT:
                self.double_click_ready_flag = False
