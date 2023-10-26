import cv2
import numpy as np
import constants
import mouse


class EyeDetector:
    """
    Eye detection, gaze detection, and mouse pointer movement.
    """

    def __init__(self):

        # The landmarks from face landmark predictor.
        self.landmarks = None

    ###################################################################
    # Eye image processing section
    ###################################################################

    def __get_region(self, landmark_index):
        """
        Get the landmarks based on the index.
        :param landmark_index: find landmarks from 'constants.' like constants.LEFT_PUPIL_LANDMARK.
        :return: an array of [x, y] points of landmarks
        """

        # initialze with zeros and replace them with xy coords
        region = np.zeros(shape=(len(landmark_index), 2), dtype=np.int32)

        for i, point in enumerate(landmark_index):
            region[i] = self.landmarks[point]

        return region

    ###################################################################
    # Gaze detection section
    ###################################################################

    def __get_horizontal_pupil_direction(self):
        """
        Calculate the horizontal gaze direction based on certain horizontal thresholds.
        :return: constants.LEFT, RIGHT, or CENTER
        """

        # get center x between pupils
        left_pupil = self.__get_region(constants.LEFT_PUPIL_LANDMARK)[0]
        right_pupil = self.__get_region(constants.RIGHT_PUPIL_LANDMARK)[0]
        center_x = int((left_pupil[0] + right_pupil[0]) / 2)

        # get the x direction threshold
        # I needed landmarks that will act as x-thresholds to determine left and right.
        # Nose is the part in the body the moves the least, do it will be a good reference point.
        nose_left_x, nose_left_y = self.__get_region(constants.NOSE_TIP_LANDMARK_LEFT)[0]
        nose_right_x, nose_right_y = self.__get_region(constants.NOSE_TIP_LANDMARK_RIGHT)[0]

        # check direction
        if center_x < nose_left_x:
            return constants.LEFT
        elif center_x > nose_right_x:
            return constants.RIGHT
        else:
            return constants.CENTER

    def __get_vertical_pupil_direction(self):
        """
        Calculate the vertical gaze direction based on certain vertical thresholds.
        :return: constants.UP, DOWN, or CENTER
        """
        # get average y between pupils
        left_pupil = self.__get_region(constants.LEFT_EYELID_TOP_LANDMARK)[0]
        right_pupil = self.__get_region(constants.RIGHT_EYELID_TOP_LANDMARK)[0]
        average_y = int((left_pupil[1] + right_pupil[1]) / 2)

        nose_lower_x, nose_lower_y = self.__get_region(constants.NOSE_TIP_LANDMARK_LOW)[0]
        nose_higher_x, nose_higher_y = self.__get_region(constants.NOSE_TIP_LANDMARK_HIGH)[0]

        # Vertical gaze detection is a bit less intuitive than horizontal gaze detection.
        # I cannot use the pupil's relative position against the eye to determine
        # its vertical position. It's due to the limitation of MediaPipe face mesh:
        # the iris landmarks always stay within the eye fissure landmarks.
        # In other words, if the person looks up and the iris moves up,
        # the entire eye landmarks will follow the iris instead of just the iris moving up.
        # So, instead of using the pupil's relative position against the eye fissure as
        # the measure to determine its vertical motion,
        # I measured its distance to the tip of the nose.
        nose2nose = nose_lower_y - nose_higher_y + 1
        nose2iris = nose_lower_y - average_y
        ratio = nose2iris / nose2nose

        # check direction
        if ratio > constants.NOSE2EYE_RATIO_UP_THRESH:
            return constants.UP
        elif ratio < constants.NOSE2EYE_RATIO_DOWN_THRESH:
            return constants.DOWN
        else:
            return constants.CENTER

    def __move_mouse_with_gaze(self):
        """
        Move the mouse pointer based on the gaze.
        :return:
        """
        hor_dir = self.__get_horizontal_pupil_direction()
        ver_dir = self.__get_vertical_pupil_direction()

        mouse.move_mouse_pointer_4dir(hor_dir, ver_dir)

        return hor_dir, ver_dir

    def apply_face_center_mask(self, image):
        """
        Highlight the human outline on the webcam image to
        help the user center their body for better tracking.
        :param image:
        :return:
        """

        head_center_coordinates = (320, 260)
        head_axesLength = (125, 150)

        body_center_coordinates = (320, 500)
        body_axesLength = (225, 100)

        angle = 0
        startAngle = 0
        endAngle = 360
        color = (255, 255, 255)
        thickness = -1

        mask = np.zeros(shape=(480, 640), dtype=np.uint8)

        mask = cv2.ellipse(mask, head_center_coordinates, head_axesLength,
                           angle, startAngle, endAngle, color, thickness)

        mask = cv2.ellipse(mask, body_center_coordinates, body_axesLength,
                           angle, startAngle, endAngle, color, thickness)

        return cv2.bitwise_and(image, image, mask=mask)

    def __show_info_window(self, image, head_rotation, hor_dir, ver_dir):
        """
        Display the window that describes the status of gaze detection.

        :param image:
        :param head_rotation:
        :param hor_dir:
        :param ver_dir:
        :return:
        """

        # blur the rest of face for privacy
        blurred_image = cv2.blur(image, (21, 21))
        blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)

        # draw iris
        iris_region_l = self.__get_region(constants.LEFT_IRIS_LANDMARKS)
        iris_region_r = self.__get_region(constants.RIGHT_IRIS_LANDMARKS)

        # only reveal the area near eyes
        minx = np.min(iris_region_l[:, 0]) - 15
        maxx = np.max(iris_region_r[:, 0]) + 15
        miny = np.min(iris_region_l[:, 1]) - 5
        maxy = np.max(iris_region_r[:, 1]) + 5

        blur_mask = np.zeros(shape=image.shape, dtype=np.uint8)
        blur_mask = cv2.rectangle(blur_mask, (minx, miny), (maxx, maxy), (255, 255, 255), -1)
        image = np.where(blur_mask == (255, 255, 255), image, blurred_image)

        # apply another mask
        image = self.apply_face_center_mask(image)

        (lx, ly), lr = cv2.minEnclosingCircle(iris_region_l)
        lr = int(lr)
        lx = int(lx)
        ly = int(ly)

        (rx, ry), rr = cv2.minEnclosingCircle(iris_region_r)
        rr = int(rr)
        rx = int(rx)
        ry = int(ry)

        cv2.circle(image, (lx, ly), lr, (0, 0, 255), 2)
        cv2.circle(image, (rx, ry), rr, (0, 0, 255), 2)

        head_rot_color = (255, 255, 255)

        if head_rotation != constants.CENTER:
            head_rot_color = (0, 0, 255)

            cv2.putText(image, 'Align yourself to the outline and', (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_rot_color, 2)
            cv2.putText(image, 'face the camera for better result.', (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_rot_color, 2)

        cv2.putText(image, f'Head rotation: {head_rotation}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, head_rot_color, 2)

        return image

    def update(self, landmarks, bgr_img, eye_closed, head_rotation):
        """
        Update the gaze detection and control the mouse.
        """
        self.landmarks = landmarks

        hor_dir = None
        ver_dir = None

        if not eye_closed and head_rotation == constants.CENTER:
            hor_dir, ver_dir = self.__move_mouse_with_gaze()
            bgr_img = self.__show_info_window(bgr_img, head_rotation, hor_dir, ver_dir)
            bgr_img = draw_arrow(bgr_img, hor_dir, ver_dir)
            return bgr_img

        return self.__show_info_window(bgr_img, head_rotation, hor_dir, ver_dir)


def draw_arrow(image, hor_dir, ver_dir):
    """
    Draw the directional arrow on the image

    :param image:
    :param hor_dir:
    :param ver_dir:
    :return:
    """
    radius = 80

    start_point = (270, radius + 20)

    # End coordinate
    end_x = start_point[0]
    end_y = start_point[1]

    if ver_dir == constants.UP:
        end_y -= radius
    elif ver_dir == constants.DOWN:
        end_y += radius

    if hor_dir == constants.LEFT:
        end_x -= radius
    elif hor_dir == constants.RIGHT:
        end_x += radius

    # Red color in BGR
    color = (0, 255, 0)

    # Line thickness
    thickness = 5

    image = cv2.arrowedLine(image, start_point, (end_x, end_y),
                            color, thickness, tipLength=0.2)
    return image
