import cv2
import mediapipe as mp
import numpy as np

import constants


def get_region(landmarks, landmark_index):
    """
    Get the landmarks based on the index.
    :param landmark_index: find landmarks from 'constants.' like constants.LEFT_PUPIL_LANDMARK.
    :return: an array of [x, y] points of landmarks
    """

    # initialze with zeros and replace them with xy coords
    region = np.zeros(shape=(len(landmark_index), 2), dtype=np.int32)

    for i, point in enumerate(landmark_index):
        region[i] = landmarks[point]

    return region


class FaceDetector:
    """
    To get a grayscale webcam image and a face landmarks from it.
    """

    def __init__(self):

        # webcam
        self.__cap = cv2.VideoCapture(0)

        # load the face detection tool
        self.__mp_face_mesh = mp.solutions.face_mesh
        self.__face_mesh = self.__mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # add more landmarks like iris from the newer model
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6)

        self.__bgr_image = None
        self.__face_detected = False

        # an array of x,y pixel points
        self.__landmarks = None

        # an array of x,y,z landmarks (normalized)
        self.__landmarks_raw = None

    def update(self):
        """
        Update the face detection based on the new camera feed.
        """

        is_webcam_on, frame = self.__cap.read()

        # pass if no image is provided
        if is_webcam_on is not True:
            return

        # flip the webcam image horizontally so it's like looking in a mirror
        frame = cv2.flip(frame, 1)

        # update the image field
        self.__bgr_image = frame

        # mediapipe takes RGB
        rgb_image = cv2.cvtColor(self.__bgr_image, cv2.COLOR_BGR2RGB)

        # This will pass image by reference, improving performance for the landmark detection
        rgb_image.flags.writeable = False

        # process the face mesh
        result = self.__face_mesh.process(rgb_image)

        # set is back to mutable image
        rgb_image.flags.writeable = True

        # Facial landmarks or None
        self.__landmarks_raw = result.multi_face_landmarks

        # update face detected flag
        self.__face_detected = self.__landmarks_raw is not None

        height, width, _ = rgb_image.shape

        # convert the raw normalized landmark into x,y pixel format
        # x: 0.25341567397117615
        # y: 0.71121746301651
        # z: -0.03244325891137123 (but z is not needed in my application)
        if self.is_face_detected():
            self.__landmarks = np.array(
                [np.multiply([land.x, land.y], [width, height]).astype(int) for land in
                 self.__landmarks_raw[0].landmark])

    def is_face_detected(self):
        return self.__face_detected

    def get_landmarks(self):
        """
        :return: An array of face landmark points [x,y] in pixel.
        """
        return self.__landmarks if self.is_face_detected() else None

    def get_bgr_image(self):
        return self.__bgr_image if self.is_face_detected() else None

    def turn_off_webcam(self):
        self.__cap.release()

    def get_head_rotation(self):
        """
        Get the 3D face rotation based on the landmarks.
        Source: https://www.youtube.com/watch?v=-toNMaS4SeQ
        :return: constant.UP, DOWN, LEFT, or RIGHT
        """

        img_h, img_w, img_c = self.__bgr_image.shape
        face_3d = []
        face_2d = []

        if self.is_face_detected():
            for idx, lm in enumerate(self.__landmarks_raw[0].landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            # It seems like there is a slight bias towards right
            # (camera calibration problem, maybe)
            if y < 3:
                text = constants.LEFT
            elif y > 8:
                text = constants.RIGHT
            elif x < 2:
                text = constants.DOWN
            elif x > 6.5:
                text = constants.UP
            else:
                text = constants.CENTER

            return text
