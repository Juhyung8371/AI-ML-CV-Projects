
import time
import cv2
import blink_detector as bp
import constants
import eye_detector as ed
import face_detector as fd
import mouse

# determine whether the detection should run
is_enabled = True

# initialize detectors
face_detect = fd.FaceDetector()
blink_detect = bp.BlinkProcessor()
eye_detect = ed.EyeDetector()

# display the gui for testing if wanted
# the_gui = gui.GUI()
# the_gui.show()


def press_key_to_quit():
    """
    If any key is pressed, terminate the program
    and return True, otherwise False.
    :return:
    """
    key = cv2.waitKey(1)
    if key != -1:
        face_detect.turn_off_webcam()
        cv2.destroyAllWindows()
        return True
    return False


def main():
    """
    The main loop.
    :return:
    """

    # main loop
    while is_enabled:

        global eye_detect
        # for FPS checking
        start_time = time.time()

        # update the face detection first
        face_detect.update()

        # do gaze and blink detection if the face is detected
        if face_detect.is_face_detected():

            # get the facial landmarks
            landmarks = face_detect.get_landmarks()

            # update the blink detector
            blink_flag = blink_detect.update(landmarks)

            info_image = eye_detect.update(landmarks,
                                           face_detect.get_bgr_image(),
                                           blink_detect.is_eye_closed(),
                                           face_detect.get_head_rotation())

            if blink_flag == constants.BLINK_FLAG_DOUBLE:
                cv2.putText(info_image, 'Double click', (450, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            elif blink_flag == constants.BLINK_FLAG_SINGLE:
                cv2.putText(info_image, 'Single click', (450, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Info', info_image)

            mouse.update()

        # press any key to exit
        if press_key_to_quit():
            break

        # control FPS via pooling method
        while (time.time() - start_time) < (1 / bp.FPS):
            continue


# run the program
main()

face_detect.turn_off_webcam()

# the_gui.close()
