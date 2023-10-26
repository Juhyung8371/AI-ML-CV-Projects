"""
To keep track of constants in the project.

Keep in mind certain constants are subjective, so calibration is recommended.
"""

###########################################
# Direction
###########################################

LEFT = 'left'
RIGHT = 'right'
UP = 'up'
DOWN = 'down'
CENTER = 'center'

###########################################
# Landmarks
###########################################

# See Tensorflow page for more info:
# https://github.com/tensorflow/tfjs-models/blob/ad17ade67add3e84fee0895c938ea4e1cd4d50e4/face-landmarks-detection/src/constants.ts

# The webcam image is flipped horizontally, so it's like looking in a mirror.
# Left and right is based on the viewer's point of view

# it starts from the right corner and goes clockwise
LEFT_EYE_LANDMARKS = [133, 155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE_LANDMARKS = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

# it starts from the right corner and goes clockwise
LEFT_IRIS_LANDMARKS = [469, 472, 471, 470]
RIGHT_IRIS_LANDMARKS = [474, 477, 476, 475]

LEFT_PUPIL_LANDMARK = [468]
RIGHT_PUPIL_LANDMARK = [473]

# For vertical gaze detection
LEFT_EYELID_TOP_LANDMARK = [158]
RIGHT_EYELID_TOP_LANDMARK = [385]

# For vertical gaze detection
NOSE_TIP_LANDMARK_LOW = [1]
NOSE_TIP_LANDMARK_HIGH = [5]

# for horizontal gaze detection
NOSE_TIP_LANDMARK_LEFT = [44]
NOSE_TIP_LANDMARK_RIGHT = [274]

###########################################
# Thresholds
###########################################

# the nose-to-iris ratio threshold to be considered gaze up or down
NOSE2EYE_RATIO_UP_THRESH = 2.9
NOSE2EYE_RATIO_DOWN_THRESH = 2.6

# the gaze ratio threshold to be considered a blink
BLINK_THRESHOLD = 6
# the gaze ratio threshold to be considered a closed eye
EYE_CLOSED_THRESH = 10

###########################################
# Flags
###########################################

BLINK_FLAG_NONE = 0
BLINK_FLAG_SINGLE = 1
BLINK_FLAG_DOUBLE = 2
