"""
For mouse features.
"""

import pyautogui

import constants

_velx = 0
_vely = 0
_base_speed = 10
_max_abs_velocity = 30
_decelaration = 5


def update():
    """
    Move the mouse pointer based on the velocity and reduce the velocity value.
    :return:
    """
    global _velx, _vely

    pyautogui.moveRel(_velx, 0, duration=0)
    pyautogui.moveRel(0, _vely, duration=0)

    if _velx > 0:
        _velx -= _decelaration
    elif _velx < 0:
        _velx += _decelaration

    if _vely > 0:
        _vely -= _decelaration
    elif _vely < 0:
        _vely += _decelaration


def _add_velocity(x=0, y=0):
    """
    Add x and y value to the mouse velocity. The velocity has upper and lower limits.
    :param x: Horizontal movement in pixel
    :param y: Vertical movement in pixel
    """
    global _velx, _vely

    _velx += x
    _vely += y

    # limit the max speed
    if _velx < -_max_abs_velocity:
        _velx = -_max_abs_velocity
    elif _velx > _max_abs_velocity:
        _velx = _max_abs_velocity

    if _vely < -_max_abs_velocity:
        _vely = -_max_abs_velocity
    elif _vely > _max_abs_velocity:
        _vely = _max_abs_velocity


def move_mouse_pointer_4dir(hor, ver):
    """
    Add velocity to up, down, left, and right direction.
    :param hor: constants.LEFT or RIGHT
    :param ver: constants.UP or DOWN
    :return:
    """

    if hor == constants.LEFT:
        _add_velocity(x=-_base_speed)
    elif hor == constants.RIGHT:
        _add_velocity(x=_base_speed)

    if ver == constants.UP:
        _add_velocity(y=-_base_speed)
    elif ver == constants.DOWN:
        _add_velocity(y=_base_speed)
