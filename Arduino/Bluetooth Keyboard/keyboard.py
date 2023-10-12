
from timeit import default_timer
import bluetooth_stuff


class Keyboard:

    def __init__(self):

        # the delay between the first stroke and spam - when the key is pressed for long
        self._FIRST_STROKE_DELAY = 500
        # available keys
        self._NUM_KEYS = 232
        # the time when the key was first stroked before it was released
        self._first_stroke_time = [default_timer() for _ in range(0, self._NUM_KEYS)]
        # to check if you can spam keys now
        self._is_first_stroke = [True for _ in range(0, self._NUM_KEYS)]

    def update(self, key_presses):
        """
        Update the keystroke delays and send the keys via bluetooth.

        :param key_presses: pygame.key.get_pressed() - boolean list of pressed keys.
        :return: None
        """

        # check all the key presses
        for key_code, is_pressed in enumerate(key_presses):

            # pygame has up to 512 keys, but
            # I only care what keyboards can actually press normally.
            # (also want to keep it under uint8)
            if key_code >= self._NUM_KEYS:
                break

            # check if the key is pressed
            if is_pressed:
                # if this is the first stroke, record the time
                if self._is_first_stroke[key_code]:
                    self._is_first_stroke[key_code] = False
                    self._first_stroke_time[key_code] = default_timer()
                    bluetooth_stuff.send_key(key_code)

                # otherwise, check if the first stroke delay is over and can spam
                elif (default_timer() - self._first_stroke_time[key_code]) * 1000 > self._FIRST_STROKE_DELAY:
                    bluetooth_stuff.send_key(key_code)

            # reset the first stroke flag to default
            else:
                self._is_first_stroke[key_code] = True
