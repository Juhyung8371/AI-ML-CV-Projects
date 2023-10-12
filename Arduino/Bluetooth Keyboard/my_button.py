import pygame

# initialize pygame
pygame.init()

# font for the program
_font = pygame.font.Font('freesansbold.ttf', 18)


# a button class for a clickable UI
class MyButton:
    __width = 100
    __height = 50
    __screen = None

    def __init__(self, _screen, _text, _x, _y, _enabled, _action):
        self.text = _text
        self.x = _x
        self.y = _y
        self.enabled = _enabled
        self.action = _action
        self.button_rect = pygame.rect.Rect((self.x, self.y),
                                            (MyButton.__width, MyButton.__height))

        self.clicked = False

        MyButton.__screen = _screen
        # draw on creation
        self.__draw()

    def __draw(self):
        """
        Draw and update the button
        :return:
        """
        button_text = _font.render(self.text, True, 'black')

        if self.enabled:
            # if clicked
            if self.__check_clicked():
                # 0 for solid (other number for edge width), 5 for rounded-ness
                pygame.draw.rect(MyButton.__screen, 'dark gray', self.button_rect, 0, 5)

                if self.clicked is False:
                    self.clicked = True
                    self.action()

            else:
                pygame.draw.rect(MyButton.__screen, 'light gray', self.button_rect, 0, 5)
                self.clicked = False
        else:
            pygame.draw.rect(MyButton.__screen, 'red', self.button_rect, 0, 5)

        pygame.draw.rect(MyButton.__screen, 'black', self.button_rect, 2, 5)
        MyButton.__screen.blit(button_text, (self.x + 3, self.y + 3))

    def __check_clicked(self):
        """
        Check if the cursor is clicked on the button rect.
        :return:
        """
        mouse_pos = pygame.mouse.get_pos()
        left_click = pygame.mouse.get_pressed()[0]

        return left_click and self.button_rect.collidepoint(mouse_pos) and self.enabled

    def set_text(self, text):
        self.text = str(text)

    def update(self):
        self.__draw()
