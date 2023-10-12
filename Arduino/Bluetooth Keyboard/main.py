import pygame

import bluetooth_stuff
import keyboard
import my_button

# initialize pygame
pygame.init()

pygame.event.set_grab(True)  # Keeps the cursor within the pygame window

# screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 300

# frame rate
FPS = 60

# make the screen and set the title
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Bluetooth Keyboard')

# game clock
timer = pygame.time.Clock()

# font for the program
font = pygame.font.Font('freesansbold.ttf', 18)

# main loop running flag
run = True


# quit button action
def quit_program():
    global run
    run = False


# quit button
quit_btn = my_button.MyButton(screen, 'Quit', 150, 100, True, quit_program)

# the keyboard object
the_keyboard = keyboard.Keyboard()

# the main loop
while run:

    # background white
    screen.fill('white')

    # show the connection text
    _text = font.render('Connected', True, 'black')
    screen.blit(_text, (150, 50))

    # update the keyboard
    the_keyboard.update(pygame.key.get_pressed())

    # update the quit button
    quit_btn.update()

    # also quit the program if the X is clicked
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # update everything on the screen
    # use update() for regional update
    pygame.display.flip()

    # keep the game loop at a certain update rate
    timer.tick(FPS)

# close things before ending the program
pygame.quit()
bluetooth_stuff.close()
