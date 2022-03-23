"""Script for interactively contolling the 3D Double Ball Balancer in the pybullet simulation
"""
import copy
import pygame
import time

import numpy as np

from pybullet_simulation import PyBulletSim, Params, VELOCITY_MODE

# https://stackoverflow.com/questions/42014195/rendering-text-with-multiple-lines-in-pygame


def blit_text(surface, text, pos, font, color=pygame.Color('white')):
    words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
    space = font.size(' ')[0]  # The width of a space.
    max_width, max_height = surface.get_size()
    x, y = pos
    for line in words:
        for word in line:
            word_surface = font.render(word, 0, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = pos[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = pos[0]  # Reset the x.
        y += word_height  # Start on new row.


class KeyboardCommander:
    def __init__(self):
        self.cmd = np.zeros(2)
        self.cmd_limits = np.array([1.2, 0.3])
        self.increment_time = 0.05

        self.cmd_increment_scale = np.array([0.1, 0.02])
        buttons = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]

        self.pressed = {key: None for key in buttons}

        self.increment = {pygame.K_UP: np.array([1, 0]),
                          pygame.K_DOWN: np.array([-1, 0]),
                          pygame.K_LEFT: np.array([0, 1]),
                          pygame.K_RIGHT: np.array([0, -1])}

        pygame.init()
        self.display_surface = pygame.display.set_mode((700, 200))
        pygame.display.set_caption('Keyboard Control Panel')

        self.font = pygame.font.Font('freesansbold.ttf', 20)

        self.text = "After clicking on this window:\n- press q / Esc to quit\n- press the arrow keys to change the speed commands \n- press s to stop the robot"

        self.updateDisplay()

    def updateDisplay(self):
        self.display_surface.fill(color=pygame.Color('black'))
        text = self.text + "\n\nspeed commands: forward: {:.2f}  turn: {:.2f}".format(self.cmd[0], self.cmd[1])
        blit_text(self.display_surface, text, (20, 20), self.font)
        pygame.display.update()

    def processKeyEvents(self):
        prev_cmd = copy.copy(self.cmd)

        time_now = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in [
                    pygame.K_ESCAPE, pygame.K_q]):
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.cmd = np.zeros(2)
            elif event.type in [pygame.KEYDOWN, pygame.KEYUP] and event.key in self.pressed.keys():
                self.pressed[event.key] = [time_now, 10 * self.increment_time] if event.type == pygame.KEYDOWN else None

        for key, val in self.pressed.items():
            if val is not None and time_now >= val[0]:
                self.pressed[key] = [val[0] + val[1], self.increment_time]
                self.cmd += self.cmd_increment_scale * self.increment[key]

        self.cmd = np.clip(self.cmd, -self.cmd_limits, self.cmd_limits)

        if any(prev_cmd != self.cmd):
            self.updateDisplay()

        return True


if __name__ == '__main__':
    param = Params()
    param.realtime_factor = 2
    param.camera_pitch = -30
    sim = PyBulletSim(param)

    k = KeyboardCommander()

    while k.processKeyEvents():
        sim.simulate_step(k.cmd[0], VELOCITY_MODE, k.cmd[1])

    sim.terminate()
