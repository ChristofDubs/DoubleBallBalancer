"""Script for interactively contolling the 3D Double Ball Balancer in the pybullet simulation
"""
import time
from enum import Enum

import numpy as np
import pygame
from pybullet_simulation import VELOCITY_MODE, PyBulletSim

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
    class Action(Enum):
        NONE = 1
        QUIT = 2
        RESPAWN = 3

    def __init__(self):
        self.cmd_scale = 100

        self.cmd = np.zeros(2)
        self.cmd_limits = np.array([1.5, 1]) * self.cmd_scale
        self.increment_time = 0.05

        self.increment_count = -8 * np.ones(2)
        self.cmd_increments = {pygame.K_UP: np.array([1, 0]),
                               pygame.K_DOWN: np.array([-1, 0]),
                               pygame.K_LEFT: np.array([0, 1]),
                               pygame.K_RIGHT: np.array([0, -1])}

        self.increment_increments = {pygame.K_w: np.array([1, 0]),
                                     pygame.K_s: np.array([-1, 0]),
                                     pygame.K_a: np.array([0, 1]),
                                     pygame.K_d: np.array([0, -1])}

        self.pressed = {key: None for key in list(self.cmd_increments.keys()) + list(self.increment_increments.keys())}

        pygame.init()
        self.display_surface = pygame.display.set_mode((700, 350))
        pygame.display.set_caption('Keyboard Control Panel')

        self.font = pygame.font.SysFont('liberationsans', 20)

        self.text = "After clicking on this window: \n" \
                    "- press Q / Esc to quit \n" \
                    "- press space bar to stop the robot \n" \
                    "- press R to respawn the robot \n" \
                    "- press / hold the arrow keys to change the speed commands \n" \
                    "- press the W/S/A/D keys to change the speed increments \n"

    def updateDisplay(self, realtime_factor):
        self.display_surface.fill(color=pygame.Color('black'))
        text = "speed commands: forward: {0:.2f}  turn: {1:.2f}".format(*list(self.getCommand())) + "\n\nincrements: forward: {0:.2f}  turn: {1:.2f}".format(
            *list(self.getCommandIncrements() / self.cmd_scale)) + "\n realtime factor: {:.2f} \n\n".format(realtime_factor) + self.text
        blit_text(self.display_surface, text, (20, 20), self.font)
        pygame.display.update()

    def processKeyEvents(self) -> Action:
        time_now = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in [
                    pygame.K_ESCAPE, pygame.K_q]):
                return self.Action.QUIT
            elif event.type == pygame.KEYDOWN and event.key in [pygame.K_SPACE, pygame.K_r]:
                self.cmd = np.zeros(2)
                if event.key == pygame.K_r:
                    return self.Action.RESPAWN
            elif event.type in [pygame.KEYDOWN, pygame.KEYUP] and event.key in self.pressed.keys():
                self.pressed[event.key] = [time_now, 10 * self.increment_time] if event.type == pygame.KEYDOWN else None

        for key, val in self.pressed.items():
            if val is not None and time_now >= val[0]:
                self.pressed[key] = [val[0] + val[1], self.increment_time]
                if key in self.cmd_increments:
                    self.cmd += self.cmd_increments[key] * self.getCommandIncrements()
                if key in self.increment_increments:
                    self.increment_count += self.increment_increments[key]

        self.cmd = np.clip(self.cmd, -self.cmd_limits, self.cmd_limits)
        self.increment_count = np.clip(self.increment_count, -30 * np.ones(2), np.zeros(2))

        return self.Action.NONE

    def getCommand(self):
        return self.cmd / self.cmd_scale

    def getCommandIncrements(self):
        return self.cmd_limits * 2**(1 + self.increment_count / 4)


if __name__ == '__main__':
    sim = PyBulletSim()

    k = KeyboardCommander()

    filtered_rtf = 1
    default_filter_constant = 0.05

    action = k.Action.NONE
    while action != k.Action.QUIT:
        actual_rtf = sim.simulate_step(k.getCommand()[0], VELOCITY_MODE, -k.getCommand()[1])
        filter_constant = min(1, default_filter_constant / actual_rtf**2)
        filtered_rtf = (1 - filter_constant) * filtered_rtf + filter_constant * actual_rtf
        action = k.processKeyEvents()
        k.updateDisplay(filtered_rtf)
        if action == k.Action.RESPAWN:
            sim.respawn()

    sim.terminate()
