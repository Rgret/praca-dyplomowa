from collections import deque
from gymnasium import spaces
import math
import gymnasium as gym
import random
import numpy as np
import pygame
import sys
import time

class Bullet():
    x = 0; x_velocity = 4
    y = 0; y_velocity = 4
    size = 15
    def __init__(self):
        self.x = random.randint(0, 100) if random.randint(0,1) else random.randint(0, 100) + 490
        self.y = random.randint(0, 100) if random.randint(0,1) else random.randint(0, 100) + 490
        self.x_velocity = random.randint(-8, 8)
        self.y_velocity = random.randint(-8, 8)
        self.speed = self.x_velocity if self.x_velocity > self.y_velocity else self.y_velocity
        self.speed = self.speed if self.speed > 0 else self.speed * -1
        self.size = random.randint(8, 20)

    def move(self, player_pos):
        if self.x + self.x_velocity < 0 or self.x + self.x_velocity > 600: 
            self.x_velocity = -(self.x_velocity + random.random())
            if random.random() < 0.3:
                x = player_pos[0] - self.x
                y = player_pos[1] - self.y
                d = math.sqrt(x**2 + y**2)
                x=x/d; y=y/d
                self.x_velocity = x * 4
                self.y_velocity = y * 4
            #del self
            #return
        if self.y + self.y_velocity < 0 or self.y + self.y_velocity > 600: 
            self.y_velocity = -(self.y_velocity + random.random())
            if random.random() < 0.3:
                x = player_pos[0] - self.x
                y = player_pos[1] - self.y
                d = math.sqrt(x**2 + y**2)
                x=x/d; y=y/d
                self.x_velocity = x * 6
                self.y_velocity = y * 6
            #del self
            #return
        
        self.x += self.x_velocity
        self.y += self.y_velocity

    def position(self):
        return [self.x, self.y]

class BulletHell(gym.Env):
    max_bullets = random.randint(15,20)
    player_speed = 4
    moves = deque(maxlen=10)
    def __init__(self, mode):
        super(BulletHell, self).__init__()

        self.action_space = spaces.Box(low=0, high=6, shape=(6,1))
        self.observation_space = spaces.Box(low=0, high=255, shape=(5,42), dtype=np.uint16)

        self.player_position = [300, 300]
        self.player_velocity = [0, 0]
        self.bullets = []

        self.mode = mode
        pygame.init()
        if mode == "human":
            self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()

    def reset(self):
        self.player_position = [300, 300]
        self.player_velocity = [0, 0]
        self.bullets = []
        return self._get_observation()
    
    def step(self, action):
        if action == 0:
            self.player_velocity = [0, 0]
        # ^ up
        if action == 1 and self.player_position[1] > 0:
            self.player_position[1] = max(0, self.player_position[1] - self.player_speed)
            self.player_velocity[1] = self.player_speed
        # > ^ right-up
        # if action == 2:
        #     self.player_position[0] = min(600, self.player_position[0] + 1)
        #     self.player_position[1] = min(600, self.player_position[1] - 1)
        #     self.player_velocity[0] = 1
        #     self.player_velocity[1] = -1
        # > right
        if action == 2 and self.player_position[0] < 590:
            self.player_position[0] = min(600, self.player_position[0] + self.player_speed)
            self.player_velocity[0] = self.player_speed
        # > \/ right-down
        # if action == 4:
        #     self.player_position[0] = min(600, self.player_position[0] + 1)
        #     self.player_position[1] = min(600, self.player_position[1] + 1)
        #     self.player_velocity[0] = 1
        #     self.player_velocity[1] = 1
        # \/ down
        if action == 3 and self.player_position[1] < 590:
            self.player_position[1] = min(600, self.player_position[1] + self.player_speed)
            self.player_velocity[1] = self.player_speed
        # < \/ left-down
        # if action == 6:
        #     self.player_position[0] = max(0, self.player_position[0] - 1)
        #     self.player_position[1] = min(600, self.player_position[1] + 1)
        #     self.player_velocity[0] = -1
        #     self.player_velocity[1] = 1
        # < left
        if action == 4 and self.player_position[0] > 0:
            self.player_position[0] = max(0, self.player_position[0] - self.player_speed)
            self.player_velocity[1] = -self.player_speed
        # < ^ left-up
        # if action == 8:
        #     self.player_position[0] = max(0, self.player_position[0] - 1)
        #     self.player_position[1] = max(0, self.player_position[1] - 1)
        #     self.player_velocity[0] = -1
        #     self.player_velocity[1] = 1
            
        self.moves.append(action)

        reward = 0.0
        done = False

        if np.random.rand() < 0.2 and len(self.bullets) < self.max_bullets:
            random.seed(time.time())
            self.bullets.append(Bullet())

        for bullet in self.bullets:
            bullet.move(self.player_position)
            if (
                bullet.x - bullet.size <= self.player_position[0] <= bullet.x + bullet.size
                and bullet.y - bullet.size <= self.player_position[1] <= bullet.y + bullet.size
            ):
                reward = -1.0
                done = True
                break
            else:
                reward = 1.0
                done = False
        if self.player_position[0] > 589 or self.player_position[0] < 14 or self.player_position[1] > 589 or self.player_position[1] < 14:
            reward = -1.0
            done = True
        
        return self._get_observation(), reward, done, {}
    
    def render(self):
        if self.mode != 'human': 
            return
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (0, 255, 0), (*self.player_position, 15, 15))
        for bullet in self.bullets:
            pygame.draw.rect(self.screen, (255, 0, 0), (bullet.x, bullet.y, bullet.size, bullet.size))
        pygame.display.flip()
        self.clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        pygame.quit()
    
    def _get_observation(self):
        observation = np.zeros((42, 5), dtype=np.uint8)
        observation[0] = [self.player_position[0], self.player_position[1], self.player_velocity[0], self.player_velocity[1], 15]
        for i, bullet in enumerate(self.bullets):
            observation[i+1] = [bullet.x, bullet.y, bullet.x_velocity, bullet.y_velocity, bullet.size]
        return observation