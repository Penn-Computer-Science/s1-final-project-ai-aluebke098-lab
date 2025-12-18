import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

# reset
# reward
# play(action) -> returns direction
# game_iteration -> keep track of the current frame
# is_collision

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)

BLOCK_SIZE = 40
SPEED = 20


class Viper:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
    #     # init display
    #     self.display = pygame.display.set_mode((self.w, self.h))
    #     pygame.display.set_caption('Snake')
    #     self.clock = pygame.time.Clock()
    #     self.reset()

    # def reset(self):
    #     # init game state
        
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    # def play_step(self, action):
    #     self.frame_iteration += 1
    #     # 1. collect user input
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             quit()
        
    #     # 2. move
    #     self._move(action) # update the head
    #     self.snake.insert(0, self.head)
        
    #     # 3. check if game over
    #     reward = 0
    #     game_over = False
    #     if self.is_collision() or self.frame_iteration > 100*len(self.snake):
    #         game_over = True
    #         reward = -10
    #         return reward, game_over, self.score
            
    #     # 4. place new food or just move
    #     if self.head == self.food:
    #         self.score += 1
    #         reward = 10
    #         self._place_food()
    #     else:
    #         self.snake.pop()
        
    #     # 5. update ui and clock
    #     self._update_ui()
    #     self.clock.tick(SPEED)
    #     # 6. return game over and score
    #     return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:] or pt in SnakeGameAI.snake:
            return True
        
        return False
        
    # def _update_ui(self):
    #     self.display.fill(BLACK)
        
    #     for pt in self.snake:
    #         pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
    #         pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
    #     pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
    #     text = font.render("Score: " + str(self.score), True, WHITE)
    #     self.display.blit(text, [0, 0])
    #     pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] #no change(straight)
        if np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn (+1 clockwise, modulus 4 to stay in range)
        else: # [0, 0, 1]
            next_idx = (idx -1) % 4
            new_dir = clock_wise[next_idx] # left turn (-1 clockwise, modulus 4 to stay in range)

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)


viper = Viper()


class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        self.snake = [0]
        viper.snake = [0]

    def reset(self):
        # init game state
        self.direction = Direction.LEFT
        
        self.head = Point(self.w*2/3, self.h*2/3)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        viper.direction = Direction.RIGHT
        
        viper.head = Point(self.w/3, self.h/3)
        viper.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        viper.score = 0
        viper.food = None
        viper._place_food()
        viper.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        viper._move(action)
        viper.snake.insert(0, viper.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        if viper.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        if viper.head == viper.food:
            viper.score += 1
            viper._place_food()
        else:
            viper.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:] or pt in viper.snake:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        for pt in viper.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(viper.food.x, viper.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        text_v = font.render("Score: " + str(viper.score), True, WHITE)
        self.display.blit(text, [0, 0])
        self.display.blit(text_v, [300, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] #no change(straight)
        if np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn (+1 clockwise, modulus 4 to stay in range)
        else: # [0, 0, 1]
            next_idx = (idx -1) % 4
            new_dir = clock_wise[next_idx] # left turn (-1 clockwise, modulus 4 to stay in range)

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)