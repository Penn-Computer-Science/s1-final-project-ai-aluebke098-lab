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
PURPLE = (179, 0, 179)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (31, 122, 31)
GREEN2 = (37, 142, 37)

BLOCK_SIZE = 20
SPEED = 20

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # initiate display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # initiate game state
        self.direction = Direction.LEFT
        
        self.head = Point(self.w/2 +(5*BLOCK_SIZE), self.h/2 +(5*BLOCK_SIZE))
        self.snake = [self.head, 
                      Point(self.head.x+BLOCK_SIZE, self.head.y),
                      Point(self.head.x+(2*BLOCK_SIZE), self.head.y)]
        
        self.direction_v = Direction.RIGHT
        
        self.head_v = Point(self.w/2 -(5*BLOCK_SIZE), self.h/2 -(5*BLOCK_SIZE))
        self.viper = [self.head_v, 
                      Point(self.head_v.x-BLOCK_SIZE, self.head_v.y),
                      Point(self.head_v.x-(2*BLOCK_SIZE), self.head_v.y)]
        
        self.score = 0
        self.food = None
        self.score_v = 0
        self.food_v = None
        
        self._place_food()
        self.frame_iteration = 0
        self._place_food_v()
        self.frame_iteration_v = 0

        self.game_over = False
        self.game_over_v = False
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.viper or self.food == self.food_v:
            self._place_food()

    def _place_food_v(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food_v = Point(x, y)
        if self.food_v in self.snake or self.food_v in self.viper or self.food_v == self.food:
            self._place_food_v()
        
    def play_step(self, action, action_v):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        if not self.game_over:
            self._move(action) # update the head
            self.snake.insert(0, self.head)
        if not self.game_over_v:
            self._move_v(action_v)
            self.viper.insert(0, self.head_v)
        
        reward = 0
        reward_v = 0

        if not self.game_over:
            # 3. check if game over
            if self.is_collision() or self.frame_iteration > 100*len(self.snake):
                self.game_over = True
                reward = -10
            # 4. place new food or just move
            if self.head == self.food:
                self.score += 1
                reward = 10
                self._place_food()
            else:
                self.snake.pop()
        
        if not self.game_over_v:
            # 3. check if game over
            if self.is_collision_v() or self.frame_iteration_v > 100*len(self.viper):
                self.game_over_v = True
                reward_v = -10
            # 4. place new food or just move
            if self.head_v == self.food_v:
                self.score_v += 1
                reward_v = 10
                self._place_food_v()
            else:
                self.viper.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, self.game_over, self.score, reward_v, self.game_over_v, self.score_v
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:] or pt in self.viper:
            return True
        
        return False
    
    def is_collision_v(self, pt=None):
        if pt is None:
            pt = self.head_v
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.viper[1:] or pt in self.snake:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, BLOCK_SIZE*3/5, BLOCK_SIZE*3/5))
        for pt in self.viper:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, BLOCK_SIZE*3/5, BLOCK_SIZE*3/5))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, PURPLE, pygame.Rect(self.food_v.x, self.food_v.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score S: " + str(self.score), True, WHITE)
        text_v = font.render("Score V: " + str(self.score_v), True, WHITE)
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

    def _move_v(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction_v)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] #no change(straight)
        if np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn (+1 clockwise, modulus 4 to stay in range)
        else: # [0, 0, 1]
            next_idx = (idx -1) % 4
            new_dir = clock_wise[next_idx] # left turn (-1 clockwise, modulus 4 to stay in range)

        self.direction_v = new_dir

        x = self.head_v.x
        y = self.head_v.y
        if self.direction_v == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction_v == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction_v == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction_v == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head_v = Point(x, y)