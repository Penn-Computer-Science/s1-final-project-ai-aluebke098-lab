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

BLOCK_SIZE = 20
SPEED = 20

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        # init game state
        self.direction = Direction.LEFT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x+BLOCK_SIZE, self.head.y),
                      Point(self.head.x+(2*BLOCK_SIZE), self.head.y)]
        
        self.direction_v = Direction.RIGHT
        
        self.head_v = Point(self.w/2, self.h/2)
        self.viper = [self.head_v, 
                      Point(self.head_v.x-BLOCK_SIZE, self.head_v.y),
                      Point(self.head_v.x-(2*BLOCK_SIZE), self.head_v.y)]
        
        self.score = 0
        self.food = None
        self.score_v = 0
        self.food_v = None
        self._place_food()
        self._place_food_v()
        print("setup done")
        
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
        
    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        # 2. move
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)

        self._move_v(self.direction_v)
        self.viper.insert(0, self.head_v)
        
        # 3. check if game over
        game_over = False
        if self.is_collision():
            game_over = True
            return game_over, self.score
        if self.is_collision_v():
            game_over = True
            return game_over, self.score_v
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        if self.head_v == self.food_v:
            self.score_v += 1
            self._place_food_v()
        else:
            self.viper.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            print("hi")
            return True
        # hits itself
        if pt in self.snake[1:]: #or pt in self.viper:
            print("snake")
            return True
        return False
    
    def is_collision_v(self, pt=None):
        if pt is None:
            pt = self.head_v
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            print("bye")
            return True
        # hits itself
        if pt in self.viper[1:]: #or pt in self.snake:
            print("viper")
            return True
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        for pt in self.viper:
            pygame.draw.rect(self.display, RED, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.food_v.x, self.food_v.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score S: " + str(self.score), True, WHITE)
        text_v = font.render("Score V: " + str(self.score_v), True, WHITE)
        self.display.blit(text, [0, 0])
        self.display.blit(text_v, [300, 0])
        pygame.display.flip()
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

    def _move_v(self, direction):
        x = self.head_v.x
        y = self.head_v.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head_v = Point(x, y)

if __name__ == '__main__':
    game = SnakeGame()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()