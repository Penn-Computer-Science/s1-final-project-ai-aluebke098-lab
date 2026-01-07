import torch
import random
import numpy as np
from collections import deque
from snake_game_env import SnakeGameAI, Direction, Point, BLOCK_SIZE
from snake_model import Linear_QNet, QTrainer
from graph_helper import make_graphs
# from snake_agent import Agent
# from viper_agent import Agent_V

MAX_MEMORY = 50_000
BATCH_SIZE = 1_000
LR = 0.001 #learning rate

class Agent:

    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # randomness factor
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen = MAX_MEMORY) # popleft() - automatically done when it exceeds the mas memory, removes the left (firstmost) element 
        self.model = Linear_QNet(11, 256, 3) # layers: 11 input, 256 hidden, 3 output
        self.trainer = QTrainer(self.model, lr=LR, gamma = self.gamma)

        self.memory_v = deque(maxlen = MAX_MEMORY)
        self.model_v = Linear_QNet(11, 256, 3)
        self.trainer_v = QTrainer(self.model_v, lr=LR, gamma = self.gamma)

    def get_state(self, game, snake):
        if snake == "snake":
            head = game.snake[0]
            direction = game.direction
            collision = game.is_collision
            food = game.food
        elif snake == "viper":
            head = game.viper[0]
            direction = game.direction_v
            collision = game.is_collision_v
            food = game.food_v
        
        point_l = Point(head.x - BLOCK_SIZE, head.y) # making points next to the head in all directions
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = direction == Direction.LEFT # boolean, true if it's going in that direction
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        state = [
            #danger straight
            (dir_r and collision(point_r)) or
            (dir_l and collision(point_l)) or
            (dir_u and collision(point_u)) or
            (dir_d and collision(point_d)),

            #danger right
            (dir_u and collision(point_r)) or
            (dir_d and collision(point_l)) or
            (dir_l and collision(point_u)) or
            (dir_r and collision(point_d)),

            #danger left
            (dir_d and collision(point_r)) or
            (dir_u and collision(point_l)) or
            (dir_r and collision(point_u)) or
            (dir_l and collision(point_d)),

            #move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #food location
            food.x < head.x, # food left
            food.x > head.x, # food right
            food.y < head.y, # food up
            food.y > head.y, # food down
            ]

        return np.array(state, dtype=int) #converts true/false to 1/0

    def remember(self, state, action, reward, next_state, done, snake):
        if snake == "snake":
            self.memory.append((state, action, reward, next_state, done)) # will popleft() if MAX_MEMORY is reached (remove earliest entry)
        # appends the data as a tuple
        elif snake == "viper":
            self.memory_v.append((state, action, reward, next_state, done))

    def train_long_memory(self, snake):
        if snake == "snake":
            mem = self.memory
            train = self.trainer
        elif snake == "viper":
            mem = self.memory_v
            train = self.trainer_v
        
        if  len(mem) > BATCH_SIZE:
            mini_sample = random.sample(mem, BATCH_SIZE) # picks random samples from memory, len would equal BATCH_SIZE
        else:
            mini_sample = mem
        
        #extract infor from memory to use for training
        states, actions, rewards, next_states, dones = zip(*mini_sample) #tutorial doesn't know exactly how zip function works so idk either rn
        #train
        train.train_step(states, actions, rewards, next_states, dones) #vars all plural bc many data points
        #could also be done in a for loop

    def train_short_memory(self, state, action, reward, next_state, done, snake):
        if snake == "snake":
            self.trainer.train_step(state, action, reward, next_state, done)
        elif snake == "viper":
            self.trainer_v.train_step(state, action, reward, next_state, done)

    def get_action(self, state, snake):
        #  random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games # epsilon gets smaller the more games are played
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon: # the smaller epsilon gets, the chance of getting a random value goes down
            move = random.randint(0,2) # gives the index of the move (0, 1, or 2)
            final_move[move] = 1 # sets the value to true
        else:
            state0 = torch.tensor(state, dtype=torch.float) # converts the state from a list into a tensor
            if snake == "snake":
                prediction = self.model(state0) #gives a prediction based on the current state, according to memory
            elif snake == "viper":
                prediction = self.model_v(state0)
            move = torch.argmax(prediction).item() #converts it to only one number (index of move)
            final_move[move] = 1 

        return final_move

def train():
    plot_scores = [] #score after each game as points for graph
    plot_mean_scores = [] #average score after each game as points for graph
    total_score = 0 #cumultive score total
    record = 0 #high score
    agent = Agent()
    game = SnakeGameAI()
    done = False

    plot_scores_v = [] 
    plot_mean_scores_v = [] 
    total_score_v = 0 
    record_v = 0 
    done_v = False

    snake = "snake"
    viper = "viper"

    while True:
        # get old state
        state_old = agent.get_state(game, snake)
        state_old_v = agent.get_state(game, viper)

        # get move
        final_move = agent.get_action(state_old, snake)
        final_move_v = agent.get_action(state_old_v, viper)

        # perform move and get new state
        if not done or not done_v:
            reward, done, score, reward_v, done_v, score_v = game.play_step(final_move, final_move_v)

        state_new = agent.get_state(game, snake)
        state_new_v = agent.get_state(game, viper)

        #train short memory (only for one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done, snake)
        agent.train_short_memory(state_old_v, final_move_v, reward_v, state_new_v, done_v, viper)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done, snake)
        agent.remember(state_old_v, final_move_v, reward_v, state_new_v, done_v, viper)

        if done and done_v:
            # train long memory - aka replay memory aka experience replay - train based on all prior information
            game.reset()
            agent.n_games += 1
            agent.train_long_memory(snake)
            agent.train_long_memory(viper)

            done = False
            done_v = False

            if score > record: # update high score
                record = score
                agent.model.save()
            if score_v > record_v:
                record_v = score_v
                agent.model_v.save('viper.pth')

            print('Game', agent.n_games, '\nSnake Score:', score, 'Record:', record, '\nViper Score:', score_v,'Record:', record_v)

            plot_scores.append(score)
            plot_scores_v.append(score_v)
            total_score += score
            total_score_v += score_v
            mean_score = total_score / agent.n_games
            mean_score_v = total_score_v / agent.n_games
            plot_mean_scores.append(mean_score)
            plot_mean_scores_v.append(mean_score_v)
            make_graphs(plot_scores, plot_mean_scores, plot_scores_v, plot_mean_scores_v)


if __name__ == '__main__':
    train()