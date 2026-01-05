import torch
import random
import numpy as np
from collections import deque
from snake_game_env import SnakeGameAI, Direction, Point, BLOCK_SIZE
from snake_model import Linear_QNet, QTrainer
from graph_helper import make_graphs
# from snake_agent import Agent
# from viper_agent import Agent_V

MAX_MEMORY = 100_000
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

    def get_state(self, game):
        head = game.snake[0] # front of snake
        point_l = Point(head.x - BLOCK_SIZE, head.y) # making points next to the head in all directions
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT # boolean, true if it's going in that direction
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            #move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y, # food down
            ]

        return np.array(state, dtype=int) #converts true/false to 1/0

    def get_state_v(self, game):
        head = game.viper[0] # front of snake
        point_l = Point(head.x - BLOCK_SIZE, head.y) # making points next to the head in all directions
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction_v == Direction.LEFT # boolean, true if it's going in that direction
        dir_r = game.direction_v == Direction.RIGHT
        dir_u = game.direction_v == Direction.UP
        dir_d = game.direction_v == Direction.DOWN

        state = [
            #danger straight
            (dir_r and game.is_collision_v(point_r)) or
            (dir_l and game.is_collision_v(point_l)) or
            (dir_u and game.is_collision_v(point_u)) or
            (dir_d and game.is_collision_v(point_d)),

            #danger right
            (dir_u and game.is_collision_v(point_r)) or
            (dir_d and game.is_collision_v(point_l)) or
            (dir_l and game.is_collision_v(point_u)) or
            (dir_r and game.is_collision_v(point_d)),

            #danger left
            (dir_d and game.is_collision_v(point_r)) or
            (dir_u and game.is_collision_v(point_l)) or
            (dir_r and game.is_collision_v(point_u)) or
            (dir_l and game.is_collision_v(point_d)),

            #move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #food location
            game.food_v.x < game.head_v.x, # food left
            game.food_v.x > game.head_v.x, # food right
            game.food_v.y < game.head_v.y, # food up
            game.food_v.y > game.head_v.y, # food down
            ]

        return np.array(state, dtype=int) #converts true/false to 1/0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # will popleft() if MAX_MEMORY is reached (remove earliest entry)
        # appends the data as a tuple

    def train_long_memory(self):
        if  len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # picks random samples from self.memory, len would equal BATCH_SIZE
        else:
            mini_sample = self.memory
        
        #extract infor from memory to use for training
        states, actions, rewards, next_states, dones = zip(*mini_sample) #tutorial doesn't know exactly how zip function works so idk either rn
        #train
        self.trainer.train_step(states, actions, rewards, next_states, dones) #vars all plural bc many data points
        #could also be done in a for loop

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #  random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games # epsilon gets smaller the more games are played
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon: # the smaller epsilon gets, the chance of getting a random value goes down
            move = random.randint(0,2) # gives the index of the move (0, 1, or 2)
            final_move[move] = 1 # sets the value to true
        else:
            state0 = torch.tensor(state, dtype=torch.float) # converts the state from a list into a tensor
            prediction = self.model(state0) #gives a prediction based on the current state, according to memory
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
    # agent_v = Agent_V()
    # game_v = Viper()
    done_v = False

    while True:
        # get old state
        state_old = agent.get_state(game)
        state_old_v = agent.get_state_v(game)

        # get move
        final_move = agent.get_action(state_old)
        final_move_v = agent.get_action(state_old_v)

        # perform move and get new state
        if not done and done_v:
            reward, done, score, reward_v, done_v, score_v = game.play_step(final_move, final_move_v)
        # if not done_v:
        #     reward_v, done_v, score_v = game_v.play_step(final_move)

        state_new = agent.get_state(game)
        # state_new_v = agent.get_state(game_v)

        #train short memory (only for one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # agent_v.train_short_memory(state_old_v, final_move_v, reward_v, state_new_v, done_v)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)
        # agent_v.remember(state_old_v, final_move_v, reward_v, state_new_v, done_v)

        if done:
            # train long memory - aka replay memory aka experience replay - train based on all prior information
            game.reset()
            # game_v.reset()
            agent.n_games += 1
            # agent_v.n_games += 1
            agent.train_long_memory()
            # agent_v.train_long_memory()

            done = False
            # done_v = False

            if score > record: # update high score
                record = score
                agent.model.save()
            # if score_v > record_v:
            #     record_v = score_v
            #     agent_v.model.save()

            print('Snake: Game', agent.n_games, 'Score', score, 'Record:', record)
            # print('Viper: Game', agent_v.n_games, 'Score', score_v, 'Record:', record_v)

            plot_scores.append(score)
            # plot_scores_v.append(score_v)
            total_score += score
            # total_score_v += score_v
            mean_score = total_score / agent.n_games
            # mean_score_v = total_score_v / agent_v.n_games
            plot_mean_scores.append(mean_score)
            # plot_mean_scores_v.append(mean_score_v)
            make_graphs(plot_scores, plot_mean_scores, [0], [0]) #temp values until both snakes work


if __name__ == '__main__':
    train()