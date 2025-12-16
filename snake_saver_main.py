import torch
import random
import numpy as np
from collections import deque
from snake_game_env import SnakeGameAI, Viper, Direction, Point, BLOCK_SIZE
from snake_model import Linear_QNet, QTrainer
from graph_helper import make_graphs
from snake_agent import Agent
from viper_agent import Agent_V

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
    agent_v = Agent_V()
    game_v = Viper()
    done_v = False

    while True:
        # get old state
        state_old = agent.get_state(game)
        state_old_v = agent_v.get_state(game_v)

        # get move
        final_move = agent.get_action(state_old)
        final_move_v = agent_v.get_action(state_old_v)

        # perform move and get new state
        if not done:
            reward, done, score = game.play_step(final_move)
        if not done_v:
            reward_v, done_v, score_v = game_v.play_step(final_move)

        state_new = agent.get_state(game)
        state_new_v = agent.get_state(game_v)

        #train short memory (only for one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent_v.train_short_memory(state_old_v, final_move_v, reward_v, state_new_v, done_v)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)
        agent_v.remember(state_old_v, final_move_v, reward_v, state_new_v, done_v)

        if done or done_v:
            # train long memory - aka replay memory aka experience replay - train based on all prior information
            game.reset()
            game_v.reset()
            agent.n_games += 1
            agent_v.n_games += 1
            agent.train_long_memory()
            agent_v.train_long_memory()

            done = False
            done_v = False

            if score > record: # update high score
                record = score
                agent.model.save()
            if score_v > record_v:
                record_v = score_v
                agent_v.model.save()

            print('Snake: Game', agent.n_games, 'Score', score, 'Record:', record)
            print('Viper: Game', agent_v.n_games, 'Score', score_v, 'Record:', record_v)

            plot_scores.append(score)
            plot_scores_v.append(score_v)
            total_score += score
            total_score_v += score_v
            mean_score = total_score / agent.n_games
            mean_score_v = total_score_v / agent_v.n_games
            plot_mean_scores.append(mean_score)
            plot_mean_scores_v.append(mean_score_v)
            make_graphs(plot_scores, plot_mean_scores, plot_scores_v, plot_mean_scores_v)


if __name__ == '__main__':
    train()