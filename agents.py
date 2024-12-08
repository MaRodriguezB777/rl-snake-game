import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from util import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self, model_name=None):
        self.n_games = 0
        self.gamma = 0.9 # discount rate (now vs. future reward)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        
        self.model = Linear_QNet(
            11, # state size
            256, # hidden layer size
            3, # action size
        )

        if model_name is not None:
            print('Loading model: {}'.format(model_name))
            
            self.model.load_state_dict(torch.load(model_name))
            
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: SnakeGameAI):
        head = game.head # game.snake[0]

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        going_right = game.direction == Direction.RIGHT
        going_left = game.direction == Direction.LEFT
        going_up = game.direction == Direction.UP
        going_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (going_up and game.is_collision(point_u) or
             going_down and game.is_collision(point_d) or
             going_left and game.is_collision(point_l) or
             going_right and game.is_collision(point_r)),
            
            # Danger right
            (going_up and game.is_collision(point_r) or
             going_down and game.is_collision(point_l) or
             going_left and game.is_collision(point_u) or
             going_right and game.is_collision(point_d)),
            
            # Danger left
            (going_up and game.is_collision(point_l) or
             going_down and game.is_collision(point_r) or
             going_left and game.is_collision(point_d) or
             going_right and game.is_collision(point_u)), 
        
            # Move direction
            going_right,
            going_left,
            going_up,
            going_down,

            # Relative Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            batch = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            batch = self.memory

        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, prev_mean=None, curr_mean=None):
        # sometimes make random moves and sometimes make optimal moves
        # games_since_last_record = self.n_games - last_record_game
        # if prev_mean is None or curr_mean is None:
        #     epsilon = (1 - self.n_games / 80) / 3
        # else:
        #     epsilon = curr_mean / prev_mean * 0.01
        epsilon = 80 - self.n_games
        final_move = [0,0,0]

        if random.randint(0, 200) < epsilon:
            move_idx = random.randint(0, 2)
            final_move[move_idx] = 1
        else:
            state = torch.tensor(state, dtype=torch.float)
            pred_moves = self.model(state)
            move_idx = torch.argmax(pred_moves).item()
            final_move[move_idx] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    last_record_game = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        # get move
        if agent.n_games > 100:
            prev_mean = np.mean(plot_mean_scores[-100:-50])
            curr_mean = np.mean(plot_mean_scores[-50:])
            action = agent.get_action(state_old, prev_mean, curr_mean)
        else:
            action = agent.get_action(state_old)

        # perform move and get new data
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # remember
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games} | Score {score} | Record {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

def play():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    last_record_game = 0
    agent = Agent('model/model_1layer_256.pth')
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        # get move
        action = agent.get_action(state_old)

        # perform move and get new data
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1

            if score > record:
                record = score

            print(f'Game {agent.n_games} | Score {score} | Record {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    play()