import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


pygame.init()
# font = pygame.font.Font('arial.ttf', 25)
font = pygame.font.SysFont('CURLZ__', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

BLOCK_SIZE = 20
SPEED = 40

def initialize_game(w=640, h=480):
    display = pygame.display.set_mode((w, h))
    pygame.display.set_caption('Snake')
    clock = pygame.time.Clock()
    return display, clock

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display, self.clock = initialize_game(w, h)
        self.reset()


    def reset(self):
        self.direction = Direction.RIGHT
        self.snake = [Point(self.w/2 - i * BLOCK_SIZE, self.h/2) for i in range(3)]
        self.head = self.snake[0]
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()


    def _place_food(self):
        possible_positions = [Point(x, y) for x in range(0, self.w, BLOCK_SIZE) for y in range(0, self.h, BLOCK_SIZE)]
        valid_positions = [pos for pos in possible_positions if pos not in self.snake]
        self.food = random.choice(valid_positions)


    def play_step(self, action):
        self.frame_iteration += 1
        # Collect user input
        handle_quit = lambda: [pygame.quit(), quit()] if pygame.event.peek(pygame.QUIT) else None
        handle_quit()

        # Move
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        # Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # Return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
         # Check if point is outside the boundary
        boundary = (pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0)
        
        # Check if point hits itself
        self_collision = pt in self.snake[1:]

        return boundary or self_collision


    def draw_snake(self):
        for pt in self.snake:
            pygame.draw.rect(self.display, green, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

    def draw_food(self):
        pygame.draw.rect(self.display, white, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

    def draw_score(self):
        text = font.render("Score: " + str(self.score), True, white)
        self.display.blit(text, [0, 0])

    def _update_ui(self):
        self.display.fill(black)
        self.draw_snake()
        self.draw_food()
        self.draw_score()
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        action_map = {
            (1, 0, 0): clock_wise[idx],  # no change
            (0, 1, 0): clock_wise[(idx + 1) % 4],  # right turn
            (0, 0, 1): clock_wise[(idx - 1) % 4]  # left turn
        }

        new_dir = action_map[tuple(action)]
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



class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


import torch
import random
import numpy as np
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        
        state = [
            # Danger straight
            (game.direction == Direction.RIGHT and game.is_collision(Point(head.x + 20, head.y))) or 
            (game.direction == Direction.LEFT and game.is_collision(Point(head.x - 20, head.y))) or 
            (game.direction == Direction.UP and game.is_collision(Point(head.x, head.y - 20))) or 
            (game.direction == Direction.DOWN and game.is_collision(Point(head.x, head.y + 20))),

            # Danger right
            (game.direction == Direction.UP and game.is_collision(Point(head.x + 20, head.y))) or 
            (game.direction == Direction.DOWN and game.is_collision(Point(head.x - 20, head.y))) or 
            (game.direction == Direction.LEFT and game.is_collision(Point(head.x, head.y - 20))) or 
            (game.direction == Direction.RIGHT and game.is_collision(Point(head.x, head.y + 20))),

            # Danger left
            (game.direction == Direction.DOWN and game.is_collision(Point(head.x + 20, head.y))) or 
            (game.direction == Direction.UP and game.is_collision(Point(head.x - 20, head.y))) or 
            (game.direction == Direction.RIGHT and game.is_collision(Point(head.x, head.y - 20))) or 
            (game.direction == Direction.LEFT and game.is_collision(Point(head.x, head.y + 20))),
            

            # Move direction
            game.direction == Direction.LEFT,
            game.direction == Direction.RIGHT,
            game.direction == Direction.UP,
            game.direction == Direction.DOWN,

            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()