import pygame
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Deep Q Network (DQN) model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()

# Window size
frame_size_x = 480
frame_size_y = 360
screen = pygame.display.set_mode((frame_size_x, frame_size_y))

# Colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

# Snake parameters
snake_pos = [360, 240]
snake_body = [[360, 240]]
food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
food_spawn = True
direction = 'RIGHT'
change_to = direction
score = 0

# Game variables
speed = 50
total_episodes = 1000  # Number of training episodes

# Deep Q-learning parameters
GAMMA = 0.9  # discount factor
EPSILON = 80  # exploration rate
MEMORY_SIZE = 10000  # replay memory size
BATCH_SIZE = 64  # minibatch size
UPDATE_TARGET_FREQ = 100  # update target network frequency
LEARNING_RATE = 0.001  # learning rate

# Initialize replay memory D with capacity N
replay_memory = []

# Initialize Q-networks: main network and target network
input_size = 6  # size of the state vector
output_size = 4  # number of actions
main_dqn = DQN(input_size, output_size)
target_dqn = DQN(input_size, output_size)
target_dqn.load_state_dict(main_dqn.state_dict())
target_dqn.eval()  # target network is not trainable

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(main_dqn.parameters(), lr=LEARNING_RATE)

# Lists to store scores and episode numbers for tracking
scores = []
episodes_list = []

def preprocess_state(snake_pos, food_pos):
    # Preprocess state: compute relative positions
    return np.array([snake_pos[0] - food_pos[0], snake_pos[1] - food_pos[1], 
                     food_pos[0] - snake_pos[0], food_pos[1] - snake_pos[1], 
                     snake_pos[0], snake_pos[1]])

def choose_action(state):
    if random.random() < EPSILON:
        return random.randrange(output_size)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = main_dqn(state_tensor)
            return q_values.argmax().item()

def get_reward(snake_pos, food_pos):
    if snake_pos == food_pos:
        return 10  # Increase the reward for finding food
    elif snake_pos[0] < 0 or snake_pos[0] >= frame_size_x or snake_pos[1] < 0 or snake_pos[1] >= frame_size_y:
        return -10  # Decrease the penalty for hitting the wall
    elif snake_pos in snake_body[1:]:
        return -10  # Decrease the penalty for colliding with itself
    else:
        return -1  # A small penalty for not finding food

def update_replay_memory(state, action, reward, next_state):
    replay_memory.append((state, action, reward, next_state))
    if len(replay_memory) > MEMORY_SIZE:
        replay_memory.pop(0)

def train_dqn():
    if len(replay_memory) < BATCH_SIZE:
        return

    # Sample minibatch from replay memory
    minibatch = random.sample(replay_memory, BATCH_SIZE)
    states, actions, rewards, next_states = zip(*minibatch)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    # Compute Q-values for current states and next states
    q_values = main_dqn(states).gather(1, actions)
    next_q_values = target_dqn(next_states).max(1)[0].detach()
    target_q_values = rewards + GAMMA * next_q_values

    # Update main network's weights using TD-error
    loss = criterion(q_values, target_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
for episode in range(total_episodes):
    snake_pos = [360, 240]
    snake_body = [[360, 240]]
    food_pos = [random.randrange(1, (frame_size_x // 10)) * 10, random.randrange(1, (frame_size_y // 10)) * 10]
    food_spawn = True
    direction = 'RIGHT'
    change_to = direction
    score = 0

    while True:
        # Choose action based on Q-values
        state = preprocess_state(snake_pos, food_pos)
        action = choose_action(state)

        # Apply action to move the snake
        if action == 0:  # UP
            direction = 'UP'
        elif action == 1:  # DOWN
            direction = 'DOWN'
        elif action == 2:  # LEFT
            direction = 'LEFT'
        elif action == 3:  # RIGHT
            direction = 'RIGHT'

        # Moving the snake
        if direction == 'UP':
            snake_pos[1] -= 10
        elif direction == 'DOWN':
            snake_pos[1] += 10
        elif direction == 'LEFT':
            snake_pos[0] -= 10
        elif direction == 'RIGHT':
            snake_pos[0] += 10

        # Snake body growing mechanism
        snake_body.insert(0, list(snake_pos))
        if snake_pos == food_pos:
            score += 1
            food_spawn = False
        else:
            snake_body.pop()

        # Spawning food on the screen
        if not food_spawn:
            food_pos = [random.randrange(1, (frame_size_x // 10)) * 10,
                        random.randrange(1, (frame_size_y // 10)) * 10]
            food_spawn = True

        # Game Over conditions
        if snake_pos[0] < 0 or snake_pos[0] > frame_size_x - 10:
            break
        if snake_pos[1] < 0 or snake_pos[1] > frame_size_y - 10:
            break
        if snake_pos in snake_body[1:]:
            break

        # Get reward and next state
        reward = get_reward(snake_pos, food_pos)
        next_state = preprocess_state(snake_pos, food_pos)

        # Update replay memory and train DQN
        update_replay_memory(state, action, reward, next_state)
        train_dqn()

        # Update tracking lists
        scores.append(score)
        episodes_list.append(episode)

        # Update screen
        screen.fill(black)

        for pos in snake_body:
            pygame.draw.rect(screen, green, pygame.Rect(pos[0], pos[1], 20, 20))
        pygame.draw.rect(screen, white, pygame.Rect(food_pos[0], food_pos[1], 20, 20))
        pygame.display.flip()
        clock.tick(speed)

    # Update target network every UPDATE_TARGET_FREQ episodes
    if episode % UPDATE_TARGET_FREQ == 0:
        target_dqn.load_state_dict(main_dqn.state_dict())

# Plot scores
plt.plot(episodes_list, scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Deep Q-Learning Snake')
plt.show()
