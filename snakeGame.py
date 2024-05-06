import pygame
import random
import numpy as np
import time
import matplotlib.pyplot as plt

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
snake_body = [[360, 240], [340, 240], [320, 240], [300, 240]]  # Initial snake body with four segments
food_pos = [random.randrange(1, (frame_size_x//10)) * 10, random.randrange(1, (frame_size_y//10)) * 10]
food_spawn = True
direction = 'RIGHT'
change_to = direction
score = 0

# Game variables
speed = 50
total_episodes = 100  # Number of training episodes

# Q-learning parameters
Q_VALUES = {}
ALPHA = 0.01  # learning rate
GAMMA = 0.9  # discount factor
EPSILON = 80  # exploration rate
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Lists to store scores and episode numbers for tracking
scores = []
episodes_list = []

def show_score(choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(score), True, color)
    score_rect = score_surface.get_rect()
    if choice == 1:
        score_rect.midtop = (frame_size_x / 10, 15)
    else:
        score_rect.midtop = (frame_size_x / 2, frame_size_y / 1.25)
    screen.blit(score_surface, score_rect)

# Create a figure for live tracking
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Episode')
ax.set_ylabel('Score')
line, = ax.plot(episodes_list, scores)

def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)
    else:
        return max(ACTIONS, key=lambda x: Q_VALUES.get((state, x), 0))
    

def get_state(snake_pos, food_pos, direction):
    if direction == 'UP':
        return (snake_pos[0] - food_pos[0], snake_pos[1] - food_pos[1], 1)
    elif direction == 'DOWN':
        return (food_pos[0] - snake_pos[0], food_pos[1] - snake_pos[1], 2)
    elif direction == 'LEFT':
        return (snake_pos[1] - food_pos[1], food_pos[0] - snake_pos[0], 3)
    elif direction == 'RIGHT':
        return (food_pos[1] - snake_pos[1], snake_pos[0] - food_pos[0], 4)


def get_reward(snake_pos, food_pos):
    if snake_pos == food_pos:
        return 10  # Increase the reward for finding food
    elif snake_pos[0] < 0 or snake_pos[0] >= frame_size_x or snake_pos[1] < 0 or snake_pos[1] >= frame_size_y:
        return -10  # Decrease the penalty for hitting the wall
    elif snake_pos in snake_body[1:]:
        return -10  # Decrease the penalty for colliding with itself
    else:
        return -1  # A small penalty for not finding food

def update_q_value(state, action, reward, next_state):
    max_next_q_value = max([Q_VALUES.get((next_state, a), 0) for a in ACTIONS])
    Q_VALUES[(state, action)] = (1 - ALPHA) * Q_VALUES.get((state, action), 0) + ALPHA * (reward + GAMMA * max_next_q_value)
    

def game_over():
    my_font = pygame.font.SysFont('CURLZ__', 50)
    game_over_surface = my_font.render('Your Score is : ' + str(score), True, red)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (frame_size_x / 2, frame_size_y / 4)
    screen.fill(black)
    screen.blit(game_over_surface, game_over_rect)
    pygame.display.flip()
    time.sleep(2)

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
        state = get_state(snake_pos, food_pos, direction)
        action = choose_action(state)

		# Apply action to move the snake
        if action == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if action == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if action == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if action == 'RIGHT' and direction != 'LEFT':
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
            game_over()
            break
        if snake_pos[1] < 0 or snake_pos[1] > frame_size_y - 10:
            game_over()
            break
        for block in snake_body[1:]:
            if snake_pos == block:
                game_over()
                break

        # Update Q-values
        next_state = get_state(snake_pos, food_pos, direction)
        reward = get_reward(snake_pos, food_pos)
        update_q_value(state, action, reward, next_state)

        # Update tracking lists
        scores.append(score)
        episodes_list.append(episode)

        # Update live tracking plot
        line.set_xdata(episodes_list)
        line.set_ydata(scores)
        ax.relim()
        ax.autoscale_view()

        # Pause for a short while to allow live tracking to update
        plt.pause(0.001)

        # Update screen
        screen.fill(black)

        for pos in snake_body:
            pygame.draw.rect(screen, green, pygame.Rect(pos[0], pos[1], 20, 20))
        pygame.draw.rect(screen, white, pygame.Rect(food_pos[0], food_pos[1], 20, 20))
        pygame.display.flip()
        clock.tick(speed)

# Keep the plot open after the training loop ends
plt.ioff()
plt.show()
