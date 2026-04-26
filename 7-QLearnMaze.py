# 7 ONE-CELL COMPLETE Q-LEARNING MAZE (CLEAN + FIXED)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from gymnasium import spaces

# ----- Environment -----
class MazeEnv(gym.Env):
    def __init__(self):
        self.maze = np.array([
            [0,0,0,0,0],
            [1,1,1,1,0],
            [0,0,0,0,0],
            [0,1,1,1,1],
            [0,0,0,0,0]
        ])
        self.start, self.goal = (0,0), (4,4)
        self.state = self.start
        self.action_space = spaces.Discrete(4)

    def reset(self):
        self.state = self.start
        return self.state, {}

    def step(self, action):
        x, y = self.state
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        nx, ny = x + moves[action][0], y + moves[action][1]

        if not (0 <= nx < 5 and 0 <= ny < 5 and self.maze[nx, ny] == 0):
            return self.state, -5, False, False, {}

        self.state = (nx, ny)

        if self.state == self.goal:
            return self.state, 100, True, False, {}

        return self.state, -1, False, False, {}

    def render(self):
        grid = self.maze.copy()
        x, y = self.state
        grid[x, y] = 8
        plt.imshow(grid)
        plt.title("Maze Agent")
        plt.axis("off")
        plt.show()

# ----- Setup -----
env = MazeEnv()
q_table = {(x,y): np.zeros(4) for x in range(5) for y in range(5)}

alpha, gamma = 0.1, 0.9
epsilon, decay, min_eps = 1.0, 0.995, 0.01
episodes = 1000

rewards = []

# ----- Training -----
for _ in range(episodes):
    state, _ = env.reset()
    total = 0

    for _ in range(50):
        action = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)

        q_table[state][action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
        )

        state = next_state
        total += reward

        if done:
            break

    epsilon = max(min_eps, epsilon * decay)
    rewards.append(total)

# ----- Plot Training -----
plt.plot(rewards)
plt.title("Training Progress")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()

# ----- Test Path -----
state, _ = env.reset()
path = [state]

for _ in range(50):
    action = np.argmax(q_table[state])
    state, _, done, _, _ = env.step(action)
    path.append(state)
    if done:
        break

print("Path:", path)
print("Reached Goal!" if state == env.goal else "Failed")

# ----- Success Rate -----
success = 0
for _ in range(50):
    state, _ = env.reset()
    for _ in range(50):
        state, _, done, _, _ = env.step(np.argmax(q_table[state]))
        if done:
            success += 1
            break

print("Success Rate:", success/50)

# ----- Render Path -----
state, _ = env.reset()
for _ in range(20):
    env.render()
    action = np.argmax(q_table[state])
    state, _, done, _, _ = env.step(action)
    if done:
        env.render()
        break

# ----- Q-Table Heatmap -----
max_q = np.zeros((5,5))
for (x,y), q_vals in q_table.items():
    max_q[x,y] = np.max(q_vals)

plt.figure(figsize=(7,6))
sns.heatmap(max_q, annot=True, cmap="viridis", fmt=".1f")
plt.title("Max Q-Value Heatmap")
plt.xlabel("Y")
plt.ylabel("X")
plt.show()
