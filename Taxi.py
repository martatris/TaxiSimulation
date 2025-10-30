import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# ============================================================
# 1️⃣ CUSTOM TAXI ENVIRONMENT
# ============================================================

class MultiPassengerTaxiEnv(gym.Env):
    def __init__(self, grid_size=10, num_passengers=2):
        super(MultiPassengerTaxiEnv, self).__init__()

        self.grid_size = grid_size
        self.num_passengers = num_passengers
        self.action_space = spaces.Discrete(6)  # 0=S,1=N,2=E,3=W,4=Pickup,5=Dropoff
        obs_len = 2 + num_passengers * 3
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(obs_len,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.taxi_pos = np.random.randint(0, self.grid_size, size=2)
        self.passengers = []
        for _ in range(self.num_passengers):
            start = np.random.randint(0, self.grid_size, size=2)
            dest = np.random.randint(0, self.grid_size, size=2)
            while np.array_equal(dest, start):
                dest = np.random.randint(0, self.grid_size, size=2)
            self.passengers.append({"start": start, "dest": dest, "in_taxi": False, "done": False})
        return self._get_obs(), {}

    def _get_obs(self):
        obs = list(self.taxi_pos)
        for p in self.passengers:
            obs += list(p["start"])
            obs.append(float(p["in_taxi"] or p["done"]))
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        reward = -0.5  # small time penalty
        done = False
        found_pickup = False
        successful_dropoff = False

        # === Movement ===
        if action == 0 and self.taxi_pos[0] < self.grid_size - 1:  # South
            self.taxi_pos[0] += 1
        elif action == 1 and self.taxi_pos[0] > 0:  # North
            self.taxi_pos[0] -= 1
        elif action == 2 and self.taxi_pos[1] < self.grid_size - 1:  # East
            self.taxi_pos[1] += 1
        elif action == 3 and self.taxi_pos[1] > 0:  # West
            self.taxi_pos[1] -= 1

        # === Pickup ===
        elif action == 4:
            for p in self.passengers:
                if not p["in_taxi"] and not p["done"] and np.array_equal(self.taxi_pos, p["start"]):
                    p["in_taxi"] = True
                    found_pickup = True
                    break
            if not found_pickup:
                reward -= 10

        # === Dropoff ===
        elif action == 5:
            for p in self.passengers:
                if p["in_taxi"] and np.array_equal(self.taxi_pos, p["dest"]):
                    p["in_taxi"] = False
                    p["done"] = True
                    successful_dropoff = True
            if not successful_dropoff:
                reward -= 10

        # === Reward shaping ===
        if found_pickup:
            reward += 15
        if successful_dropoff:
            reward += 30
        if all(p["done"] for p in self.passengers):
            reward += 100
            done = True

        return self._get_obs(), reward, done, False, {}


# ============================================================
# 2️⃣ DQN NETWORK + REPLAY MEMORY
# ============================================================

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 3️⃣ TRAINING FUNCTION (NOW WITH CONFIGURABLE EPISODE LENGTH)
# ============================================================

def train_dqn(env, episodes=3000, max_steps=1500, gamma=0.99, lr=5e-4, batch_size=128, epsilon_decay=0.997):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer()

    epsilon = 1.0
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        total_reward = 0

        for _ in range(max_steps):  # 👈 longer episode duration
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            memory.push(state.cpu().numpy(), action, reward, next_state, done)
            state = next_state_tensor

            if len(memory) > batch_size:
                states, actions, rewards_b, next_states, dones = memory.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32).to(device).squeeze(1)
                actions = torch.tensor(actions).to(device)
                rewards_b = torch.tensor(rewards_b, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = target_net(next_states).max(1)[0]
                target_q = rewards_b + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(0.05, epsilon * epsilon_decay)
        rewards.append(total_reward)

        if episode % 50 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            avg = np.mean(rewards[-50:])
            print(f"Episode {episode}, Avg Reward: {avg:.2f}, Epsilon: {epsilon:.2f}")

    return rewards, policy_net


# ============================================================
# 4️⃣ TRAINING & PLOTTING
# ============================================================

env = MultiPassengerTaxiEnv(grid_size=10, num_passengers=2)
rewards, model = train_dqn(env, episodes=3000, max_steps=1500)

window = 50
plt.figure(figsize=(9, 4))
plt.plot(np.convolve(rewards, np.ones(window) / window, mode='valid'))
plt.title("🚕 Training Progress - Average Reward (10x10 Taxi, Long Episodes)")
plt.xlabel("Episode")
plt.ylabel("Smoothed Reward")
plt.grid(True)
plt.show()


# ============================================================
# 5️⃣ VISUALIZATION (ANIMATION)
# ============================================================

def animate_taxi(env, model, steps=100, interval=400):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.grid(True)
    plt.gca().invert_yaxis()

    taxi_rect = patches.Rectangle((env.taxi_pos[1] - 0.4, env.taxi_pos[0] - 0.4), 0.8, 0.8, color="red")
    ax.add_patch(taxi_rect)
    passenger_circles = []
    dest_rects = []

    for p in env.passengers:
        c = patches.Circle((p["start"][1], p["start"][0]), 0.3, color="blue")
        d = patches.Rectangle((p["dest"][1] - 0.3, p["dest"][0] - 0.3), 0.6, 0.6, fill=False, edgecolor="green",
                              linewidth=2)
        passenger_circles.append(c)
        dest_rects.append(d)
        ax.add_patch(c)
        ax.add_patch(d)

    def update(_):
        nonlocal state_tensor
        with torch.no_grad():
            action = model(state_tensor).argmax().item()
        next_state, _, done, _, _ = env.step(action)
        state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        taxi_rect.set_xy((env.taxi_pos[1] - 0.4, env.taxi_pos[0] - 0.4))

        for i, p in enumerate(env.passengers):
            if p["done"]:
                passenger_circles[i].set_visible(False)
            elif p["in_taxi"]:
                passenger_circles[i].center = (env.taxi_pos[1], env.taxi_pos[0])

        if done:
            print("🎉 Task Completed!")
            anim.event_source.stop()
        return [taxi_rect] + passenger_circles

    anim = animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=True, repeat=False)
    plt.show()


# Run animation
animate_taxi(env, model, steps=80)