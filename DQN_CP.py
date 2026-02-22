import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

# DQN parameters
LR = 0.001
DF = 0.99
EP = 1.0
EP_DECAY = 0.995
MIN_EP = 0.01
EPOCHS = 1500
BATCH = 64
M_SIZE = 10000
UPDATE = 10

# Define the neural network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

def train_dqn(env_name="CartPole-v1"):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    policy_net = DQN(state_dim, action_space)
    target_net = DQN(state_dim, action_space)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=M_SIZE)
    epsilon = EP
    rewards = []
    epsilon_values = []
    best_reward = float('-inf')
    
    def select_action(state):
        if np.random.rand() < epsilon:
            return np.random.choice(action_space)
        else:
            with torch.no_grad():
                return torch.argmax(policy_net(torch.FloatTensor(state))).item()
    
    def optimize_model():
        if len(memory) < BATCH:
            return
        
        batch = random.sample(memory, BATCH)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states, dtype=np.float32))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        q_values = policy_net(states).gather(1, actions).squeeze()
        
        with torch.no_grad():
            next_q_values = target_net(next_states).max(1)[0]
            target_q_values = rewards + (DF * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    for episode in range(EPOCHS):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward
            optimize_model()
            if truncated:
                done = True
        
        rewards.append(total_reward)
        epsilon = max(MIN_EP, epsilon * EP_DECAY)
        epsilon_values.append(epsilon)
        
        if (episode + 1) % UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Avg Reward: {np.mean(rewards[-100:])}, Epsilon: {epsilon:.3f}")
    
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(), "dqn_cp.pth")
            print(f"New Best Model Saved! Reward: {best_reward}")

    env.close()

    plt.figure(figsize=(12, 5))
    # Plot 1: Average Reward vs. Episodes
    plt.subplot(1, 2, 1)
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), label="Episode Reward", color="blue")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward (100 episodes)")
    plt.title("Reward Progress Over Episodes")
    plt.legend()
    # Plot 2: Epsilon Decay vs. Episodes
    plt.subplot(1, 2, 2)
    plt.plot(epsilon_values, label="Epsilon Decay", color="red")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon (Exploration Rate)")
    plt.title("Epsilon Decay Over Time")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('dqn_cp.png') 
    plt.close()

if __name__ == "__main__":
    train_dqn()