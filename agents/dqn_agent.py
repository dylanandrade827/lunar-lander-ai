import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from networks.q_network import QNetwork
from replay_buffer.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim, config, device="cpu"):
        self.device = device
        self.action_dim = action_dim
        self.gamma = config["gamma"]

        self.q_net = QNetwork(state_dim, action_dim, config["hidden_sizes"]).to(device)
        self.target_q_net = QNetwork(state_dim, action_dim, config["hidden_sizes"]).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config["learning_rate"])
        self.replay_buffer = ReplayBuffer(config["replay_buffer_size"], state_dim)

        self.batch_size = config["batch_size"]
        self.min_replay_size = config["min_replay_size"]

        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay_episodes = config["epsilon_decay_episodes"]

        self.total_steps = 0
        self.target_update_freq = config["target_update_freq"]
        self.train_freq = config["train_freq"]

    def select_action(self, state):
        # ε-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax(dim=1).item()

    def push_transition(self, *transition):
        self.replay_buffer.push(*transition)

    def maybe_update_epsilon(self, episode):
        # Linear decay
        slope = (self.epsilon_end - self.epsilon) / max(1, self.epsilon_decay_episodes)
        self.epsilon = max(self.epsilon_end, self.epsilon + slope)

    def train_step(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        if self.total_steps % self.train_freq != 0:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values
        q_values = self.q_net(states).gather(1, actions)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
