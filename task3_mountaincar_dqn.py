"""
Task 3: Deep Q-Network for MountainCar
Alternative Implementation with Enhanced Features

This version includes:
- Different network architecture (3 hidden layers)
- Prioritized experience replay concept
- Learning rate scheduling
- Enhanced reward shaping
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    print("ERROR: PyTorch not installed!")
    print("Install with: pip install torch")
    exit(1)

print("="*70)
print("TASK 3: DEEP Q-NETWORK FOR MOUNTAINCAR (ALTERNATIVE)")
print("="*70)

# Set seeds for reproducibility
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nCompute Device: {device}")

# =============================================================================
# Neural Network Architecture (Alternative Design)
# =============================================================================
class EnhancedDQN(nn.Module):
    """
    Enhanced DQN with 3 hidden layers and dropout
    Architecture: Input -> 256 -> 128 -> 64 -> Output
    """
    def __init__(self, state_dim, action_dim):
        super(EnhancedDQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# =============================================================================
# Enhanced Replay Buffer
# =============================================================================
class PrioritizedReplayBuffer:
    """
    Simple version of prioritized experience replay
    Stores transitions with sampling probabilities
    """
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, transition, priority=1.0):
        self.buffer.append(transition)
        self.priorities.append(priority)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        # Sample based on priorities (simplified)
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones)), indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

# =============================================================================
# DQN Agent (Enhanced Version)
# =============================================================================
class EnhancedDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Networks
        self.policy_net = EnhancedDQN(state_dim, action_dim).to(device)
        self.target_net = EnhancedDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer with learning rate scheduling
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996
        self.batch_size = 128
        self.target_update_freq = 5

        # Replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=20000)

        # Tracking
        self.steps_done = 0

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        priority = 1.0  # Default priority
        self.memory.push((state, action, reward, next_state, done), priority)

    def train(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        result = self.memory.sample(self.batch_size)
        if result is None:
            return None

        batch, indices = result
        states, actions, rewards, next_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Compute Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calculate loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Update priorities based on TD error
        td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 0.01)

        self.steps_done += 1
        return loss.item()

    def update_target_net(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# =============================================================================
# Environment Setup
# =============================================================================
env = gym.make('MountainCar-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"\nEnvironment Specifications:")
print(f"  State Dimension: {state_dim} (position, velocity)")
print(f"  Action Dimension: {action_dim}")
print(f"  Actions: 0=Push Left, 1=No Action, 2=Push Right")

# =============================================================================
# Training Configuration
# =============================================================================
NUM_EPISODES = 600
MAX_STEPS = 200

print(f"\nTraining Configuration:")
print(f"  Episodes: {NUM_EPISODES}")
print(f"  Max Steps per Episode: {MAX_STEPS}")
print(f"  Batch Size: 128")
print(f"  Replay Buffer: 20,000")
print(f"  Network: 256-128-64 with dropout")

# =============================================================================
# Initialize Agent
# =============================================================================
agent = EnhancedDQNAgent(state_dim, action_dim)

# =============================================================================
# Training Loop
# =============================================================================
print("\n" + "="*70)
print("TRAINING IN PROGRESS")
print("="*70)
print("This may take several minutes...\n")

episode_rewards = []
episode_lengths = []
losses = []
epsilon_history = []
success_count = 0

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    episode_losses = []

    for step in range(MAX_STEPS):
        # Select action
        action = agent.select_action(state)

        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Enhanced reward shaping
        position, velocity = next_state
        shaped_reward = reward

        # Reward for moving toward goal
        if position > state[0]:
            shaped_reward += 0.1

        # Reward for positive velocity toward goal
        if velocity > 0:
            shaped_reward += 0.05

        # Large bonus for reaching goal
        if position >= 0.5:
            shaped_reward += 100
            success_count += 1

        # Store transition
        agent.store_transition(state, action, shaped_reward, next_state, done)

        # Train
        loss = agent.train()
        if loss is not None:
            episode_losses.append(loss)

        state = next_state
        total_reward += reward

        if done:
            break

    # Update target network periodically
    if episode % agent.target_update_freq == 0:
        agent.update_target_net()

    # Decay epsilon and learning rate
    agent.decay_epsilon()
    if episode % 50 == 0:
        agent.scheduler.step()

    # Record metrics
    episode_rewards.append(total_reward)
    episode_lengths.append(step + 1)
    epsilon_history.append(agent.epsilon)
    avg_loss = np.mean(episode_losses) if episode_losses else 0
    losses.append(avg_loss)

    # Progress report
    if (episode + 1) % 100 == 0:
        recent_avg = np.mean(episode_rewards[-100:])
        recent_length = np.mean(episode_lengths[-100:])
        print(f"Episode {episode+1:3d}/{NUM_EPISODES} | "
              f"Avg Reward: {recent_avg:7.2f} | "
              f"Avg Length: {recent_length:6.1f} | "
              f"Epsilon: {agent.epsilon:.4f} | "
              f"Loss: {avg_loss:.4f}")

# =============================================================================
# Evaluation
# =============================================================================
print("\n" + "="*70)
print("EVALUATION PHASE")
print("="*70)

test_rewards = []
test_lengths = []

for _ in range(50):
    state, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    test_rewards.append(total_reward)
    test_lengths.append(step + 1)

print(f"\nTest Results (50 episodes):")
print(f"  Average Reward: {np.mean(test_rewards):.2f}")
print(f"  Average Length: {np.mean(test_lengths):.1f}")
print(f"  Success Rate: {np.sum(np.array(test_rewards) > -200) / 50:.2%}")
print(f"  Best Reward: {np.max(test_rewards):.0f}")

# =============================================================================
# Visualization
# =============================================================================
print("\nGenerating visualizations...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. Training rewards
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(episode_rewards, alpha=0.4, linewidth=0.8, color='steelblue', label='Episode Reward')
window = 50
ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
ax1.plot(range(window-1, len(episode_rewards)), ma,
         linewidth=2.5, color='darkblue', label=f'{window}-Episode Average')
ax1.axhline(y=-110, color='green', linestyle='--', linewidth=2, label='Target (-110)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward')
ax1.set_title('Training Progress: Episode Rewards', fontweight='bold', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Episode lengths
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(episode_lengths, alpha=0.5, linewidth=0.8, color='coral')
ma_len = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
ax2.plot(range(window-1, len(episode_lengths)), ma_len,
         linewidth=2, color='darkred', label=f'{window}-Ep Avg')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Steps')
ax2.set_title('Episode Length', fontweight='bold', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Training loss
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(losses, alpha=0.5, linewidth=0.8, color='orange')
if len(losses) >= window:
    ma_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
    ax3.plot(range(window-1, len(losses)), ma_loss,
             linewidth=2, color='darkred', label=f'{window}-Ep Avg')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Loss')
ax3.set_title('Training Loss', fontweight='bold', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Epsilon decay
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(epsilon_history, linewidth=2, color='forestgreen')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Epsilon')
ax4.set_title('Exploration Rate', fontweight='bold', fontsize=13)
ax4.grid(True, alpha=0.3)

# 5. Test results distribution
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(test_rewards, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax5.axvline(np.mean(test_rewards), color='red', linestyle='--',
            linewidth=2, label=f'Mean: {np.mean(test_rewards):.1f}')
ax5.set_xlabel('Reward')
ax5.set_ylabel('Frequency')
ax5.set_title('Test Performance Distribution', fontweight='bold', fontsize=13)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

plt.savefig('github_upload/assignment-3-v2/task3_results.png', dpi=300, bbox_inches='tight')
print("Saved: task3_results.png")

# Save model
torch.save(agent.policy_net.state_dict(), 'github_upload/assignment-3-v2/task3_model.pth')
print("Saved: task3_model.pth")

plt.show()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("IMPLEMENTATION SUMMARY")
print("="*70)
print("""
Key Features of This Implementation:
1. Deeper network (3 hidden layers: 256-128-64)
2. Prioritized experience replay (simplified version)
3. Learning rate scheduling (decay every 50 episodes)
4. Enhanced reward shaping (position + velocity bonuses)
5. Dropout for regularization
6. Larger replay buffer (20,000 vs 10,000)
7. Larger batch size (128 vs 64)

Advantages:
- Better function approximation with deeper network
- More stable learning with prioritized replay
- Adaptive learning rate prevents overshooting
- Enhanced reward shaping speeds up learning

Trade-offs:
- More parameters means longer training time
- Higher memory usage with larger buffer
- More complex architecture may overfit on small tasks
""")
print("="*70)

env.close()
print("\nTask 3 Complete!\n")
