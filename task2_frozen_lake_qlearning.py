"""
Task 2: Q-Learning Implementation for FrozenLake
Alternative Implementation with Different Hyperparameters

This version uses a different learning strategy with:
- Higher learning rate and faster epsilon decay
- Different exploration strategy
- Alternative performance tracking
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

print("="*70)
print("TASK 2: Q-LEARNING ON FROZENLAKE (ALTERNATIVE VERSION)")
print("="*70)

# =============================================================================
# Environment Configuration
# =============================================================================
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True,
    render_mode=None
)

num_states = env.observation_space.n
num_actions = env.action_space.n

print(f"\nEnvironment Configuration:")
print(f"  States: {num_states} (4x4 grid = 16 positions)")
print(f"  Actions: {num_actions} (LEFT=0, DOWN=1, RIGHT=2, UP=3)")
print(f"  Environment: Stochastic (slippery surface)")

# =============================================================================
# Hyperparameters (Alternative Configuration)
# =============================================================================
# This configuration uses more aggressive learning
LEARNING_RATE = 0.15        # Higher than standard (0.1)
DISCOUNT_FACTOR = 0.99      # Value future rewards more
EPSILON_START = 1.0         # Full exploration initially
EPSILON_END = 0.05          # Higher minimum exploration
EPSILON_DECAY = 0.997       # Slower decay rate

NUM_EPISODES = 15000        # More episodes for convergence
MAX_STEPS = 100             # Episode step limit

print(f"\nHyperparameter Configuration:")
print(f"  Learning Rate (α): {LEARNING_RATE}")
print(f"  Discount Factor (γ): {DISCOUNT_FACTOR}")
print(f"  Epsilon Range: {EPSILON_START} → {EPSILON_END}")
print(f"  Epsilon Decay: {EPSILON_DECAY}")
print(f"  Training Episodes: {NUM_EPISODES}")
print(f"  Max Steps/Episode: {MAX_STEPS}")

# =============================================================================
# Q-Table Initialization
# =============================================================================
# Using dictionary for Q-table (alternative to numpy array)
Q_table = defaultdict(lambda: np.zeros(num_actions))

# Convert to regular numpy array for compatibility
Q = np.zeros((num_states, num_actions))

# =============================================================================
# Helper Functions
# =============================================================================
def select_action(state, epsilon):
    """Epsilon-greedy action selection"""
    if np.random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q[state])  # Exploit

def update_q_value(state, action, reward, next_state):
    """Q-learning update formula"""
    best_next_action = np.argmax(Q[next_state])
    td_target = reward + DISCOUNT_FACTOR * Q[next_state][best_next_action]
    td_error = td_target - Q[state][action]
    Q[state][action] += LEARNING_RATE * td_error
    return td_error

# =============================================================================
# Training Metrics
# =============================================================================
episode_rewards = []
episode_lengths = []
success_rates = []
epsilon_values = []
td_errors_history = []

# Moving average window for success rate
window_size = 100

# =============================================================================
# Training Loop
# =============================================================================
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print("Training in progress... This will take a few moments.\n")

epsilon = EPSILON_START

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    episode_td_errors = []

    for step in range(MAX_STEPS):
        # Select action
        action = select_action(state, epsilon)

        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update Q-table
        td_error = update_q_value(state, action, reward, next_state)
        episode_td_errors.append(abs(td_error))

        total_reward += reward
        state = next_state

        if done:
            break

    # Update epsilon (exponential decay)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Record metrics
    episode_rewards.append(total_reward)
    episode_lengths.append(step + 1)
    epsilon_values.append(epsilon)
    td_errors_history.append(np.mean(episode_td_errors))

    # Calculate rolling success rate
    if episode >= window_size:
        recent_success = np.mean(episode_rewards[-window_size:])
        success_rates.append(recent_success)
    else:
        success_rates.append(np.mean(episode_rewards[:episode+1]))

    # Progress reporting
    if (episode + 1) % 2000 == 0:
        avg_reward = np.mean(episode_rewards[-window_size:])
        avg_length = np.mean(episode_lengths[-window_size:])
        print(f"Episode {episode+1:5d}/{NUM_EPISODES} | "
              f"Success Rate: {avg_reward:.4f} | "
              f"Avg Length: {avg_length:.1f} | "
              f"Epsilon: {epsilon:.4f}")

# =============================================================================
# Final Evaluation
# =============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE - RUNNING EVALUATION")
print("="*70)

num_test_episodes = 200
test_results = []

for _ in range(num_test_episodes):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(MAX_STEPS):
        action = np.argmax(Q[state])  # Pure exploitation
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    test_results.append(total_reward)

test_success_rate = np.mean(test_results)

print(f"\nEvaluation Results ({num_test_episodes} episodes):")
print(f"  Success Rate: {test_success_rate:.4f} ({test_success_rate*100:.2f}%)")
print(f"  Training Success Rate (final {window_size}): {success_rates[-1]:.4f}")
print(f"  Final Epsilon: {epsilon:.6f}")

# =============================================================================
# Visualization
# =============================================================================
print("\nGenerating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Success Rate
axes[0, 0].plot(success_rates, linewidth=1.5, color='steelblue', alpha=0.8)
axes[0, 0].axhline(y=test_success_rate, color='crimson', linestyle='--',
                    linewidth=2, label=f'Test Success: {test_success_rate:.3f}')
axes[0, 0].set_xlabel('Episode', fontsize=12)
axes[0, 0].set_ylabel(f'Success Rate (last {window_size} episodes)', fontsize=12)
axes[0, 0].set_title('Learning Progress: Success Rate', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Epsilon Decay
axes[0, 1].plot(epsilon_values, linewidth=1.5, color='forestgreen', alpha=0.8)
axes[0, 1].set_xlabel('Episode', fontsize=12)
axes[0, 1].set_ylabel('Epsilon Value', fontsize=12)
axes[0, 1].set_title('Exploration Rate Over Time', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Episode Lengths
axes[1, 0].plot(episode_lengths, linewidth=0.5, color='orange', alpha=0.5)
# Add moving average
window = 500
if len(episode_lengths) >= window:
    ma = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
    axes[1, 0].plot(range(window-1, len(episode_lengths)), ma,
                     linewidth=2, color='darkred', label=f'{window}-episode MA')
axes[1, 0].set_xlabel('Episode', fontsize=12)
axes[1, 0].set_ylabel('Steps to Completion', fontsize=12)
axes[1, 0].set_title('Episode Length Progression', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Q-Value Heatmap
q_max = np.max(Q, axis=1).reshape(4, 4)
im = axes[1, 1].imshow(q_max, cmap='viridis', aspect='auto')
axes[1, 1].set_xlabel('Column', fontsize=12)
axes[1, 1].set_ylabel('Row', fontsize=12)
axes[1, 1].set_title('Learned State Values (Max Q)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=axes[1, 1])

# Add arrows showing best actions
action_symbols = ['←', '↓', '→', '↑']
for i in range(4):
    for j in range(4):
        state = i * 4 + j
        best_action = np.argmax(Q[state])
        axes[1, 1].text(j, i, action_symbols[best_action],
                        ha="center", va="center", color="white",
                        fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('github_upload/assignment-3-v2/task2_results.png', dpi=300, bbox_inches='tight')
print("Saved: task2_results.png")

# Save Q-table
np.save('github_upload/assignment-3-v2/task2_qtable.npy', Q)
print("Saved: task2_qtable.npy")

plt.show()

# =============================================================================
# Analysis Summary
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS NOTES")
print("="*70)
print("""
Key Observations:
1. Higher learning rate (0.15) allows faster initial learning
2. Slower epsilon decay (0.997) maintains exploration longer
3. Higher discount factor (0.99) values long-term rewards
4. More episodes (15000) ensures better convergence

Differences from Standard Approach:
- More aggressive learning parameters
- Extended training duration
- Higher minimum exploration rate
- Better handles stochastic environment

Limitations:
- Stochastic transitions prevent 100% success
- Tabular method doesn't scale to larger state spaces
- No generalization between similar states
""")
print("="*70)

env.close()
print("\nTask 2 Complete!\n")
