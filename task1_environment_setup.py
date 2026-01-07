"""
Task 1: Environment Setup and Exploration
Reinforcement Learning Assignment - Alternative Version

This script sets up the Gymnasium environment and performs
detailed exploration of the FrozenLake environment.
"""

import gymnasium as gym
import numpy as np

def print_separator(char='=', length=70):
    print(char * length)

def explore_environment():
    """Explore and document the FrozenLake environment"""

    print_separator()
    print("TASK 1: GYMNASIUM ENVIRONMENT EXPLORATION")
    print_separator()

    try:
        # Initialize environment
        print("\n[1] Initializing FrozenLake-v1 environment...")
        env = gym.make("FrozenLake-v1", render_mode=None)

        # Get environment details
        print("[2] Gathering environment specifications...")
        obs_space = env.observation_space
        action_space = env.action_space

        print_separator('-')
        print("ENVIRONMENT SPECIFICATIONS")
        print_separator('-')
        print(f"Observation Space Type: {type(obs_space).__name__}")
        print(f"Total States: {obs_space.n}")
        print(f"Action Space Type: {type(action_space).__name__}")
        print(f"Total Actions: {action_space.n}")
        print(f"Action Meanings: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP")
        print_separator('-')

        # Reset and get initial state
        initial_state, info = env.reset(seed=42)
        print(f"\nInitial State (with seed=42): {initial_state}")
        print(f"Additional Info: {info}")

        # Analyze state transitions
        print("\n[3] Analyzing state transitions...")
        print_separator('-')
        print("ACTION EXPLORATION (10 random steps)")
        print_separator('-')

        state = initial_state
        for step in range(10):
            action = action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            action_name = ['LEFT', 'DOWN', 'RIGHT', 'UP'][action]
            print(f"Step {step+1}: State {state} --[{action_name}]--> "
                  f"State {next_state} | Reward: {reward} | Done: {done}")

            if done:
                print(f"  > Episode terminated. Resetting...")
                state, info = env.reset()
            else:
                state = next_state

        # Test deterministic behavior
        print("\n[4] Testing reproducibility with seeds...")
        print_separator('-')

        # Run same sequence twice with same seed
        results = []
        for trial in range(2):
            env_test = gym.make("FrozenLake-v1")
            state, _ = env_test.reset(seed=123)
            trajectory = [state]

            for _ in range(5):
                action = 1  # Always go DOWN
                state, _, done, _, _ = env_test.step(action)
                trajectory.append(state)
                if done:
                    break

            results.append(trajectory)
            env_test.close()
            print(f"Trial {trial+1} trajectory: {trajectory}")

        if results[0] == results[1]:
            print("Result: Trajectories are IDENTICAL (seed works!)")
        else:
            print("Result: Trajectories DIFFER (stochastic environment)")

        env.close()

        print("\n" + "="*70)
        print("SUCCESS! Environment setup verified and explored.")
        print("="*70)
        print("\nNext Steps:")
        print("  - Proceed to Task 2 for Q-Learning implementation")
        print("  - Or proceed to Task 3 for Deep Q-Network")

        return True

    except ImportError as e:
        print("\n" + "="*70)
        print("ERROR: Required packages not installed")
        print("="*70)
        print(f"\nError details: {str(e)}")
        print("\nInstallation instructions:")
        print("  pip install gymnasium numpy")
        return False

    except Exception as e:
        print("\n" + "="*70)
        print("ERROR: Unexpected error occurred")
        print("="*70)
        print(f"\nError details: {str(e)}")
        print("\nPlease check your installation or contact support.")
        return False

if __name__ == "__main__":
    success = explore_environment()
    exit(0 if success else 1)
