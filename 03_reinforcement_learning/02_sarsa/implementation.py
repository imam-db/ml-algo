"""
SARSA (State-Action-Reward-State-Action) Implementation from Scratch
On-policy TD control algorithm for reinforcement learning

SARSA vs Q-Learning:
- SARSA: On-policy (learns from actions actually taken)
- Q-Learning: Off-policy (learns from optimal actions regardless of what was taken)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import sys
import os

# Import GridWorldSPG from Q-learning module using importlib to avoid conflicts
_current_dir = os.path.dirname(os.path.abspath(__file__))
_qlearning_impl_path = os.path.join(_current_dir, '../01_q_learning/implementation.py')
import importlib.util
_spec = importlib.util.spec_from_file_location("qlearning_gridworld", _qlearning_impl_path)
_qlearning_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qlearning_module)
GridWorldSPG = _qlearning_module.GridWorldSPG
del _spec, _qlearning_module, _qlearning_impl_path


class SARSAAgent:
    """
    SARSA agent yang belajar mengunjungi outlet SPG secara efisien.
    
    SARSA adalah algoritma on-policy: agent belajar dari aksi yang benar-benar
    diambilnya (termasuk eksplorasi), bukan aksi optimal teoritis.
    
    Update rule:
    Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
    
    Di mana a' adalah aksi yang BENAR-BENAR diambil di s' (bisa suboptimal karena eksplorasi).
    
    Parameters:
    -----------
    env : GridWorldSPG
        Environment untuk training
    
    learning_rate : float
        Learning rate (α) untuk update Q-values
    
    discount_factor : float
        Discount factor (γ) untuk future rewards
    
    epsilon : float
        Epsilon untuk epsilon-greedy policy
    
    epsilon_decay : float
        Decay rate untuk epsilon per episode
    
    epsilon_min : float
        Minimum epsilon value
    
    seed : int, optional
        Random seed untuk reproducibility
    """
    
    def __init__(self, 
                 env: GridWorldSPG,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 seed: Optional[int] = None):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize Q-table: Q[state][action]
        self.q_table = {}
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                state = (i, j)
                self.q_table[state] = np.zeros(len(env.actions))
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_steps': [],
            'epsilon_values': []
        }
    
    def select_action(self, state: Tuple[int, int], use_epsilon: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            use_epsilon: Whether to use epsilon-greedy (False for evaluation)
        
        Returns:
            action: Selected action
        """
        if use_epsilon and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(self.env.actions)
        else:
            # Exploit: best action
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: Tuple[int, int], action: int,
                      reward: float, next_state: Tuple[int, int], next_action: int):
        """
        Update Q-value using SARSA update rule.
        
        SARSA update:
        Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
        
        Key difference from Q-learning:
        - Uses Q(s', a') where a' is the ACTUAL next action taken
        - Q-learning would use max_a' Q(s', a')
        """
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]  # Actual next action, not max!
        
        # TD target
        target = reward + self.discount_factor * next_q
        
        # TD error
        td_error = target - current_q
        
        # Update
        self.q_table[state][action] = current_q + self.learning_rate * td_error
    
    def train(self, n_episodes: int = 1000, max_steps: int = 100, verbose: bool = True) -> Dict:
        """
        Train agent using SARSA algorithm.
        
        Args:
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
        
        Returns:
            training_history: Dictionary of training metrics
        """
        for episode in range(n_episodes):
            # Reset environment
            state = self.env.reset()
            
            # Select first action using epsilon-greedy
            action = self.select_action(state, use_epsilon=True)
            
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps):
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Select next action using epsilon-greedy (KEY: on-policy)
                next_action = self.select_action(next_state, use_epsilon=True)
                
                # SARSA update using the actual next_action
                self.update_q_value(state, action, reward, next_state, next_action)
                
                # Move to next state and action
                state = next_state
                action = next_action  # Use the action we selected (on-policy!)
                
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Record history
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_steps'].append(episode_steps)
            self.training_history['epsilon_values'].append(self.epsilon)
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                avg_steps = np.mean(self.training_history['episode_steps'][-100:])
                print(f"Episode {episode + 1}/{n_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Steps: {avg_steps:.1f} | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return self.training_history
    
    def evaluate(self, n_episodes: int = 100) -> Tuple[float, float]:
        """
        Evaluate agent performance without exploration.
        
        Args:
            n_episodes: Number of evaluation episodes
        
        Returns:
            avg_reward: Average reward per episode
            avg_steps: Average steps per episode
        """
        total_reward = 0
        total_steps = 0
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for _ in range(100):
                action = self.select_action(state, use_epsilon=False)
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            total_reward += episode_reward
            total_steps += episode_steps
        
        return total_reward / n_episodes, total_steps / n_episodes
    
    def get_policy(self) -> Dict[Tuple[int, int], int]:
        """Extract learned policy from Q-table."""
        policy = {}
        for state in self.q_table:
            policy[state] = np.argmax(self.q_table[state])
        return policy


def compare_sarsa_vs_qlearning(env: GridWorldSPG, 
                               n_episodes: int = 1000,
                               seed: int = 42) -> Tuple[SARSAAgent, object]:
    """
    Bandingkan SARSA dengan Q-Learning pada environment yang sama.
    
    Args:
        env: Grid-world environment
        n_episodes: Number of training episodes
        seed: Random seed
    
    Returns:
        sarsa_agent: Trained SARSA agent
        qlearning_agent: Trained Q-Learning agent
    """
    # Import Q-learning agent
    sys.path.insert(0, _qlearning_dir)
    try:
        from implementation import QLearningAgent
    finally:
        sys.path.pop(0)
    
    print("Training SARSA Agent...")
    print("=" * 50)
    sarsa_agent = SARSAAgent(env, seed=seed)
    sarsa_agent.train(n_episodes=n_episodes, verbose=True)
    
    print("\n" + "=" * 50)
    print("Training Q-Learning Agent...")
    print("=" * 50)
    qlearning_agent = QLearningAgent(env, seed=seed)
    qlearning_agent.train(n_episodes=n_episodes, verbose=True)
    
    # Evaluate both
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    
    sarsa_reward, sarsa_steps = sarsa_agent.evaluate(n_episodes=100)
    print(f"SARSA - Avg Reward: {sarsa_reward:.2f}, Avg Steps: {sarsa_steps:.1f}")
    
    qlearning_reward, qlearning_steps = qlearning_agent.evaluate(n_episodes=100)
    print(f"Q-Learning - Avg Reward: {qlearning_reward:.2f}, Avg Steps: {qlearning_steps:.1f}")
    
    return sarsa_agent, qlearning_agent


def plot_training_comparison(sarsa_agent: SARSAAgent, qlearning_agent: object,
                            title: str = "SARSA vs Q-Learning Training"):
    """
    Plot training curves untuk SARSA dan Q-Learning.
    
    Args:
        sarsa_agent: Trained SARSA agent
        qlearning_agent: Trained Q-Learning agent
        title: Plot title
    
    Returns:
        fig: matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Rewards
    ax = axes[0]
    window = 50
    
    sarsa_rewards = sarsa_agent.training_history['episode_rewards']
    qlearning_rewards = qlearning_agent.training_history['episode_rewards']
    
    sarsa_smoothed = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
    qlearning_smoothed = np.convolve(qlearning_rewards, np.ones(window)/window, mode='valid')
    
    ax.plot(sarsa_smoothed, label='SARSA (on-policy)', linewidth=2, alpha=0.8)
    ax.plot(qlearning_smoothed, label='Q-Learning (off-policy)', linewidth=2, alpha=0.8)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Learning Curve (Rewards)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Steps
    ax = axes[1]
    sarsa_steps = sarsa_agent.training_history['episode_steps']
    qlearning_steps = qlearning_agent.training_history['episode_steps']
    
    sarsa_smoothed_steps = np.convolve(sarsa_steps, np.ones(window)/window, mode='valid')
    qlearning_smoothed_steps = np.convolve(qlearning_steps, np.ones(window)/window, mode='valid')
    
    ax.plot(sarsa_smoothed_steps, label='SARSA', linewidth=2, alpha=0.8)
    ax.plot(qlearning_smoothed_steps, label='Q-Learning', linewidth=2, alpha=0.8)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Steps', fontsize=12)
    ax.set_title('Learning Curve (Steps)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_policy(agent: SARSAAgent, title: str = "Learned Policy"):
    """
    Visualize learned policy pada grid.
    
    Args:
        agent: Trained agent
        title: Plot title
    
    Returns:
        fig: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    grid_size = agent.env.grid_size
    policy = agent.get_policy()
    
    # Draw grid
    for i in range(grid_size):
        for j in range(grid_size):
            state = (i, j)
            
            # Color based on cell type
            if state == agent.env.start_pos:
                color = 'lightgreen'
                ax.add_patch(plt.Rectangle((j, grid_size-1-i), 1, 1, 
                                          facecolor=color, edgecolor='black', linewidth=2))
                ax.text(j+0.5, grid_size-1-i+0.5, 'START', ha='center', va='center',
                       fontsize=9, fontweight='bold')
            elif state in agent.env.outlet_positions:
                color = 'gold'
                ax.add_patch(plt.Rectangle((j, grid_size-1-i), 1, 1,
                                          facecolor=color, edgecolor='black', linewidth=2))
                ax.text(j+0.5, grid_size-1-i+0.5, 'OUTLET', ha='center', va='center',
                       fontsize=9, fontweight='bold')
            elif state in agent.env.obstacle_positions:
                color = 'lightcoral'
                ax.add_patch(plt.Rectangle((j, grid_size-1-i), 1, 1,
                                          facecolor=color, edgecolor='black', linewidth=2))
                ax.text(j+0.5, grid_size-1-i+0.5, 'X', ha='center', va='center',
                       fontsize=12, fontweight='bold')
            else:
                color = 'white'
                ax.add_patch(plt.Rectangle((j, grid_size-1-i), 1, 1,
                                          facecolor=color, edgecolor='gray', linewidth=1))
            
            # Draw arrow for policy
            if state not in agent.env.obstacle_positions:
                action = policy[state]
                arrow = agent.env.action_names[action]
                ax.text(j+0.5, grid_size-1-i+0.2, arrow, ha='center', va='center',
                       fontsize=20, color='blue')
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.set_xticks(range(grid_size+1))
    ax.set_yticks(range(grid_size+1))
    ax.grid(True)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("SARSA Implementation Demo")
    print("=" * 60)
    
    # Create environment
    env = GridWorldSPG(grid_size=5, n_outlets=3, n_obstacles=3, seed=42)
    print(f"Environment: {env.grid_size}x{env.grid_size} grid")
    print(f"Outlets: {env.n_outlets}, Obstacles: {env.n_obstacles}\n")
    
    # Train SARSA agent
    print("Training SARSA Agent...")
    sarsa_agent = SARSAAgent(env, seed=42)
    sarsa_agent.train(n_episodes=1000, verbose=True)
    
    # Evaluate
    print("\n" + "=" * 60)
    avg_reward, avg_steps = sarsa_agent.evaluate(n_episodes=100)
    print(f"Evaluation Results:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Steps: {avg_steps:.1f}")
    
    # Visualize policy
    try:
        fig = visualize_policy(sarsa_agent, "SARSA Learned Policy")
        fig.savefig('03_reinforcement_learning/02_sarsa/sarsa_policy.png',
                   dpi=150, bbox_inches='tight')
        print("\n✓ Policy visualization saved")
    except Exception as e:
        print(f"\n! Could not save policy visualization: {e}")
