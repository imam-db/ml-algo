"""
Q-Learning Implementation from Scratch
Grid-world environment themed as SPG (Sales Promoter Girl) visiting outlets
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import random


class GridWorldSPG:
    """
    Grid-world environment where an SPG agent learns to visit outlets efficiently.
    
    Grid layout:
    - S: Start position (SPG home base)
    - O: Outlet (reward)
    - X: Obstacle (penalty)
    - Empty: Regular path (small step penalty)
    """
    
    def __init__(self, grid_size: int = 5, n_outlets: int = 3, n_obstacles: int = 3, seed: Optional[int] = None):
        """
        Initialize the grid-world environment.
        
        Args:
            grid_size: Size of the square grid
            n_outlets: Number of outlets to place
            n_obstacles: Number of obstacles to place
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.grid_size = grid_size
        self.n_outlets = n_outlets
        self.n_obstacles = n_obstacles
        self.seed = seed
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [0, 1, 2, 3]
        self.action_names = ['↑', '→', '↓', '←']
        
        # Initialize grid
        self._initialize_grid()
        
    def _initialize_grid(self):
        """Initialize grid with start, outlets, and obstacles."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Place start position at top-left
        self.start_pos = (0, 0)
        self.grid[self.start_pos] = 1  # 1 = start
        
        # Get available positions (excluding start)
        available_positions = [(i, j) for i in range(self.grid_size) 
                              for j in range(self.grid_size) if (i, j) != self.start_pos]
        
        # Place outlets (2 = outlet)
        self.outlet_positions = []
        for _ in range(self.n_outlets):
            if available_positions:
                pos = random.choice(available_positions)
                available_positions.remove(pos)
                self.outlet_positions.append(pos)
                self.grid[pos] = 2
        
        # Place obstacles (3 = obstacle)
        self.obstacle_positions = []
        for _ in range(self.n_obstacles):
            if available_positions:
                pos = random.choice(available_positions)
                available_positions.remove(pos)
                self.obstacle_positions.append(pos)
                self.grid[pos] = 3
        
        # Track visited outlets
        self.visited_outlets = set()
        
    def reset(self) -> Tuple[int, int]:
        """Reset environment to start position."""
        self.current_pos = self.start_pos
        self.visited_outlets = set()
        return self.current_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0=up, 1=right, 2=down, 3=left)
            
        Returns:
            next_state: Next position
            reward: Reward received
            done: Whether episode is finished
            info: Additional information
        """
        # Calculate next position
        row, col = self.current_pos
        
        if action == 0:  # up
            next_row, next_col = max(0, row - 1), col
        elif action == 1:  # right
            next_row, next_col = row, min(self.grid_size - 1, col + 1)
        elif action == 2:  # down
            next_row, next_col = min(self.grid_size - 1, row + 1), col
        elif action == 3:  # left
            next_row, next_col = row, max(0, col - 1)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        next_pos = (next_row, next_col)
        
        # Calculate reward
        reward = -0.1  # Small penalty for each step
        done = False
        info = {}
        
        # Check if hit obstacle
        if next_pos in self.obstacle_positions:
            reward = -10.0  # Large penalty for hitting obstacle
            info['hit_obstacle'] = True
        # Check if reached outlet
        elif next_pos in self.outlet_positions:
            if next_pos not in self.visited_outlets:
                reward = 10.0  # Reward for visiting new outlet
                self.visited_outlets.add(next_pos)
                info['visited_outlet'] = True
                
                # Episode ends when all outlets visited
                if len(self.visited_outlets) == len(self.outlet_positions):
                    done = True
                    reward = 50.0  # Bonus for visiting all outlets
                    info['all_outlets_visited'] = True
            else:
                reward = 0.0  # No reward for revisiting outlet
        
        self.current_pos = next_pos
        
        return next_pos, reward, done, info
    
    def get_state_index(self, pos: Tuple[int, int]) -> int:
        """Convert (row, col) position to state index."""
        row, col = pos
        return row * self.grid_size + col
    
    def get_pos_from_state(self, state_index: int) -> Tuple[int, int]:
        """Convert state index to (row, col) position."""
        row = state_index // self.grid_size
        col = state_index % self.grid_size
        return (row, col)
    
    def render(self, q_table: Optional[np.ndarray] = None):
        """Visualize the grid world."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Color cells
                if (i, j) == self.start_pos:
                    color = 'lightgreen'
                    text = 'S'
                elif (i, j) in self.outlet_positions:
                    if (i, j) in self.visited_outlets:
                        color = 'gold'
                        text = 'O✓'
                    else:
                        color = 'yellow'
                        text = 'O'
                elif (i, j) in self.obstacle_positions:
                    color = 'red'
                    text = 'X'
                elif (i, j) == self.current_pos:
                    color = 'lightblue'
                    text = 'SPG'
                else:
                    color = 'white'
                    text = ''
                
                # Draw cell
                rect = plt.Rectangle((j, self.grid_size - 1 - i), 1, 1, 
                                    facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                
                # Add text
                if text:
                    ax.text(j + 0.5, self.grid_size - 1 - i + 0.5, text,
                           ha='center', va='center', fontsize=12, fontweight='bold')
                
                # Add Q-values arrows if Q-table provided
                if q_table is not None:
                    state_idx = self.get_state_index((i, j))
                    best_action = np.argmax(q_table[state_idx])
                    
                    # Draw small arrow for best action
                    arrow_size = 0.2
                    dx, dy = 0, 0
                    if best_action == 0:  # up
                        dy = arrow_size
                    elif best_action == 1:  # right
                        dx = arrow_size
                    elif best_action == 2:  # down
                        dy = -arrow_size
                    elif best_action == 3:  # left
                        dx = -arrow_size
                    
                    if dx != 0 or dy != 0:
                        ax.arrow(j + 0.5, self.grid_size - 1 - i + 0.5, 
                                dx, dy, head_width=0.15, head_length=0.1,
                                fc='blue', ec='blue', alpha=0.5)
        
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='lightgreen', label='Start (S)'),
            plt.Rectangle((0, 0), 1, 1, fc='yellow', label='Outlet (O)'),
            plt.Rectangle((0, 0), 1, 1, fc='red', label='Obstacle (X)'),
            plt.Rectangle((0, 0), 1, 1, fc='lightblue', label='SPG'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.title('SPG Grid World', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, ax


class QLearningAgent:
    """
    Q-Learning agent that learns optimal policy through interaction.
    """
    
    def __init__(self, n_states: int, n_actions: int, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 seed: Optional[int] = None):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_states: Number of possible states
            n_actions: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum epsilon value
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.seed = seed
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Tracking
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'epsilons': []
        }
    
    def choose_action(self, state_index: int, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state_index: Current state index
            training: Whether in training mode (use epsilon-greedy) or not
            
        Returns:
            action: Selected action
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(range(self.n_actions))
        else:
            # Exploitation: best known action
            return np.argmax(self.q_table[state_index])
    
    def update(self, state_index: int, action: int, reward: float, 
               next_state_index: int, done: bool):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        Args:
            state_index: Current state index
            action: Action taken
            reward: Reward received
            next_state_index: Next state index
            done: Whether episode is finished
        """
        # Current Q-value
        current_q = self.q_table[state_index, action]
        
        # Maximum Q-value for next state
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state_index])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_index, action] = new_q
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env: GridWorldSPG, n_episodes: int = 1000, 
              max_steps_per_episode: int = 100) -> Dict:
        """
        Train the agent using Q-learning.
        
        Args:
            env: Environment to train on
            n_episodes: Number of episodes to train
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            training_history: Dictionary with training metrics
        """
        for episode in range(n_episodes):
            state = env.reset()
            state_index = env.get_state_index(state)
            
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                # Choose action
                action = self.choose_action(state_index, training=True)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                next_state_index = env.get_state_index(next_state)
                
                # Update Q-table
                self.update(state_index, action, reward, next_state_index, done)
                
                # Update state
                state_index = next_state_index
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Record history
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(episode_reward)
            self.training_history['steps'].append(episode_steps)
            self.training_history['epsilons'].append(self.epsilon)
        
        return self.training_history
    
    def evaluate(self, env: GridWorldSPG, n_episodes: int = 100,
                max_steps_per_episode: int = 100) -> Dict:
        """
        Evaluate the learned policy.
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes to evaluate
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            results: Dictionary with evaluation metrics
        """
        total_rewards = []
        total_steps = []
        success_count = 0
        
        for _ in range(n_episodes):
            state = env.reset()
            state_index = env.get_state_index(state)
            
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                # Choose best action (no exploration)
                action = self.choose_action(state_index, training=False)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                next_state_index = env.get_state_index(next_state)
                
                state_index = next_state_index
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    if info.get('all_outlets_visited', False):
                        success_count += 1
                    break
            
            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps),
            'success_rate': success_count / n_episodes
        }


class RandomAgent:
    """Random agent for baseline comparison."""
    
    def __init__(self, n_actions: int, seed: Optional[int] = None):
        """Initialize random agent."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.n_actions = n_actions
    
    def choose_action(self, state_index: int = None) -> int:
        """Choose random action."""
        return random.choice(range(self.n_actions))
    
    def evaluate(self, env: GridWorldSPG, n_episodes: int = 100,
                max_steps_per_episode: int = 100) -> Dict:
        """Evaluate random policy."""
        total_rewards = []
        total_steps = []
        success_count = 0
        
        for _ in range(n_episodes):
            state = env.reset()
            
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                # Choose random action
                action = self.choose_action()
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    if info.get('all_outlets_visited', False):
                        success_count += 1
                    break
            
            total_rewards.append(episode_reward)
            total_steps.append(episode_steps)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps),
            'success_rate': success_count / n_episodes
        }


def plot_learning_curve(history: Dict, figsize: Tuple[int, int] = (15, 5)):
    """Plot learning curves from training history."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot rewards
    axes[0].plot(history['episodes'], history['rewards'], alpha=0.3)
    # Moving average
    window = 50
    if len(history['rewards']) >= window:
        moving_avg = np.convolve(history['rewards'], 
                                np.ones(window)/window, mode='valid')
        axes[0].plot(history['episodes'][window-1:], moving_avg, 
                    linewidth=2, label=f'{window}-episode moving avg')
        axes[0].legend()
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Learning Curve: Rewards')
    axes[0].grid(True, alpha=0.3)
    
    # Plot steps
    axes[1].plot(history['episodes'], history['steps'], alpha=0.3)
    if len(history['steps']) >= window:
        moving_avg = np.convolve(history['steps'], 
                                np.ones(window)/window, mode='valid')
        axes[1].plot(history['episodes'][window-1:], moving_avg, 
                    linewidth=2, label=f'{window}-episode moving avg')
        axes[1].legend()
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps per Episode')
    axes[1].set_title('Learning Curve: Steps')
    axes[1].grid(True, alpha=0.3)
    
    # Plot epsilon
    axes[2].plot(history['episodes'], history['epsilons'])
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Epsilon (Exploration Rate)')
    axes[2].set_title('Epsilon Decay')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_policies(q_agent_results: Dict, random_agent_results: Dict):
    """Compare Q-learning agent with random agent."""
    print("\n" + "="*60)
    print("POLICY COMPARISON")
    print("="*60)
    
    print("\nQ-Learning Agent:")
    print(f"  Mean Reward: {q_agent_results['mean_reward']:.2f} ± {q_agent_results['std_reward']:.2f}")
    print(f"  Mean Steps: {q_agent_results['mean_steps']:.2f} ± {q_agent_results['std_steps']:.2f}")
    print(f"  Success Rate: {q_agent_results['success_rate']*100:.1f}%")
    
    print("\nRandom Agent:")
    print(f"  Mean Reward: {random_agent_results['mean_reward']:.2f} ± {random_agent_results['std_reward']:.2f}")
    print(f"  Mean Steps: {random_agent_results['mean_steps']:.2f} ± {random_agent_results['std_steps']:.2f}")
    print(f"  Success Rate: {random_agent_results['success_rate']*100:.1f}%")
    
    print("\nImprovement:")
    reward_improvement = ((q_agent_results['mean_reward'] - random_agent_results['mean_reward']) 
                         / abs(random_agent_results['mean_reward']) * 100)
    print(f"  Reward: {reward_improvement:+.1f}%")
    
    if random_agent_results['mean_steps'] > 0:
        steps_improvement = ((random_agent_results['mean_steps'] - q_agent_results['mean_steps']) 
                           / random_agent_results['mean_steps'] * 100)
        print(f"  Steps (fewer is better): {steps_improvement:+.1f}%")
    
    success_improvement = (q_agent_results['success_rate'] - random_agent_results['success_rate']) * 100
    print(f"  Success Rate: {success_improvement:+.1f} percentage points")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    SEED = 42
    
    # Create environment
    print("Creating SPG Grid World environment...")
    env = GridWorldSPG(grid_size=5, n_outlets=3, n_obstacles=3, seed=SEED)
    
    # Visualize initial environment
    print("\nInitial Grid World:")
    env.reset()
    fig, _ = env.render()
    plt.savefig('gridworld_initial.png', dpi=150, bbox_inches='tight')
    print("Saved: gridworld_initial.png")
    plt.close()
    
    # Create Q-learning agent
    print("\nInitializing Q-Learning agent...")
    n_states = env.grid_size ** 2
    n_actions = len(env.actions)
    
    q_agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=SEED
    )
    
    # Train agent
    print("\nTraining Q-Learning agent...")
    print("Episodes: 1000")
    history = q_agent.train(env, n_episodes=1000, max_steps_per_episode=100)
    print("Training completed!")
    
    # Plot learning curve
    print("\nPlotting learning curves...")
    fig = plot_learning_curve(history)
    plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
    print("Saved: learning_curve.png")
    plt.close()
    
    # Visualize learned policy
    print("\nVisualizing learned policy...")
    env.reset()
    fig, _ = env.render(q_table=q_agent.q_table)
    plt.savefig('learned_policy.png', dpi=150, bbox_inches='tight')
    print("Saved: learned_policy.png")
    plt.close()
    
    # Evaluate Q-learning agent
    print("\nEvaluating Q-Learning agent...")
    q_results = q_agent.evaluate(env, n_episodes=100, max_steps_per_episode=100)
    
    # Create and evaluate random agent
    print("\nEvaluating Random agent (baseline)...")
    random_agent = RandomAgent(n_actions=n_actions, seed=SEED)
    random_results = random_agent.evaluate(env, n_episodes=100, max_steps_per_episode=100)
    
    # Compare policies
    compare_policies(q_results, random_results)
    
    print("\n✓ Q-Learning demonstration completed!")
    print("Check the output images for visualizations.")
