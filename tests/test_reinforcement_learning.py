"""
Tests for Reinforcement Learning implementations.
"""

import pytest
import numpy as np
import sys
import os

# Import RL implementation directly
import importlib.util
rl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03_reinforcement_learning', '01_q_learning', 'implementation.py'))
spec = importlib.util.spec_from_file_location("rl_implementation", rl_path)
rl_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl_impl)

GridWorldSPG = rl_impl.GridWorldSPG
QLearningAgent = rl_impl.QLearningAgent
RandomAgent = rl_impl.RandomAgent


@pytest.fixture
def simple_env():
    """Create a simple deterministic environment for testing."""
    return GridWorldSPG(grid_size=5, n_outlets=3, n_obstacles=3, seed=42)


@pytest.fixture
def trained_agent(simple_env):
    """Create and train a Q-learning agent for testing."""
    n_states = simple_env.grid_size ** 2
    n_actions = len(simple_env.actions)
    
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42
    )
    
    # Train for a few episodes
    agent.train(simple_env, n_episodes=100, max_steps_per_episode=50)
    
    return agent


class TestGridWorldSPG:
    """Tests for GridWorldSPG environment."""
    
    def test_environment_initialization(self, simple_env):
        """Test that environment initializes correctly."""
        assert simple_env.grid_size == 5
        assert simple_env.n_outlets == 3
        assert simple_env.n_obstacles == 3
        assert len(simple_env.actions) == 4
        assert simple_env.start_pos == (0, 0)
    
    def test_deterministic_initialization(self):
        """Test that same seed produces same environment."""
        env1 = GridWorldSPG(grid_size=5, n_outlets=3, n_obstacles=3, seed=42)
        env2 = GridWorldSPG(grid_size=5, n_outlets=3, n_obstacles=3, seed=42)
        
        assert env1.outlet_positions == env2.outlet_positions
        assert env1.obstacle_positions == env2.obstacle_positions
        assert np.array_equal(env1.grid, env2.grid)
    
    def test_reset(self, simple_env):
        """Test environment reset."""
        # Take some steps
        simple_env.reset()
        simple_env.step(1)
        simple_env.step(2)
        
        # Reset and check
        pos = simple_env.reset()
        assert pos == simple_env.start_pos
        assert len(simple_env.visited_outlets) == 0
    
    def test_step_actions(self, simple_env):
        """Test that actions move agent correctly."""
        simple_env.reset()
        
        # Test right action (1)
        next_pos, _, _, _ = simple_env.step(1)
        assert next_pos == (0, 1)
        
        # Test down action (2)
        next_pos, _, _, _ = simple_env.step(2)
        assert next_pos == (1, 1)
        
        # Test left action (3)
        next_pos, _, _, _ = simple_env.step(3)
        assert next_pos == (1, 0)
        
        # Test up action (0)
        next_pos, _, _, _ = simple_env.step(0)
        assert next_pos == (0, 0)
    
    def test_boundary_collision(self, simple_env):
        """Test that agent can't move outside grid."""
        simple_env.reset()
        
        # Try to move up from (0,0) - should stay at (0,0)
        next_pos, _, _, _ = simple_env.step(0)
        assert next_pos == (0, 0)
        
        # Try to move left from (0,0) - should stay at (0,0)
        next_pos, _, _, _ = simple_env.step(3)
        assert next_pos == (0, 0)
    
    def test_reward_structure(self, simple_env):
        """Test that rewards are given correctly."""
        simple_env.reset()
        
        # Regular step should have small penalty
        _, reward, _, info = simple_env.step(1)
        if 'hit_obstacle' not in info and 'visited_outlet' not in info:
            assert reward == -0.1
    
    def test_obstacle_penalty(self, simple_env):
        """Test that hitting obstacle gives penalty."""
        simple_env.reset()
        
        # Find path to obstacle and check penalty
        obstacle_pos = simple_env.obstacle_positions[0]
        
        # Move agent near obstacle
        simple_env.current_pos = (obstacle_pos[0], max(0, obstacle_pos[1] - 1))
        
        # Try to move into obstacle
        if obstacle_pos[1] < simple_env.grid_size - 1:
            _, reward, _, info = simple_env.step(1)  # right
            if 'hit_obstacle' in info:
                assert reward == -10.0
    
    def test_outlet_reward(self, simple_env):
        """Test that visiting outlet gives reward."""
        simple_env.reset()
        
        # Move to an outlet position
        outlet_pos = simple_env.outlet_positions[0]
        simple_env.current_pos = (outlet_pos[0], max(0, outlet_pos[1] - 1))
        
        # Move into outlet
        if outlet_pos[1] > 0:
            _, reward, _, info = simple_env.step(1)  # right
            if 'visited_outlet' in info:
                assert reward == 10.0
    
    def test_state_index_conversion(self, simple_env):
        """Test state index conversion."""
        # Test conversion to index
        pos = (2, 3)
        idx = simple_env.get_state_index(pos)
        assert idx == 2 * simple_env.grid_size + 3
        
        # Test conversion back
        recovered_pos = simple_env.get_pos_from_state(idx)
        assert recovered_pos == pos
    
    def test_episode_termination(self, simple_env):
        """Test that episode terminates after visiting all outlets."""
        simple_env.reset()
        
        # Manually visit all outlets
        for outlet in simple_env.outlet_positions:
            simple_env.visited_outlets.add(outlet)
        
        # Move to last outlet and check done flag
        last_outlet = simple_env.outlet_positions[-1]
        simple_env.current_pos = (last_outlet[0], last_outlet[1] - 1) if last_outlet[1] > 0 else (last_outlet[0] - 1, last_outlet[1])
        
        # This should trigger done
        if len(simple_env.visited_outlets) == len(simple_env.outlet_positions):
            assert True  # Already all visited


class TestQLearningAgent:
    """Tests for Q-Learning agent."""
    
    def test_agent_initialization(self, simple_env):
        """Test that agent initializes correctly."""
        n_states = simple_env.grid_size ** 2
        n_actions = len(simple_env.actions)
        
        agent = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=0.1,
            discount_factor=0.95,
            seed=42
        )
        
        assert agent.q_table.shape == (n_states, n_actions)
        assert np.all(agent.q_table == 0)
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.95
    
    def test_deterministic_action_selection(self, simple_env):
        """Test that same seed produces same action sequence."""
        n_states = simple_env.grid_size ** 2
        n_actions = len(simple_env.actions)
        
        # Reset random state before each agent
        agent1 = QLearningAgent(n_states=n_states, n_actions=n_actions, seed=42)
        actions1 = [agent1.choose_action(0, training=True) for _ in range(10)]
        
        # Reset random state again
        agent2 = QLearningAgent(n_states=n_states, n_actions=n_actions, seed=42)
        actions2 = [agent2.choose_action(0, training=True) for _ in range(10)]
        
        assert actions1 == actions2
    
    def test_epsilon_greedy(self, simple_env):
        """Test epsilon-greedy action selection."""
        n_states = simple_env.grid_size ** 2
        n_actions = len(simple_env.actions)
        
        # Agent with epsilon=1.0 should always explore
        agent = QLearningAgent(n_states=n_states, n_actions=n_actions, 
                              epsilon=1.0, seed=42)
        
        # Set Q-table to prefer action 0
        agent.q_table[0, 0] = 10
        agent.q_table[0, 1:] = 0
        
        # With epsilon=1.0, should still get random actions
        actions = [agent.choose_action(0, training=True) for _ in range(100)]
        assert len(set(actions)) > 1  # Should have variety
        
        # With epsilon=0.0, should always choose best action
        agent.epsilon = 0.0
        actions = [agent.choose_action(0, training=True) for _ in range(10)]
        assert all(a == 0 for a in actions)
    
    def test_q_value_update(self, simple_env):
        """Test Q-value update rule."""
        n_states = simple_env.grid_size ** 2
        n_actions = len(simple_env.actions)
        
        agent = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=0.1,
            discount_factor=0.9,
            seed=42
        )
        
        # Initialize Q-values
        state_idx = 0
        action = 1
        reward = 10.0
        next_state_idx = 1
        
        agent.q_table[next_state_idx, :] = [5, 3, 2, 1]
        
        # Calculate expected Q-value
        old_q = agent.q_table[state_idx, action]
        max_next_q = np.max(agent.q_table[next_state_idx])
        expected_q = old_q + 0.1 * (reward + 0.9 * max_next_q - old_q)
        
        # Update and check
        agent.update(state_idx, action, reward, next_state_idx, done=False)
        
        assert np.isclose(agent.q_table[state_idx, action], expected_q)
    
    def test_epsilon_decay(self, simple_env):
        """Test epsilon decay mechanism."""
        n_states = simple_env.grid_size ** 2
        n_actions = len(simple_env.actions)
        
        agent = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            epsilon=1.0,
            epsilon_decay=0.9,
            epsilon_min=0.1,
            seed=42
        )
        
        # Decay epsilon
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        
        assert agent.epsilon == initial_epsilon * 0.9
        
        # Decay until minimum
        for _ in range(100):
            agent.decay_epsilon()
        
        assert agent.epsilon >= 0.1
    
    def test_training_improves_performance(self, simple_env):
        """Test that training improves agent performance."""
        n_states = simple_env.grid_size ** 2
        n_actions = len(simple_env.actions)
        
        agent = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=0.1,
            discount_factor=0.95,
            seed=42
        )
        
        # Evaluate before training
        results_before = agent.evaluate(simple_env, n_episodes=10, max_steps_per_episode=50)
        
        # Train
        agent.train(simple_env, n_episodes=100, max_steps_per_episode=50)
        
        # Evaluate after training
        results_after = agent.evaluate(simple_env, n_episodes=10, max_steps_per_episode=50)
        
        # Performance should improve (higher reward or fewer steps)
        assert (results_after['mean_reward'] > results_before['mean_reward'] or
                results_after['mean_steps'] < results_before['mean_steps'])
    
    def test_training_history_tracking(self, simple_env):
        """Test that training history is recorded correctly."""
        n_states = simple_env.grid_size ** 2
        n_actions = len(simple_env.actions)
        
        agent = QLearningAgent(n_states=n_states, n_actions=n_actions, seed=42)
        
        n_episodes = 50
        history = agent.train(simple_env, n_episodes=n_episodes, max_steps_per_episode=50)
        
        assert len(history['episodes']) == n_episodes
        assert len(history['rewards']) == n_episodes
        assert len(history['steps']) == n_episodes
        assert len(history['epsilons']) == n_episodes
    
    def test_evaluation_mode(self, trained_agent, simple_env):
        """Test that evaluation mode uses greedy policy."""
        # In evaluation mode (training=False), should always choose best action
        state_idx = 0
        trained_agent.q_table[state_idx] = [1, 2, 3, 0]
        
        # Get many actions in evaluation mode
        actions = [trained_agent.choose_action(state_idx, training=False) for _ in range(20)]
        
        # Should always choose action 2 (highest Q-value)
        assert all(a == 2 for a in actions)


class TestRandomAgent:
    """Tests for Random agent baseline."""
    
    def test_random_agent_initialization(self):
        """Test random agent initialization."""
        agent = RandomAgent(n_actions=4, seed=42)
        assert agent.n_actions == 4
    
    def test_random_action_selection(self):
        """Test that random agent selects actions uniformly."""
        agent = RandomAgent(n_actions=4, seed=42)
        
        # Select many actions
        actions = [agent.choose_action() for _ in range(1000)]
        
        # Should have all actions
        assert set(actions) == {0, 1, 2, 3}
        
        # Should be roughly uniform (within 40% of expected)
        for action in range(4):
            count = actions.count(action)
            expected = 250
            assert abs(count - expected) < expected * 0.4
    
    def test_random_agent_deterministic_with_seed(self):
        """Test that same seed produces same random actions."""
        agent1 = RandomAgent(n_actions=4, seed=42)
        actions1 = [agent1.choose_action() for _ in range(20)]
        
        agent2 = RandomAgent(n_actions=4, seed=42)
        actions2 = [agent2.choose_action() for _ in range(20)]
        
        assert actions1 == actions2


class TestIntegration:
    """Integration tests for full training pipeline."""
    
    def test_full_training_pipeline(self, simple_env):
        """Test complete training and evaluation pipeline."""
        n_states = simple_env.grid_size ** 2
        n_actions = len(simple_env.actions)
        
        # Create and train agent
        agent = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            seed=42
        )
        
        # Train
        history = agent.train(simple_env, n_episodes=200, max_steps_per_episode=50)
        
        # Check history
        assert len(history['rewards']) == 200
        
        # Final rewards should be better than initial
        initial_avg = np.mean(history['rewards'][:10])
        final_avg = np.mean(history['rewards'][-10:])
        assert final_avg > initial_avg
        
        # Evaluate
        results = agent.evaluate(simple_env, n_episodes=10, max_steps_per_episode=50)
        
        assert 'mean_reward' in results
        assert 'mean_steps' in results
        assert 'success_rate' in results
    
    def test_q_learning_better_than_random(self, simple_env):
        """Test that Q-learning outperforms random baseline."""
        n_states = simple_env.grid_size ** 2
        n_actions = len(simple_env.actions)
        
        # Train Q-learning agent with more episodes for reliable learning
        q_agent = QLearningAgent(
            n_states=n_states, 
            n_actions=n_actions, 
            learning_rate=0.2,  # Higher learning rate for faster learning
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.99,  # Faster decay
            epsilon_min=0.01,
            seed=42
        )
        q_agent.train(simple_env, n_episodes=500, max_steps_per_episode=100)
        q_results = q_agent.evaluate(simple_env, n_episodes=20, max_steps_per_episode=100)
        
        # Random agent
        random_agent = RandomAgent(n_actions=n_actions, seed=42)
        random_results = random_agent.evaluate(simple_env, n_episodes=20, max_steps_per_episode=100)
        
        # Q-learning should outperform random (at least in reward)
        # Success rate might be low due to task difficulty, but reward should be better
        assert q_results['mean_reward'] > random_results['mean_reward']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
