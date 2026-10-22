"""
Tests for SARSA reinforcement learning implementation.
"""

import numpy as np
import pytest
import importlib.util
import sys

# Import Q-learning GridWorldSPG
spec_qlearning = importlib.util.spec_from_file_location(
    "qlearning_implementation",
    "03_reinforcement_learning/01_q_learning/implementation.py"
)
qlearning_module = importlib.util.module_from_spec(spec_qlearning)
sys.modules["qlearning_implementation"] = qlearning_module
spec_qlearning.loader.exec_module(qlearning_module)

GridWorldSPG = qlearning_module.GridWorldSPG

# Import SARSA implementation
spec_sarsa = importlib.util.spec_from_file_location(
    "sarsa_implementation",
    "03_reinforcement_learning/02_sarsa/implementation.py"
)
sarsa_module = importlib.util.module_from_spec(spec_sarsa)
sys.modules["sarsa_implementation"] = sarsa_module
spec_sarsa.loader.exec_module(sarsa_module)

SARSAAgent = sarsa_module.SARSAAgent


class TestSARSAAgent:
    """Test SARSA agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent = SARSAAgent(env, seed=42)
        
        assert agent.env == env
        assert agent.learning_rate == 0.1
        assert agent.discount_factor == 0.95
        assert agent.epsilon == 1.0
        assert len(agent.q_table) == env.grid_size ** 2
    
    def test_custom_parameters(self):
        """Test agent with custom parameters."""
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent = SARSAAgent(
            env,
            learning_rate=0.5,
            discount_factor=0.9,
            epsilon=0.5,
            epsilon_decay=0.99,
            epsilon_min=0.05,
            seed=42
        )
        
        assert agent.learning_rate == 0.5
        assert agent.discount_factor == 0.9
        assert agent.epsilon == 0.5
        assert agent.epsilon_decay == 0.99
        assert agent.epsilon_min == 0.05
    
    def test_select_action_exploit(self):
        """Test action selection in exploit mode."""
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent = SARSAAgent(env, seed=42)
        
        # Set Q-values to favor a specific action
        state = (0, 0)
        agent.q_table[state] = np.array([10.0, 0.0, 0.0, 0.0])
        
        # With use_epsilon=False, should always select action 0
        action = agent.select_action(state, use_epsilon=False)
        assert action == 0
    
    def test_select_action_explore(self):
        """Test action selection with exploration."""
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent = SARSAAgent(env, epsilon=1.0, seed=42)
        
        state = (0, 0)
        agent.q_table[state] = np.array([10.0, 0.0, 0.0, 0.0])
        
        # With epsilon=1.0, should sometimes explore (random actions)
        actions = [agent.select_action(state, use_epsilon=True) for _ in range(100)]
        
        # Should have variety of actions due to exploration
        assert len(set(actions)) > 1
    
    def test_update_q_value(self):
        """Test Q-value update using SARSA rule."""
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent = SARSAAgent(env, learning_rate=0.1, discount_factor=0.9, seed=42)
        
        state = (0, 0)
        action = 1
        reward = 1.0
        next_state = (0, 1)
        next_action = 1
        
        # Initial Q-value
        initial_q = agent.q_table[state][action]
        
        # Update
        agent.update_q_value(state, action, reward, next_state, next_action)
        
        # Q-value should change
        updated_q = agent.q_table[state][action]
        assert updated_q != initial_q
    
    def test_training_basic(self):
        """Test basic training completes successfully."""
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent = SARSAAgent(env, seed=42)
        
        history = agent.train(n_episodes=10, max_steps=20, verbose=False)
        
        assert 'episode_rewards' in history
        assert 'episode_steps' in history
        assert 'epsilon_values' in history
        assert len(history['episode_rewards']) == 10
    
    def test_training_improves_performance(self):
        """Test that training improves agent performance."""
        env = GridWorldSPG(grid_size=4, n_outlets=2, n_obstacles=2, seed=42)
        agent = SARSAAgent(env, seed=42)
        
        # Train agent
        agent.train(n_episodes=200, max_steps=50, verbose=False)
        
        # Average reward should improve over time
        early_rewards = agent.training_history['episode_rewards'][:50]
        late_rewards = agent.training_history['episode_rewards'][-50:]
        
        assert np.mean(late_rewards) > np.mean(early_rewards)
    
    def test_epsilon_decay(self):
        """Test that epsilon decays during training."""
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent = SARSAAgent(env, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01, seed=42)
        
        initial_epsilon = agent.epsilon
        agent.train(n_episodes=100, verbose=False)
        final_epsilon = agent.epsilon
        
        assert final_epsilon < initial_epsilon
        assert final_epsilon >= agent.epsilon_min
    
    def test_evaluate(self):
        """Test agent evaluation."""
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent = SARSAAgent(env, seed=42)
        
        # Train briefly
        agent.train(n_episodes=50, verbose=False)
        
        # Evaluate
        avg_reward, avg_steps = agent.evaluate(n_episodes=10)
        
        assert isinstance(avg_reward, (int, float))
        assert isinstance(avg_steps, (int, float))
        assert avg_steps > 0
    
    def test_get_policy(self):
        """Test policy extraction."""
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent = SARSAAgent(env, seed=42)
        
        # Train
        agent.train(n_episodes=50, verbose=False)
        
        # Get policy
        policy = agent.get_policy()
        
        assert isinstance(policy, dict)
        assert len(policy) == env.grid_size ** 2
        
        # All actions should be valid
        for state, action in policy.items():
            assert action in env.actions
    
    def test_deterministic_with_seed(self):
        """Test that training is deterministic with same seed."""
        env1 = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent1 = SARSAAgent(env1, seed=42)
        agent1.train(n_episodes=50, verbose=False)
        
        env2 = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent2 = SARSAAgent(env2, seed=42)
        agent2.train(n_episodes=50, verbose=False)
        
        # Q-tables should be identical
        for state in agent1.q_table:
            np.testing.assert_array_almost_equal(
                agent1.q_table[state],
                agent2.q_table[state],
                decimal=5
            )


class TestSARSAvsQLearning:
    """Test comparisons between SARSA and Q-Learning."""
    
    def test_both_agents_converge(self):
        """Test that both SARSA and Q-Learning converge to good policies."""
        env_sarsa = GridWorldSPG(grid_size=4, n_outlets=2, n_obstacles=2, seed=42)
        env_qlearning = GridWorldSPG(grid_size=4, n_outlets=2, n_obstacles=2, seed=42)
        
        QLearningAgent = qlearning_module.QLearningAgent
        
        sarsa_agent = SARSAAgent(env_sarsa, seed=42)
        sarsa_agent.train(n_episodes=300, verbose=False)
        
        n_states = env_qlearning.grid_size ** 2
        n_actions = len(env_qlearning.actions)
        qlearning_agent = QLearningAgent(n_states, n_actions, seed=42)
        qlearning_agent.train(env=env_qlearning, n_episodes=300, max_steps_per_episode=100)
        
        # Both should achieve positive average reward
        sarsa_reward, _ = sarsa_agent.evaluate(n_episodes=50)
        qlearning_results = qlearning_agent.evaluate(env_qlearning, n_episodes=50)
        qlearning_reward = qlearning_results['mean_reward']
        
        assert sarsa_reward > 0
        assert qlearning_reward > 0
    
    def test_sarsa_is_on_policy(self):
        """Test that SARSA updates use actual next action (on-policy)."""
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent = SARSAAgent(env, learning_rate=0.1, seed=42)
        
        state = (0, 0)
        action = 1
        reward = 1.0
        next_state = (0, 1)
        
        # Set specific Q-values
        agent.q_table[next_state] = np.array([5.0, 3.0, 1.0, 0.0])
        
        # The actual next action taken (not necessarily max!)
        next_action = 2  # Not the max action!
        
        initial_q = agent.q_table[state][action]
        agent.update_q_value(state, action, reward, next_state, next_action)
        
        # Update should use Q(next_state, next_action) where next_action = 2
        # Not max Q(next_state, a) which would be action 0
        expected_target = reward + agent.discount_factor * agent.q_table[next_state][next_action]
        expected_q = initial_q + agent.learning_rate * (expected_target - initial_q)
        
        np.testing.assert_almost_equal(
            agent.q_table[state][action],
            expected_q,
            decimal=10
        )


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_environment(self):
        """Test with minimal 2x2 environment."""
        env = GridWorldSPG(grid_size=2, n_outlets=1, n_obstacles=0, seed=42)
        agent = SARSAAgent(env, seed=42)
        
        agent.train(n_episodes=50, verbose=False)
        avg_reward, avg_steps = agent.evaluate(n_episodes=10)
        
        assert avg_steps > 0
    
    def test_zero_learning_rate(self):
        """Test with zero learning rate (no learning)."""
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        agent = SARSAAgent(env, learning_rate=0.0, seed=42)
        
        # Store initial Q-table
        initial_q = {s: agent.q_table[s].copy() for s in agent.q_table}
        
        agent.train(n_episodes=10, verbose=False)
        
        # Q-table should not change with learning_rate=0
        for state in agent.q_table:
            np.testing.assert_array_equal(agent.q_table[state], initial_q[state])
    
    def test_max_steps_limit(self):
        """Test that episodes respect max_steps limit."""
        env = GridWorldSPG(grid_size=5, n_outlets=3, n_obstacles=3, seed=42)
        agent = SARSAAgent(env, seed=42)
        
        max_steps = 10
        agent.train(n_episodes=20, max_steps=max_steps, verbose=False)
        
        # All episodes should have <= max_steps
        assert all(steps <= max_steps for steps in agent.training_history['episode_steps'])


class TestCompareSarsaVsQLearning:
    """Regression test for #24: compare_sarsa_vs_qlearning used an undefined name."""

    def test_compare_runs(self):
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        sarsa_agent, qlearning_agent = sarsa_module.compare_sarsa_vs_qlearning(
            env, n_episodes=5, seed=0
        )
        assert isinstance(sarsa_agent, SARSAAgent)
        assert type(qlearning_agent).__name__ == "QLearningAgent"
        assert len(qlearning_agent.training_history["rewards"]) == 5

    def test_plot_training_comparison(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        env = GridWorldSPG(grid_size=3, n_outlets=1, n_obstacles=1, seed=42)
        sarsa_agent, qlearning_agent = sarsa_module.compare_sarsa_vs_qlearning(
            env, n_episodes=60, seed=0
        )
        fig = sarsa_module.plot_training_comparison(sarsa_agent, qlearning_agent)
        assert fig is not None
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
