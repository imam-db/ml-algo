# Q-Learning

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## What is Q-Learning?

Q-Learning is an **off-policy** reinforcement learning algorithm that learns optimal action-value function Q*(s,a) through interaction with environment.

### Key Concept

- **Off-policy**: Learns optimal policy regardless of actions actually taken
- **Model-free**: Doesn't need environment model
- **Value-based**: Learns values of state-action pairs

### SPG Analogy

Imagine an SPG learning the best route to visit outlets:
- **State**: Current location
- **Action**: Direction to move
- **Reward**: Points for visiting outlets, penalties for obstacles
- **Q-value**: Expected total reward from taking action in state

---

## Q-Learning Update Rule

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

Where:
- $Q(s, a)$ = value of taking action $a$ in state $s$
- $\alpha$ = learning rate
- $r$ = immediate reward
- $\gamma$ = discount factor
- $\max_{a'} Q(s', a')$ = maximum Q-value in next state (off-policy)

---

## Algorithm

```
1. Initialize Q(s,a) = 0 for all states and actions

2. For each episode:
   a. Start at initial state s
   
   b. Repeat until terminal state:
      i.   Choose action a using ε-greedy policy
      ii.  Take action a, observe reward r and next state s'
      iii. Update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
      iv.  s ← s'
   
   c. Decay ε (reduce exploration over time)

3. Return learned Q-table
```

---

## Key Parameters

- **Learning rate (α)**: 0.1 - 0.3 typical
- **Discount factor (γ)**: 0.9 - 0.99 for long-term rewards
- **Epsilon (ε)**: Start at 1.0, decay to 0.01
- **Epsilon decay**: 0.995 per episode typical

---

## Implementation from Scratch

See `implementation.py` for complete code:

```python
from implementation import GridWorldSPG, QLearningAgent

# Create SPG grid world
env = GridWorldSPG(grid_size=5, n_outlets=3, n_obstacles=3, seed=42)

# Create Q-Learning agent
n_states = env.grid_size ** 2
n_actions = len(env.actions)
agent = QLearningAgent(n_states, n_actions, seed=42)

# Train
agent.train(env, n_episodes=1000)

# Evaluate
results = agent.evaluate(env, n_episodes=100)
print(f"Average reward: {results['mean_reward']:.2f}")
```

---

## Advantages and Disadvantages

### ✅ Advantages

1. **Off-policy** - Can learn optimal policy while exploring
2. **Simple** - Easy to understand and implement
3. **Proven convergence** - Guaranteed to converge to optimal under conditions
4. **No model needed** - Learns directly from experience

### ❌ Disadvantages

1. **Discrete only** - Requires discrete state/action spaces
2. **Doesn't scale** - Q-table grows exponentially with states
3. **Slow learning** - Can take many episodes to converge
4. **No generalization** - Each state learned independently

---

## Q-Learning vs SARSA

| Aspect | Q-Learning | SARSA |
|--------|-----------|-------|
| **Type** | Off-policy | On-policy |
| **Update** | Uses max Q(s',a') | Uses actual Q(s',a') |
| **Learning** | Learns optimal policy | Learns behavior policy |
| **Convergence** | Faster (usually) | More stable |
| **Best for** | Safe environments | Dangerous environments |

---

## References

- **Original paper**: Watkins, C.J.C.H. (1989). "Learning from Delayed Rewards"
- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (Chapter 6.5)

---

## Related Files

- `implementation.py` - Q-Learning and grid world from scratch
- `exercise.ipynb` - Interactive exercises
- `../02_sarsa/` - SARSA (on-policy) for comparison

**Happy Learning! 🎓**

*[🇮🇩 Indonesian version](README.id.md)*
