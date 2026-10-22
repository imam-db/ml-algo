# SARSA (State-Action-Reward-State-Action)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## What is SARSA?

SARSA is an **on-policy** Temporal Difference (TD) control algorithm for reinforcement learning. The name SARSA comes from the tuple used for updates: **(S, A, R, S', A')**.

### Fundamental Difference: On-Policy vs Off-Policy

| Aspect | SARSA (On-Policy) | Q-Learning (Off-Policy) |
|--------|------------------|------------------------|
| **Learns from** | Actions **actually taken** | **Theoretically optimal** actions |
| **Update rule** | Q(s,a) ← Q(s,a) + α[r + γ·Q(s',**a'**) - Q(s,a)] | Q(s,a) ← Q(s,a) + α[r + γ·**max** Q(s',a') - Q(s,a)] |
| **a' is** | Action chosen by agent (may be exploratory) | Best action (greedy) |
| **Policy** | Learns **behavior policy** (what's used) | Learns **target policy** (optimal) |
| **Conservative?** | ✅ Yes, more cautious | ❌ More aggressive |
| **Best for** | Dangerous environments, exploration matters | Safe environments, fast convergence |

---

## SPG (Sales Promotion Girl/Guy) Analogy

Imagine two SPGs learning the best route to visit outlets:

### SPG SARSA (On-Policy)
- **Learns from real experience**: "Yesterday I took a shortcut through the market and got stuck in traffic. I'll note that was bad."
- **Considers exploration**: If still trying new routes, Q-values reflect those experiments too
- **More conservative**: Doesn't assume always choosing optimal route, so more careful

### SPG Q-Learning (Off-Policy)
- **Learns from ideal**: "If I always chose the best route, results would be like this"
- **Ignores exploration**: Even when trying new routes, Q-values focus only on optimal
- **More optimistic**: Assumes always greedy in the future

**Result**:
- In **dangerous** environments (potholes, unsafe roads): SARSA safer because aware of risks
- In **safe** environments: Q-Learning learns faster by focusing directly on optimal

---

## SARSA Update Rule

### Formula

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot Q(s', a') - Q(s, a) \right]$$

Where:
- $s$ = current state
- $a$ = action taken
- $r$ = reward received
- $s'$ = next state
- **$a'$ = action that WILL BE TAKEN in $s'$** (may be exploratory!)
- $\alpha$ = learning rate
- $\gamma$ = discount factor

### Difference from Q-Learning

**Q-Learning**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]$$

Note:
- Q-Learning: $\max_{a'} Q(s', a')$ → always take the best (greedy)
- SARSA: $Q(s', a')$ → take the action actually chosen (may be exploratory!)

---

## SARSA Algorithm

```
1. Initialize Q(s,a) arbitrarily for all s, a

2. For each episode:
   
   a. Initialize state s
   
   b. Choose action a from s using policy derived from Q
      (e.g., epsilon-greedy)
   
   c. Repeat for each step of episode:
      
      i.   Take action a, observe r, s'
      
      ii.  Choose a' from s' using policy derived from Q ← KEY!
      
      iii. Update: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
      
      iv.  s ← s'; a ← a'
   
   d. Until s is terminal

3. Return learned Q-table
```

**Key**: Step c.ii — we **choose a' first** before updating, and use Q(s', a') corresponding to that action.

---

## Why is SARSA More Conservative?

### Example: Cliff Walking

Imagine a grid-world with a cliff:

```
[S][·][·][·][·][G]
[·][·][·][·][·][·]
[C][C][C][C][C][·]
```

- S = Start
- G = Goal
- C = Cliff (fall = reward -100)

**Optimal route**: Walk on middle row (safe, but longer)
**Risky route**: Walk along cliff edge (fast, but one mistake = fall)

#### Q-Learning (Off-Policy)
- Assumes agent will always be greedy (no exploration)
- Learns cliff-edge route because it's **theoretically optimal**
- **Problem**: If there's still epsilon exploration, agent sometimes falls!

#### SARSA (On-Policy)
- Aware that agent sometimes explores (random action)
- Learns safe middle route because **safer in practice**
- **Advantage**: More robust to noise and exploration

---

## When to Use SARSA vs Q-Learning?

### Use SARSA when:

✅ **Dangerous environment** - States with large penalties exist
✅ **Exploration important** - Need balance between exploration vs exploitation
✅ **Conservative better** - Prefer safe but slower learning
✅ **Real-world applications** - Agent will deploy with exploration

**Examples**: Robotics (falling = danger), autonomous driving, medical treatment

### Use Q-Learning when:

✅ **Safe environment** - No catastrophic failures
✅ **Speed matters** - Need fast convergence
✅ **Optimal policy clear** - Not much ambiguity
✅ **Offline learning** - Train in simulator, deploy greedy policy

**Examples**: Game AI, recommendation systems, trading algorithms

---

## Implementation from Scratch

See `implementation.py` for complete code. Main functions:

```python
from implementation import SARSAAgent
import sys
sys.path.append('../01_q_learning')
from implementation import GridWorldSPG

# Create environment
env = GridWorldSPG(grid_size=5, n_outlets=3, n_obstacles=3, seed=42)

# Create SARSA agent
agent = SARSAAgent(
    env=env,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    seed=42
)

# Train
history = agent.train(n_episodes=1000, verbose=True)

# Evaluate
avg_reward, avg_steps = agent.evaluate(n_episodes=100)
print(f"Average Reward: {avg_reward:.2f}")
print(f"Average Steps: {avg_steps:.1f}")
```

---

## Example: SARSA vs Q-Learning Comparison

```python
from implementation import compare_sarsa_vs_qlearning, plot_training_comparison

# Same environment
env = GridWorldSPG(grid_size=5, n_outlets=3, n_obstacles=3, seed=42)

# Train both
sarsa_agent, qlearning_agent = compare_sarsa_vs_qlearning(
    env, 
    n_episodes=1000,
    seed=42
)

# Plot comparison
fig = plot_training_comparison(sarsa_agent, qlearning_agent)
plt.show()
```

**Observations**:
- Both algorithms converge to good policies
- Q-Learning **may** learn slightly faster
- SARSA **may** be more stable (less variance)
- Differences depend on environment (safe vs risky)

---

## Advantages and Disadvantages

### ✅ Advantages of SARSA

1. **On-policy learning** - Learns from behavior actually used
2. **Conservative & safe** - More cautious in risky environments
3. **Robust to exploration** - Q-values reflect actual behavior
4. **Converges under reasonable conditions**

### ❌ Disadvantages of SARSA

1. **Learns slower** than Q-Learning in many cases
2. **Tied to behavior policy** - Must decay epsilon to converge to optimal
3. **High variance** due to random action selection in a'
4. **Requires careful epsilon decay** tuning

---

## References

- **Original paper**: Rummery, G. A., & Niranjan, M. (1994). "On-line Q-learning using connectionist systems."
- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (Chapter 6.4 - SARSA)
- **Comparison**: Cliff-walking example from Sutton & Barto

---

## Related Files

- `implementation.py` - SARSA implementation from scratch + comparison with Q-Learning
- `exercise.ipynb` - Interactive SARSA vs Q-Learning exercises
- `test_sarsa.py` - Unit tests
- `../01_q_learning/` - Q-Learning implementation for comparison

**Happy Learning! 🎓**

*[🇮🇩 Indonesian version](README.id.md)*
