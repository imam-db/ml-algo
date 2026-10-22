# SARSA (State-Action-Reward-State-Action)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## What is SARSA?

SARSA is an **on-policy** Temporal Difference (TD) control algorithm for reinforcement learning. The name SARSA comes from the tuple used for the update: **(S, A, R, S', A')**.

### Fundamental Difference: On-Policy vs Off-Policy

| Aspect | SARSA (On-Policy) | Q-Learning (Off-Policy) |
|-------|------------------|------------------------|
| **Learns from** | Actions **actually taken** | **Theoretical optimal** actions |
| **Update rule** | Q(s,a) ← Q(s,a) + α[r + γ·Q(s',**a'**) - Q(s,a)] | Q(s,a) ← Q(s,a) + α[r + γ·**max** Q(s',a') - Q(s,a)] |
| **a' is** | The action chosen by the agent (may be exploratory) | The best action (greedy) |
| **Policy** | Learns the **behavior policy** (the one used) | Learns the **target policy** (optimal) |
| **Conservative?** | ✅ Yes, more careful | ❌ More aggressive |
| **Good for** | Dangerous environments, where exploration matters | Safe environments, fast convergence |

---

## SPG Analogy (Sales Promotion Girl/Guy)

Imagine two SPGs learning the best route for visiting outlets:

### SPG SARSA (On-Policy)
- **Learns from real experience**: "Yesterday I took a shortcut through the market, and it turned out to be congested. I noted that it was bad."
- **Accounts for exploration**: If it is still trying out new routes, the Q-values reflect those trial results as well
- **More conservative**: It doesn't assume it will always choose the optimal route, so it is more careful

### SPG Q-Learning (Off-Policy)
- **Learns from the ideal**: "If I always chose the best route, the results would be like this"
- **Ignores exploration**: Although it sometimes tries new routes, the Q-values focus only on the optimal route
- **More optimistic**: It assumes it will always act greedily in the future

**Result**:
- In **dangerous** environments (with potholes, damaged roads): SARSA is safer because it is aware of the risk
- In **safe** environments: Q-Learning learns faster because it focuses directly on the optimal route

---

## SARSA Update Rule

### Formula

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot Q(s', a') - Q(s, a) \right]$$

Where:
- $s$ = current state
- $a$ = action taken
- $r$ = reward received
- $s'$ = next state
- **$a'$ = the action that will REALLY be taken in $s'$** (can be exploratory!)
- $\alpha$ = learning rate
- $\gamma$ = discount factor

### Difference from Q-Learning

**Q-Learning**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]$$

Note:
- Q-Learning: $\max_{a'} Q(s', a')$ → always takes the best (greedy)
- SARSA: $Q(s', a')$ → takes the action actually chosen (can be exploratory!)

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

**Key**: Step c.ii — we **choose a' first** before the update, then use the Q(s', a') that corresponds to that action.

---

## Why Is SARSA More Conservative?

### Example: Cliff Walking

Imagine a grid-world with a cliff:

```
[S][·][·][·][·][G]
[·][·][·][·][·][·]
[C][C][C][C][C][·]
```

- S = Start
- G = Goal
- C = Cliff (falling = reward -100)

**Optimal route**: Walk along the middle row (safe, but somewhat longer)
**Risky route**: Walk along the edge of the cliff (fast, but one mistake = falling)

#### Q-Learning (Off-Policy)
- Assumes the agent will always act greedily (no exploration)
- Learns the cliff-edge route because it is **theoretically optimal**
- **Problem**: If epsilon exploration is still present, the agent sometimes falls!

#### SARSA (On-Policy)
- Aware that the agent sometimes explores (random action)
- Learns the safe middle route because it is **safer in practice**
- **Advantage**: More robust to noise and exploration

---

## Convergence

### SARSA Convergence Theorem

SARSA converges to the optimal policy $Q^*$ if:

1. All state-action pairs are visited infinitely often
2. The policy converges in the limit to the greedy policy (epsilon → 0)
3. The learning rate $\alpha_t$ satisfies:
   - $\sum_{t=1}^{\infty} \alpha_t = \infty$ (does not converge too quickly)
   - $\sum_{t=1}^{\infty} \alpha_t^2 < \infty$ (does not diverge)

In practice: Use epsilon decay and a small constant learning rate (0.1).

---

## Parameter Tuning

### Learning Rate (α)

- **Too large** (0.5-1.0): Drastic updates, unstable, can oscillate
- **Too small** (0.001-0.01): Learns very slowly
- **Recommended**: 0.1 - 0.3

```python
agent = SARSAAgent(env, learning_rate=0.1)
```

### Discount Factor (γ)

- **γ = 0**: Myopic (only cares about immediate reward)
- **γ = 0.9-0.95**: Balance (prefer sooner rewards)
- **γ = 0.99**: Far-sighted (value future rewards almost equally)

```python
agent = SARSAAgent(env, discount_factor=0.95)
```

### Epsilon Decay

- **Start high** (1.0): Full exploration at the beginning
- **Decay gradually** (0.995 per episode): Reduce exploration
- **End low** (0.01): Still a little exploration for robustness

```python
agent = SARSAAgent(env, 
                  epsilon=1.0,
                  epsilon_decay=0.995,
                  epsilon_min=0.01)
```

---

## SARSA vs Q-Learning: When to Use Which?

### Use SARSA when:

✅ **Dangerous environments** - There are states with large penalties
✅ **Exploration matters** - You need to balance exploration vs exploitation
✅ **Conservative is better** - You prefer safe but slower learning
✅ **Real-world applications** - The agent will be deployed with exploration

**Examples**: Robotics (falling = danger), autonomous driving, medical treatment

### Use Q-Learning when:

✅ **Safe environment** - No catastrophic failures
✅ **Speed matters** - You need fast convergence
✅ **Optimal policy is clear** - Not much ambiguity
✅ **Offline learning** - Train in a simulator, deploy the greedy policy

**Examples**: Game AI, recommendation systems, trading algorithms

---

## From-Scratch Implementation

See `implementation.py` for the full code. Main functions:

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

# Environment that sama
env = GridWorldSPG(grid_size=5, n_outlets=3, n_obstacles=3, seed=42)

# Train both
sarsa_agent, qlearning_agent = compare_sarsa_vs_qlearning(
    env, 
    n_episodes=1000,
    seed=42
)

# Plot comparison
fig = plot_training_comparison(sarsa_agent, qlearning_agent)
plt.savefig('sarsa_vs_qlearning.png')
plt.show()
```

**Observations**:
- Both algorithms converge to a good policy
- Q-Learning **may** learn slightly faster
- SARSA **may** be more stable (less variance)
- The difference depends on the environment (safe vs risky)

---

## SARSA Variants

### Expected SARSA

Updates with the **expected value** over actions (weighted by the policy):

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \sum_{a'} \pi(a'|s') Q(s', a') - Q(s, a) \right]$$

- More stable than SARSA
- Reduces variance from random action selection
- Still on-policy

### SARSA(λ)

Uses **eligibility traces** for multi-step updates:

- The eligibility trace $e_t(s,a)$ tracks the "eligibility" of each state-action for the update
- Updates many state-action pairs at once based on the TD error
- More sample-efficient
- Parameter $\lambda \in [0, 1]$:
  - $\lambda = 0$: Ordinary SARSA (1-step)
  - $\lambda = 1$: Monte Carlo (full episode)

---

## Advantages and Disadvantages

### ✅ SARSA Advantages

1. **On-policy learning** - Learns from the behavior actually used
2. **Conservative & safe** - More careful in risky environments
3. **Robust to exploration** - Q-values reflect actual behavior
4. **Converges under reasonable conditions**

### ❌ SARSA Disadvantages

1. **Learns more slowly** than Q-Learning in many cases
2. **Tied to the behavior policy** - Must decay epsilon to converge to optimal
3. **High variance** because of random action selection at a'
4. **Requires careful tuning** of epsilon decay

---

## Advanced: On-Policy vs Off-Policy Deep Dive

### On-Policy (SARSA)

**The policy being learned = the policy being used**

```
Behavior Policy:  ε-greedy based on Q
Target Policy:    ε-greedy based on Q (same!)
```

**Implications**:
- The agent learns values from the policy actually used (including exploration)
- Conservative: aware of the risk from exploratory actions

### Off-Policy (Q-Learning)

**The policy being learned ≠ the policy being used**

```
Behavior Policy:  ε-greedy (for exploration)
Target Policy:    Greedy (optimal, no exploration)
```

**Implications**:
- The agent learns values from the **optimal policy** (greedy)
- Exploratory actions are only for learning and do not enter the Q-values
- More optimistic about future rewards

---

## References

- **Original paper**: Rummery, G. A., & Niranjan, M. (1994). "On-line Q-learning using connectionist systems."
- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (Chapter 6.4 - SARSA)
- **Comparison**: Cliff-walking example from Sutton & Barto

---

## Related Files

- `implementation.py` - From-scratch SARSA implementation + comparison with Q-Learning
- `exercise.ipynb` - SARSA vs Q-Learning interactive exercises
- `test_sarsa.py` - Unit tests
- `../01_q_learning/` - Q-Learning implementation for comparison

**Happy learning! 🎓**