# Q-Learning

**🏠 [Back to Main](../../README.md)** | **📁 [Reinforcement Learning](../README.md)**

## 🗺️ Quick Access

| File | Description | Run |
|------|-----------|----------|
| [`implementation.py`](./implementation.py) | From-scratch implementation | `python3 implementation.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Interactive exercises | `jupyter lab exercise.ipynb` |

## 📖 Theory and Concepts

### What is Q-Learning?

**Q-Learning** is a reinforcement learning algorithm that learns an optimal policy for taking actions in an environment by trying things out and learning from experience. The algorithm does not require an environment model (model-free) and can learn an optimal policy through exploration.

Imagine a SPG (Sales Promotion Girl) who must visit outlets in a grid area to promote a product. She needs to learn the best route for visiting all outlets while avoiding obstacles and minimizing the number of steps taken.

### Mathematical Formulation

**Q-function:** Estimates the value of actions taken in a given state:

```
Q(s, a) = expected value of taking action a in state s
```

**Q-Learning Update Rule:**

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                              a'
```

Where:
- `s` = current state (position)
- `a` = action taken
- `s'` = next state after the action
- `r` = reward received
- `α` (alpha) = learning rate (0 < α ≤ 1)
- `γ` (gamma) = discount factor (0 ≤ γ ≤ 1)
- `max Q(s',a')` = maximum Q-value for the next state

### Main Components

#### 1. **State**
The agent's position in the grid world. In the SPG example:
- State = (row, column) coordinates in the grid
- Total states = grid_size × grid_size

#### 2. **Action**
Actions the agent can take:
- 0: Up (↑)
- 1: Right (→)
- 2: Down (↓)
- 3: Left (←)

#### 3. **Reward**
Numerical feedback from the environment:
- **+10**: Visiting a new outlet
- **+50**: Bonus for visiting all outlets
- **-10**: Hitting an obstacle
- **-0.1**: Small penalty per step (encourages efficiency)
- **0**: Visiting an outlet that has already been visited

#### 4. **Policy**
Strategy for choosing actions:
- **Epsilon-greedy**: Balances exploration vs exploitation
  - With probability ε: choose a random action (exploration)
  - With probability (1-ε): choose the best action (exploitation)
  - ε decreases over time (epsilon decay)

#### 5. **Q-Table**
A table that stores Q-values for each (state, action) pair:
- Dimensions: [n_states × n_actions]
- Initialized with zeros
- Updated during training

### Q-Learning Algorithm

1. **Initialization:**
   - Create a Q-table with zeros
   - Set parameters (α, γ, ε)

2. **For each episode:**
   - Reset the environment to the initial state
   - For each step in the episode:
     - Choose an action using epsilon-greedy
     - Execute the action, obtain the reward and next state
     - Update the Q-value using the Q-learning rule
     - Update the state
     - If done, end the episode
   - Decrease epsilon (decay)

3. **Evaluation:**
   - Use the learned policy (ε = 0)
   - Measure performance

### Exploration vs Exploitation

**Fundamental dilemma in RL:**

- **Exploration:**
  - Trying new actions to find better strategies
  - Important at the beginning of training
  - Controlled by epsilon (ε)

- **Exploitation:**
  - Using knowledge that has already been learned
  - Important during evaluation
  - Choosing the action with the highest Q-value

**Epsilon-Greedy Strategy:**
```python
if random() < epsilon:
    action = random_action()  # Eksplorasi
else:
    action = argmax(Q[state])  # Eksploitasi
```

**Epsilon Decay:**
- Start with ε = 1.0 (100% exploration)
- Decrease gradually: ε = ε × decay_rate
- Minimum: ε_min = 0.01 (1% exploration)

### Hyperparameters

#### Learning Rate (α)
How much new Q-values replace the old ones:
- **α = 0**: Does not learn (Q-values do not change)
- **α = 1**: Uses only the most recent information
- **Typical: α = 0.1 - 0.3**
- Large α: learns quickly, but less stable
- Small α: learns slowly, more stable

#### Discount Factor (γ)
How important future rewards are:
- **γ = 0**: Only cares about immediate rewards
- **γ = 1**: Future rewards are equally important
- **Typical: γ = 0.9 - 0.99**
- Large γ: considers long-term rewards
- Small γ: focuses on short-term rewards

#### Epsilon (ε)
Exploration rate:
- **ε = 1.0**: 100% random (full exploration)
- **ε = 0.0**: 100% greedy (full exploitation)
- **Decay rate: 0.99 - 0.999**
- **Minimum: 0.01**

## 🎯 When to Use Q-Learning?

### ✅ Good for:
- **Discrete state and action spaces**
  - Grid worlds, simple games
- **Model-free learning**
  - No need to know the environment dynamics
- **Off-policy learning**
  - Can learn from others' experience
- **Tabular RL problems**
  - State space is not too large
- **Optimal control problems**
  - Finding an optimal policy

### ❌ Less suitable for:
- **Continuous state/action spaces**
  - Use Deep Q-Networks (DQN)
- **Very large state spaces**
  - The Q-table becomes too large
  - Use function approximation
- **Partial observability**
  - Use POMDP solvers
- **Multi-agent scenarios**
  - Requires multi-agent algorithms
- **Real-time requirements**
  - Training can be slow

## 💡 Example Implementation

### Grid World SPG

Environment used in the implementation:

```
Grid 5×5 with:
- S: Start (posisi awal SPG)
- O: Outlet (tujuan, reward +10)
- X: Obstacle (obstacles, penalty -10)
- SPG: Posisi agent saat this
```

**Example Grid:**
```
S  .  .  O  .
.  X  .  .  .
.  .  O  X  .
O  .  .  .  X
.  .  .  .  .
```

**Goal:** The SPG must visit all outlets (O) while avoiding obstacles (X) with the most efficient route possible.

### Training Results

After training for 1000 episodes:
- **Converged Q-table** shows optimal actions
- **Reward increases** from random to optimal
- **Steps decrease** as the agent learns an efficient route
- **High success rate** (>90%)

**The learning curve shows:**
1. **Exploration Phase** (episodes 0-200):
   - Volatile reward
   - High epsilon
   - Agent tries various strategies

2. **Transition Phase** (episodes 200-500):
   - Reward starts to stabilize
   - Epsilon decreases
   - Agent finds good strategies

3. **Exploitation Phase** (episodes 500-1000):
   - Stable and high reward
   - Low epsilon
   - Agent consistently uses the optimal strategy

### Comparison with Random Policy

| Metric | Q-Learning | Random | Improvement |
|--------|-----------|--------|-------------|
| Mean Reward | ~35-45 | ~-50 | +170% |
| Mean Steps | ~15-20 | ~50-100 | -70% |
| Success Rate | 90-100% | <10% | +90 pts |

## 🔬 Experiments and Variations

### 1. Grid Size
- **Small (3×3)**: Learns quickly, but limited
- **Medium (5×5)**: Balance between complexity and speed
- **Large (10×10)**: More challenging, longer training

### 2. Number of Outlets/Obstacles
- More outlets: harder task, larger reward
- More obstacles: requires a more careful strategy

### 3. Reward Shaping
Try reward variations:
```python
# Dense rewards (more many feedback)
- Reward kecil for mendekati outlet
- Penalty bertingkat for obstacle

# Sparse rewards (more realistis)
- Hanya reward saat mencapai goal
- More sulit, tapi more general
```

### 4. Learning Rate Schedule
```python
# Constant learning rate
alpha = 0.1

# Decaying learning rate
alpha = alpha_initial / (1 + episode * decay_rate)

# Adaptive learning rate
alpha = f(state_visit_count)
```

## 📊 Evaluation Metrics

### 1. Episode Reward
- Total reward per episode
- Measures the quality of the policy
- Target: maximize reward

### 2. Steps per Episode
- Number of steps until completion
- Measures efficiency
- Target: minimize steps

### 3. Success Rate
- Percentage of successful episodes
- Measures reliability
- Target: close to 100%

### 4. Convergence Speed
- Episodes until convergence
- Measures learning efficiency
- Faster = better

### 5. Q-Value Stability
- Variance of Q-values over time
- Measures learning stability
- Low = stable

## 🔗 Relationship to Other Algorithms

### Q-Learning vs SARSA
- **Q-Learning**: Off-policy (learns the optimal policy)
- **SARSA**: On-policy (learns the policy being used)
- Q-Learning is more aggressive, SARSA is safer

### Q-Learning vs Deep Q-Networks (DQN)
- **Q-Learning**: Tabular (Q-table)
- **DQN**: Neural network approximation
- DQN for large state spaces

### Q-Learning vs Policy Gradient
- **Q-Learning**: Value-based (learns the Q-function)
- **Policy Gradient**: Policy-based (learns the policy directly)
- Q-Learning is more sample-efficient for discrete actions

## 📚 Learning Resources

- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [OpenAI Spinning Up - Q-Learning](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [UC Berkeley CS188 - Reinforcement Learning](https://inst.eecs.berkeley.edu/~cs188/)

## 🎓 Exercises

See [`exercise.ipynb`](./exercise.ipynb) for hands-on practice:
1. Experiment with hyperparameters
2. Modify the reward structure
3. Build a custom environment
4. Visualize the learned policy
5. Compare with other algorithms

## 💻 Run the Code

```bash
# Run implementasi from nol
python3 implementation.py

# Buka notebook latihan
jupyter lab exercise.ipynb
```

---

**Happy Learning Q-Learning! 🎮🤖**