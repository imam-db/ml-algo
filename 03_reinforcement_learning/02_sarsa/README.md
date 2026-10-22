# SARSA (State-Action-Reward-State-Action)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## What is SARSA?

SARSA is algoritma **on-policy** Temporal Difference (TD) control for reinforcement learning. Nama SARSA comes from tuple that used for update: **(S, A, R, S', A')**.

### Perbedaan Fundamental: On-Policy vs Off-Policy

| Aspek | SARSA (On-Policy) | Q-Learning (Off-Policy) |
|-------|------------------|------------------------|
| **Belajar from** | Aksi that **actually taken** | Aksi **optimal teoritis** |
| **Update rule** | Q(s,a) ← Q(s,a) + α[r + γ·Q(s',**a'**) - Q(s,a)] | Q(s,a) ← Q(s,a) + α[r + γ·**max** Q(s',a') - Q(s,a)] |
| **a' is** | Aksi that chosen agent (bisa eksplorasi) | Aksi terbaik (greedy) |
| **Policy** | Learns **behavior policy** (that used) | Learns **target policy** (optimal) |
| **Konservatif?** | ✅ Ya, more careful | ❌ More aggressive |
| **Good for** | Environment berbahaya, eksplorasi penting | Environment aman, konvergensi cepat |

---

## Analogi SPG (Sales Promotion Girl/Guy)

Bayangkan dua SPG learns rute terbaik mengunjungi outlet:

### SPG SARSA (On-Policy)
- **Belajar from pengalaman nyata**: "Kemarin aku ambil jalan shortcuts melewati pasar, ternyata macet. Aku catat itu buruk."
- **Mempertimbangkan eksplorasi**: If sometimes masih coba-coba rute baru, Q-values mencerminkan trial results as well
- **More conservative**: Doesn't assume it will always pilih rute optimal, jadi more careful

### SPG Q-Learning (Off-Policy)
- **Belajar from that ideal**: "Seandainya aku always choose rute terbaik, results will be like this"
- **Mengabaikan eksplorasi**: Although sometimes coba rute baru, Q-values only focuses at rute optimal
- **More optimistic**: Assume dia will always greedy di masa depan

**Hasil**:
- Di environment **berbahaya** (ada lubang, jalan rusak): SARSA more aman because aware of risikonya
- Di environment **aman**: Q-Learning learns more cepat because directly focuses to optimal

---

## Update Rule SARSA

### Formula

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot Q(s', a') - Q(s, a) \right]$$

Di mana:
- $s$ = state sekarang
- $a$ = action that taken
- $r$ = reward that diterima
- $s'$ = state berikutnya
- **$a'$ = action that BENAR-BENAR will be taken di $s'$** (bisa eksplorasi!)
- $\alpha$ = learning rate
- $\gamma$ = discount factor

### Perbedaan with Q-Learning

**Q-Learning**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]$$

Perhatikan:
- Q-Learning: $\max_{a'} Q(s', a')$ → selalu ambil that terbaik (greedy)
- SARSA: $Q(s', a')$ → ambil action that actually chosen (bisa exploratory!)

---

## Algoritma SARSA

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

**Key**: Step c.ii — we **choose a' first** before update, and gunakan Q(s', a') that sesuai with that action.

---

## Kenapa SARSA More Conservative?

### Contoh: Cliff Walking

Bayangkan grid-world with cliff (jurang):

```
[S][·][·][·][·][G]
[·][·][·][·][·][·]
[C][C][C][C][C][·]
```

- S = Start
- G = Goal
- C = Cliff (jatuh = reward -100)

**Rute optimal**: Jalan di baris tengah (aman, tapi agak jauh)
**Rute berisiko**: Jalan di pinggir cliff (cepat, tapi 1 kesalahan = jatuh)

#### Q-Learning (Off-Policy)
- Assume agent will always greedy (not eksplorasi)
- Belajar rute pinggir cliff because that **optimal teoritis**
- **Masalah**: If still ada epsilon exploration, agent kadang jatuh!

#### SARSA (On-Policy)
- Aware bahwa agent kadang eksplorasi (random action)
- Belajar rute aman di tengah because **safer di praktek**
- **Keuntungan**: More robust terhadap noise and eksplorasi

---

## Konvergensi

### Teorema Konvergensi SARSA

SARSA konvergen to optimal policy $Q^*$ if:

1. Semua state-action pairs dikunjungi infinitely often
2. Policy converges in limit to greedy policy (epsilon → 0)
3. Learning rate $\alpha_t$ memenuhi:
   - $\sum_{t=1}^{\infty} \alpha_t = \infty$ (not konvergen terlalu cepat)
   - $\sum_{t=1}^{\infty} \alpha_t^2 < \infty$ (not diverge)

Praktisnya: Gunakan epsilon decay and learning rate constant kecil (0.1).

---

## Parameter Tuning

### Learning Rate (α)

- **Terlalu besar** (0.5-1.0): Update drastis, unstable, bisa oscillate
- **Terlalu kecil** (0.001-0.01): Belajar lambat sekali
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

- **Start high** (1.0): Eksplorasi penuh di awal
- **Decay gradually** (0.995 per episode): Reduce exploration
- **End low** (0.01): Masih sedikit exploration for robustness

```python
agent = SARSAAgent(env, 
                  epsilon=1.0,
                  epsilon_decay=0.995,
                  epsilon_min=0.01)
```

---

## SARSA vs Q-Learning: Kapan Gunakan Mana?

### Gunakan SARSA ketika:

✅ **Environment berbahaya** - Ada state with penalty besar
✅ **Eksplorasi penting** - Perlu balance exploration vs exploitation
✅ **Konservatif more good** - Prefer safe but slower learning
✅ **Real-world applications** - Agent will deploy with exploration

**Contoh**: Robotics (jatuh = bahaya), autonomous driving, medical treatment

### Gunakan Q-Learning ketika:

✅ **Safe environment** - No catastrophic failures
✅ **Speed matters** - Butuh konvergensi cepat
✅ **Optimal policy jelas** - Not much ambiguity
✅ **Offline learning** - Training di simulator, deploy greedy policy

**Contoh**: Game AI, recommendation systems, trading algorithms

---

## Implementation from Nol

See `implementation.py` for kode lengkap. Fungsi utama:

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

## Contoh: Perbandingan SARSA vs Q-Learning

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

**Observasi**:
- Kedua algoritma converge to policy that good
- Q-Learning **mungkin** learns sedikit more cepat
- SARSA **mungkin** more stabil (less variance)
- Perbedaan tergantung environment (safe vs risky)

---

## Varian SARSA

### Expected SARSA

Update with **expected value** over actions (weighted by policy):

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \sum_{a'} \pi(a'|s') Q(s', a') - Q(s, a) \right]$$

- More stable daripada SARSA
- Mengurangi variance from random action selection
- Masih on-policy

### SARSA(λ)

Menggunakan **eligibility traces** for multi-step updates:

- Eligibility trace $e_t(s,a)$ melacak "eligibility" each state-action for update
- Update many state-action pairs sekaligus based on TD error
- More sample-efficient
- Parameter $\lambda \in [0, 1]$:
  - $\lambda = 0$: SARSA biasa (1-step)
  - $\lambda = 1$: Monte Carlo (full episode)

---

## Advantages and Disadvantages

### ✅ Advantages SARSA

1. **On-policy learning** - Belajar from behavior that actually used
2. **Konservatif & safe** - More careful di risky environments
3. **Robust terhadap exploration** - Q-values reflect actual behavior
4. **Converges under reasonable conditions**

### ❌ Disadvantages SARSA

1. **Belajar more lambat** daripada Q-Learning di many kasus
2. **Terikat at behavior policy** - Harus decay epsilon for konvergen to optimal
3. **Variance tinggi** because random action selection di a'
4. **Perlu tuning epsilon decay** with hati-hati

---

## Advanced: On-Policy vs Off-Policy Deep Dive

### On-Policy (SARSA)

**Policy that dipelajari = Policy that used**

```
Behavior Policy:  ε-greedy based on Q
Target Policy:    ε-greedy based on Q (same!)
```

**Implikasi**:
- Agent learns value from policy that actually used (termasuk eksplorasi)
- Konservatif: aware with risk from exploratory actions

### Off-Policy (Q-Learning)

**Policy that dipelajari ≠ Policy that used**

```
Behavior Policy:  ε-greedy (for exploration)
Target Policy:    Greedy (optimal, no exploration)
```

**Implikasi**:
- Agent learns value from **optimal policy** (greedy)
- Exploratory actions for learning saja, not masuk to Q-values
- More optimistic tentang future rewards

---

## Referensi

- **Paper asli**: Rummery, G. A., & Niranjan, M. (1994). "On-line Q-learning using connectionist systems."
- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (Chapter 6.4 - SARSA)
- **Comparison**: Cliff-walking example from Sutton & Barto

---

## File Terkait

- `implementation.py` - Implementation SARSA from nol + comparison with Q-Learning
- `exercise.ipynb` - Interactive exercises SARSA vs Q-Learning
- `test_sarsa.py` - Unit tests
- `../01_q_learning/` - Q-Learning implementation for perbandingan

**Selamat learns! 🎓**