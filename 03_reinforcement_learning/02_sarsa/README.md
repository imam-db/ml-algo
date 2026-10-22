# SARSA (State-Action-Reward-State-Action)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## What is SARSA?

SARSA adalah algoritma **on-policy** Temporal Difference (TD) control untuk reinforcement learning. Nama SARSA berasal dari tuple yang digunakan untuk update: **(S, A, R, S', A')**.

### Perbedaan Fundamental: On-Policy vs Off-Policy

| Aspek | SARSA (On-Policy) | Q-Learning (Off-Policy) |
|-------|------------------|------------------------|
| **Belajar dari** | Aksi yang **benar-benar diambil** | Aksi **optimal teoritis** |
| **Update rule** | Q(s,a) ← Q(s,a) + α[r + γ·Q(s',**a'**) - Q(s,a)] | Q(s,a) ← Q(s,a) + α[r + γ·**max** Q(s',a') - Q(s,a)] |
| **a' adalah** | Aksi yang dipilih agent (bisa eksplorasi) | Aksi terbaik (greedy) |
| **Policy** | Learns **behavior policy** (yang digunakan) | Learns **target policy** (optimal) |
| **Konservatif?** | ✅ Ya, lebih hati-hati | ❌ Lebih agresif |
| **Good for** | Environment berbahaya, eksplorasi penting | Environment aman, konvergensi cepat |

---

## Analogi SPG (Sales Promotion Girl/Guy)

Bayangkan dua SPG belajar rute terbaik mengunjungi outlet:

### SPG SARSA (On-Policy)
- **Belajar dari pengalaman nyata**: "Kemarin aku ambil jalan shortcuts melewati pasar, ternyata macet. Aku catat itu buruk."
- **Mempertimbangkan eksplorasi**: Jika kadang masih coba-coba rute baru, Q-values mencerminkan hasil coba-coba itu juga
- **Lebih konservatif**: Tidak assume dia akan selalu pilih rute optimal, jadi lebih hati-hati

### SPG Q-Learning (Off-Policy)
- **Belajar dari yang ideal**: "Seandainya aku selalu pilih rute terbaik, hasilnya akan begini"
- **Mengabaikan eksplorasi**: Meski kadang coba rute baru, Q-values hanya fokus pada rute optimal
- **Lebih optimis**: Assume dia akan selalu greedy di masa depan

**Hasil**:
- Di environment **berbahaya** (ada lubang, jalan rusak): SARSA lebih aman karena aware dengan risikonya
- Di environment **aman**: Q-Learning belajar lebih cepat karena langsung fokus ke optimal

---

## Update Rule SARSA

### Formula

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot Q(s', a') - Q(s, a) \right]$$

Di mana:
- $s$ = state sekarang
- $a$ = action yang diambil
- $r$ = reward yang diterima
- $s'$ = state berikutnya
- **$a'$ = action yang BENAR-BENAR akan diambil di $s'$** (bisa eksplorasi!)
- $\alpha$ = learning rate
- $\gamma$ = discount factor

### Perbedaan dengan Q-Learning

**Q-Learning**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]$$

Perhatikan:
- Q-Learning: $\max_{a'} Q(s', a')$ → selalu ambil yang terbaik (greedy)
- SARSA: $Q(s', a')$ → ambil aksi yang benar-benar dipilih (bisa exploratory!)

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

**Kunci**: Step c.ii — kita **pilih dulu a'** sebelum update, dan gunakan Q(s', a') yang sesuai dengan aksi tersebut.

---

## Kenapa SARSA Lebih Konservatif?

### Contoh: Cliff Walking

Bayangkan grid-world dengan cliff (jurang):

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
- Assume agent akan selalu greedy (tidak eksplorasi)
- Belajar rute pinggir cliff karena itu **optimal teoritis**
- **Masalah**: Jika masih ada epsilon exploration, agent kadang jatuh!

#### SARSA (On-Policy)
- Aware bahwa agent kadang eksplorasi (random action)
- Belajar rute aman di tengah karena **safer di praktek**
- **Keuntungan**: Lebih robust terhadap noise dan eksplorasi

---

## Konvergensi

### Teorema Konvergensi SARSA

SARSA konvergen ke optimal policy $Q^*$ jika:

1. Semua state-action pairs dikunjungi infinitely often
2. Policy converges dalam limit ke greedy policy (epsilon → 0)
3. Learning rate $\alpha_t$ memenuhi:
   - $\sum_{t=1}^{\infty} \alpha_t = \infty$ (tidak konvergen terlalu cepat)
   - $\sum_{t=1}^{\infty} \alpha_t^2 < \infty$ (tidak diverge)

Praktisnya: Gunakan epsilon decay dan learning rate constant kecil (0.1).

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

- **γ = 0**: Myopic (hanya peduli immediate reward)
- **γ = 0.9-0.95**: Balance (prefer sooner rewards)
- **γ = 0.99**: Far-sighted (value future rewards almost equally)

```python
agent = SARSAAgent(env, discount_factor=0.95)
```

### Epsilon Decay

- **Start high** (1.0): Eksplorasi penuh di awal
- **Decay gradually** (0.995 per episode): Reduce exploration
- **End low** (0.01): Masih sedikit exploration untuk robustness

```python
agent = SARSAAgent(env, 
                  epsilon=1.0,
                  epsilon_decay=0.995,
                  epsilon_min=0.01)
```

---

## SARSA vs Q-Learning: Kapan Gunakan Mana?

### Gunakan SARSA ketika:

✅ **Environment berbahaya** - Ada state dengan penalty besar
✅ **Eksplorasi penting** - Perlu balance exploration vs exploitation
✅ **Konservatif lebih baik** - Prefer safe but slower learning
✅ **Real-world applications** - Agent akan deploy dengan exploration

**Contoh**: Robotics (jatuh = bahaya), autonomous driving, medical treatment

### Gunakan Q-Learning ketika:

✅ **Environment aman** - Tidak ada catastrophic failures
✅ **Speed matters** - Butuh konvergensi cepat
✅ **Optimal policy jelas** - Tidak banyak ambiguity
✅ **Offline learning** - Training di simulator, deploy greedy policy

**Contoh**: Game AI, recommendation systems, trading algorithms

---

## Implementation dari Nol

See `implementation.py` untuk kode lengkap. Fungsi utama:

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

# Environment yang sama
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
- Kedua algoritma konvergen ke policy yang baik
- Q-Learning **mungkin** belajar sedikit lebih cepat
- SARSA **mungkin** lebih stabil (less variance)
- Perbedaan tergantung environment (safe vs risky)

---

## Varian SARSA

### Expected SARSA

Update dengan **expected value** over actions (weighted by policy):

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \sum_{a'} \pi(a'|s') Q(s', a') - Q(s, a) \right]$$

- Lebih stabil daripada SARSA
- Mengurangi variance dari random action selection
- Masih on-policy

### SARSA(λ)

Menggunakan **eligibility traces** untuk multi-step updates:

- Eligibility trace $e_t(s,a)$ melacak "eligibility" setiap state-action untuk update
- Update banyak state-action pairs sekaligus berdasarkan TD error
- Lebih sample-efficient
- Parameter $\lambda \in [0, 1]$:
  - $\lambda = 0$: SARSA biasa (1-step)
  - $\lambda = 1$: Monte Carlo (full episode)

---

## Advantages and Disadvantages

### ✅ Advantages SARSA

1. **On-policy learning** - Belajar dari behavior yang benar-benar digunakan
2. **Konservatif & safe** - Lebih hati-hati di risky environments
3. **Robust terhadap exploration** - Q-values reflect actual behavior
4. **Converges under reasonable conditions**

### ❌ Disadvantages SARSA

1. **Belajar lebih lambat** daripada Q-Learning di banyak kasus
2. **Terikat pada behavior policy** - Harus decay epsilon untuk konvergen ke optimal
3. **Variance tinggi** karena random action selection di a'
4. **Perlu tuning epsilon decay** dengan hati-hati

---

## Advanced: On-Policy vs Off-Policy Deep Dive

### On-Policy (SARSA)

**Policy yang dipelajari = Policy yang digunakan**

```
Behavior Policy:  ε-greedy based on Q
Target Policy:    ε-greedy based on Q (same!)
```

**Implikasi**:
- Agent belajar value dari policy yang benar-benar digunakan (termasuk eksplorasi)
- Konservatif: aware dengan risk dari exploratory actions

### Off-Policy (Q-Learning)

**Policy yang dipelajari ≠ Policy yang digunakan**

```
Behavior Policy:  ε-greedy (for exploration)
Target Policy:    Greedy (optimal, no exploration)
```

**Implikasi**:
- Agent belajar value dari **optimal policy** (greedy)
- Exploratory actions untuk learning saja, tidak masuk ke Q-values
- Lebih optimis tentang future rewards

---

## Referensi

- **Paper asli**: Rummery, G. A., & Niranjan, M. (1994). "On-line Q-learning using connectionist systems."
- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (Chapter 6.4 - SARSA)
- **Comparison**: Cliff-walking example dari Sutton & Barto

---

## File Terkait

- `implementation.py` - Implementation SARSA dari nol + comparison dengan Q-Learning
- `exercise.ipynb` - Interactive exercises SARSA vs Q-Learning
- `test_sarsa.py` - Unit tests
- `../01_q_learning/` - Q-Learning implementation untuk perbandingan

**Selamat belajar! 🎓**