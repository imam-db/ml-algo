# Q-Learning

**🏠 [Back to Main](../../README.md)** | **📁 [Reinforcement Learning](../README.md)**

## 🗺️ Quick Access

| File | Description | Run |
|------|-----------|----------|
| [`implementation.py`](./implementation.py) | Implementation from nol | `python3 implementation.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Interactive exercises | `jupyter lab exercise.ipynb` |

## 📖 Theory and Concepts

### What is Q-Learning?

**Q-Learning** is algoritma reinforcement learning that learns kebijakan optimal for mengambil actions in an environment with cara mencoba and learns from pengalaman. Algoritma this not memerlukan environment model (model-free) and can learns kebijakan optimal melalui eksplorasi.

Bayangkan seorang SPG (Sales Promoter Girl) that must mengunjungi outlet-outlet di area grid for mempromosikan produk. Dia perlu learns rute terbaik for mengunjungi semua outlet sambil menghindari obstacles, with minimizing steps that taken.

### Formulasi Matematis

**Fungsi Q (Q-function):** Mengestimasi nilai actions that taken at suatu state:

```
Q(s, a) = nilai that diharapkan from mengambil actions a at state s
```

**Update Rule Q-Learning:**

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                              a'
```

Di mana:
- `s` = state (posisi) saat this
- `a` = action (actions) that taken
- `s'` = state berikutnya setelah actions
- `r` = reward that diterima
- `α` (alpha) = learning rate (0 < α ≤ 1)
- `γ` (gamma) = discount factor (0 ≤ γ ≤ 1)
- `max Q(s',a')` = nilai Q maksimum for state berikutnya

### Komponen Utama

#### 1. **State (Keadaan)**
Posisi agent dalam grid world. Dalam contoh SPG:
- State = koordinat (baris, kolom) dalam grid
- Total states = grid_size × grid_size

#### 2. **Action (Tindakan)**
Aksi that dapat dilakukan agent:
- 0: Naik (↑)
- 1: Kanan (→)
- 2: Turun (↓)
- 3: Kiri (←)

#### 3. **Reward (Hadiah)**
Umpan balik numerik from lingkungan:
- **+10**: Mengunjungi outlet baru
- **+50**: Bonus mengunjungi semua outlet
- **-10**: Menabrak obstacles
- **-0.1**: Penalti kecil each steps (mendorong efisiensi)
- **0**: Mengunjungi outlet that sudah dikunjungi

#### 4. **Policy (Kebijakan)**
Strategi for memilih actions:
- **Epsilon-greedy**: Seimbangkan eksplorasi vs eksploitasi
  - Dengan probabilitas ε: pilih actions random (eksplorasi)
  - Dengan probabilitas (1-ε): pilih actions terbaik (eksploitasi)
  - ε menurun seiring waktu (epsilon decay)

#### 5. **Q-Table**
Tabel that menyimpan nilai Q for each pasangan (state, action):
- Dimensi: [n_states × n_actions]
- Diinisialisasi with nilai 0
- Diperbarui selama training

### Algoritma Q-Learning

1. **Inisialisasi:**
   - Buat Q-table with nilai 0
   - Set parameter (α, γ, ε)

2. **Untuk each episode:**
   - Reset lingkungan ke state awal
   - Untuk each steps dalam episode:
     - Pilih action menggunakan epsilon-greedy
     - Eksekusi action, dapatkan reward and next state
     - Update Q-value menggunakan Q-learning rule
     - Update state
     - Jika done, akhiri episode
   - Kurangi epsilon (decay)

3. **Evaluasi:**
   - Gunakan kebijakan learned (ε = 0)
   - Ukur performa

### Exploration vs Exploitation

**Dilema fundamental dalam RL:**

- **Exploration (Eksplorasi):**
  - Mencoba actions baru for menemukan strategi that more good
  - Penting di awal training
  - Dikendalikan by epsilon (ε)

- **Exploitation (Eksploitasi):**
  - Menggunakan pengetahuan that sudah dipelajari
  - Penting saat evaluasi
  - Memilih actions with Q-value tertinggi

**Epsilon-Greedy Strategy:**
```python
if random() < epsilon:
    action = random_action()  # Eksplorasi
else:
    action = argmax(Q[state])  # Eksploitasi
```

**Epsilon Decay:**
- Mulai with ε = 1.0 (100% eksplorasi)
- Turunkan bertahap: ε = ε × decay_rate
- Minimum: ε_min = 0.01 (1% eksplorasi)

### Hyperparameter

#### Learning Rate (α)
Seberapa many Q-value baru menggantikan that lama:
- **α = 0**: Tidak learns (Q-value not berubah)
- **α = 1**: Hanya pakai informasi terbaru
- **Typical: α = 0.1 - 0.3**
- α besar: learns cepat, tapi kurang stabil
- α kecil: learns lambat, more stabil

#### Discount Factor (γ)
Seberapa penting reward masa depan:
- **γ = 0**: Hanya peduli reward langsung
- **γ = 1**: Reward masa depan sama penting
- **Typical: γ = 0.9 - 0.99**
- γ besar: pertimbangkan long-term rewards
- γ kecil: focuses at short-term rewards

#### Epsilon (ε)
Tingkat eksplorasi:
- **ε = 1.0**: 100% random (full eksplorasi)
- **ε = 0.0**: 100% greedy (full eksploitasi)
- **Decay rate: 0.99 - 0.999**
- **Minimum: 0.01**

## 🎯 When to Use Q-Learning?

### ✅ Good for:
- **Discrete state and action spaces**
  - Grid worlds, game sederhana
- **Model-free learning**
  - Tidak perlu tahu dinamika lingkungan
- **Off-policy learning**
  - Bisa learns from pengalaman orang lain
- **Tabular RL problems**
  - State space not terlalu besar
- **Optimal control problems**
  - Mencari kebijakan optimal

### ❌ Less suitable for:
- **Continuous state/action spaces**
  - Gunakan Deep Q-Networks (DQN)
- **Very large state spaces**
  - Q-table jadi terlalu besar
  - Gunakan function approximation
- **Partial observability**
  - Gunakan POMDP solvers
- **Multi-agent scenarios**
  - Perlu algoritma multi-agent
- **Real-time requirements**
  - Training bisa lama

## 💡 Contoh Implementation

### Grid World SPG

Environment that used dalam implementasi:

```
Grid 5×5 with:
- S: Start (posisi awal SPG)
- O: Outlet (tujuan, reward +10)
- X: Obstacle (obstacles, penalty -10)
- SPG: Posisi agent saat this
```

**Contoh Grid:**
```
S  .  .  O  .
.  X  .  .  .
.  .  O  X  .
O  .  .  .  X
.  .  .  .  .
```

**Tujuan:** SPG must mengunjungi semua outlet (O) sambil menghindari obstacles (X) with jalur seefisien mungkin.

### Hasil Training

Setelah training 1000 episode:
- **Q-table konvergen** menunjukkan actions optimal
- **Reward meningkat** from random ke optimal
- **Steps menurun** seiring agent learns jalur efisien
- **Success rate tinggi** (>90%)

**Learning Curve menunjukkan:**
1. **Fase Eksplorasi** (episode 0-200):
   - Reward volatile
   - Epsilon tinggi
   - Agent mencoba berbagai strategi

2. **Fase Transisi** (episode 200-500):
   - Reward mulai stabil
   - Epsilon menurun
   - Agent menemukan strategi that good

3. **Fase Eksploitasi** (episode 500-1000):
   - Reward stabil and tinggi
   - Epsilon rendah
   - Agent konsisten menggunakan strategi optimal

### Perbandingan with Random Policy

| Metric | Q-Learning | Random | Improvement |
|--------|-----------|--------|-------------|
| Mean Reward | ~35-45 | ~-50 | +170% |
| Mean Steps | ~15-20 | ~50-100 | -70% |
| Success Rate | 90-100% | <10% | +90 pts |

## 🔬 Eksperimen and Variasi

### 1. Grid Size
- **Small (3×3)**: Belajar cepat, tapi terbatas
- **Medium (5×5)**: Balance antara kompleksitas and speed
- **Large (10×10)**: More challenging, training more lama

### 2. Number of Outlets/Obstacles
- More many outlet: task more sulit, reward more besar
- More many obstacle: perlu strategi more hati-hati

### 3. Reward Shaping
Coba variasi reward:
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

## 📊 Metrik Evaluasi

### 1. Episode Reward
- Total reward per episode
- Mengukur kualitas kebijakan
- Target: maksimalisasi reward

### 2. Steps per Episode
- Jumlah steps hingga selesai
- Mengukur efisiensi
- Target: minimalisasi steps

### 3. Success Rate
- Persentase episode that berhasil
- Mengukur reliability
- Target: mendekati 100%

### 4. Convergence Speed
- Episode hingga konvergen
- Mengukur learning efficiency
- More cepat = more good

### 5. Q-Value Stability
- Variance Q-values over time
- Mengukur stabilitas learning
- Rendah = stabil

## 🔗 Hubungan with Algoritma Lain

### Q-Learning vs SARSA
- **Q-Learning**: Off-policy (learns optimal policy)
- **SARSA**: On-policy (learns policy that used)
- Q-Learning more agresif, SARSA more aman

### Q-Learning vs Deep Q-Networks (DQN)
- **Q-Learning**: Tabular (Q-table)
- **DQN**: Neural network approximation
- DQN for state space besar

### Q-Learning vs Policy Gradient
- **Q-Learning**: Value-based (learns Q-function)
- **Policy Gradient**: Policy-based (learns policy langsung)
- Q-Learning more sample-efficient for discrete actions

## 📚 Learning Resources

- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [OpenAI Spinning Up - Q-Learning](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [UC Berkeley CS188 - Reinforcement Learning](https://inst.eecs.berkeley.edu/~cs188/)

## 🎓 Exercises

See [`exercise.ipynb`](./exercise.ipynb) for hands-on practice:
1. Eksperimen with hyperparameters
2. Modifikasi reward structure
3. Buat environment custom
4. Visualisasi learned policy
5. Bandingkan with algoritma lain

## 💻 Run Kode

```bash
# Run implementasi from nol
python3 implementation.py

# Buka notebook latihan
jupyter lab exercise.ipynb
```

---

**Happy Learning Q-Learning! 🎮🤖**