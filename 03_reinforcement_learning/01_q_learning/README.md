# Q-Learning

**🏠 [Back to Main](../../README.md)** | **📁 [Reinforcement Learning](../README.md)**

## 🗺️ Quick Access

| File | Description | Run |
|------|-----------|----------|
| [`implementation.py`](./implementation.py) | Implementation dari nol | `python3 implementation.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Interactive exercises | `jupyter lab exercise.ipynb` |

## 📖 Theory and Concepts

### What is Q-Learning?

**Q-Learning** adalah algoritma reinforcement learning yang mempelajari kebijakan optimal untuk mengambil tindakan dalam suatu lingkungan dengan cara mencoba dan belajar dari pengalaman. Algoritma ini tidak memerlukan model lingkungan (model-free) dan dapat belajar kebijakan optimal melalui eksplorasi.

Bayangkan seorang SPG (Sales Promoter Girl) yang harus mengunjungi outlet-outlet di area grid untuk mempromosikan produk. Dia perlu belajar rute terbaik untuk mengunjungi semua outlet sambil menghindari hambatan, dengan meminimalkan langkah yang diambil.

### Formulasi Matematis

**Fungsi Q (Q-function):** Mengestimasi nilai tindakan yang diambil pada suatu state:

```
Q(s, a) = nilai yang diharapkan dari mengambil tindakan a pada state s
```

**Update Rule Q-Learning:**

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                              a'
```

Di mana:
- `s` = state (posisi) saat ini
- `a` = action (tindakan) yang diambil
- `s'` = state berikutnya setelah tindakan
- `r` = reward yang diterima
- `α` (alpha) = learning rate (0 < α ≤ 1)
- `γ` (gamma) = discount factor (0 ≤ γ ≤ 1)
- `max Q(s',a')` = nilai Q maksimum untuk state berikutnya

### Komponen Utama

#### 1. **State (Keadaan)**
Posisi agent dalam grid world. Dalam contoh SPG:
- State = koordinat (baris, kolom) dalam grid
- Total states = grid_size × grid_size

#### 2. **Action (Tindakan)**
Aksi yang dapat dilakukan agent:
- 0: Naik (↑)
- 1: Kanan (→)
- 2: Turun (↓)
- 3: Kiri (←)

#### 3. **Reward (Hadiah)**
Umpan balik numerik dari lingkungan:
- **+10**: Mengunjungi outlet baru
- **+50**: Bonus mengunjungi semua outlet
- **-10**: Menabrak hambatan
- **-0.1**: Penalti kecil setiap langkah (mendorong efisiensi)
- **0**: Mengunjungi outlet yang sudah dikunjungi

#### 4. **Policy (Kebijakan)**
Strategi untuk memilih tindakan:
- **Epsilon-greedy**: Seimbangkan eksplorasi vs eksploitasi
  - Dengan probabilitas ε: pilih tindakan random (eksplorasi)
  - Dengan probabilitas (1-ε): pilih tindakan terbaik (eksploitasi)
  - ε menurun seiring waktu (epsilon decay)

#### 5. **Q-Table**
Tabel yang menyimpan nilai Q untuk setiap pasangan (state, action):
- Dimensi: [n_states × n_actions]
- Diinisialisasi dengan nilai 0
- Diperbarui selama training

### Algoritma Q-Learning

1. **Inisialisasi:**
   - Buat Q-table dengan nilai 0
   - Set parameter (α, γ, ε)

2. **Untuk setiap episode:**
   - Reset lingkungan ke state awal
   - Untuk setiap langkah dalam episode:
     - Pilih action menggunakan epsilon-greedy
     - Eksekusi action, dapatkan reward dan next state
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
  - Mencoba tindakan baru untuk menemukan strategi yang lebih baik
  - Penting di awal training
  - Dikendalikan oleh epsilon (ε)

- **Exploitation (Eksploitasi):**
  - Menggunakan pengetahuan yang sudah dipelajari
  - Penting saat evaluasi
  - Memilih tindakan dengan Q-value tertinggi

**Epsilon-Greedy Strategy:**
```python
if random() < epsilon:
    action = random_action()  # Eksplorasi
else:
    action = argmax(Q[state])  # Eksploitasi
```

**Epsilon Decay:**
- Mulai dengan ε = 1.0 (100% eksplorasi)
- Turunkan bertahap: ε = ε × decay_rate
- Minimum: ε_min = 0.01 (1% eksplorasi)

### Hyperparameter

#### Learning Rate (α)
Seberapa banyak Q-value baru menggantikan yang lama:
- **α = 0**: Tidak belajar (Q-value tidak berubah)
- **α = 1**: Hanya pakai informasi terbaru
- **Typical: α = 0.1 - 0.3**
- α besar: belajar cepat, tapi kurang stabil
- α kecil: belajar lambat, lebih stabil

#### Discount Factor (γ)
Seberapa penting reward masa depan:
- **γ = 0**: Hanya peduli reward langsung
- **γ = 1**: Reward masa depan sama penting
- **Typical: γ = 0.9 - 0.99**
- γ besar: pertimbangkan long-term rewards
- γ kecil: fokus pada short-term rewards

#### Epsilon (ε)
Tingkat eksplorasi:
- **ε = 1.0**: 100% random (full eksplorasi)
- **ε = 0.0**: 100% greedy (full eksploitasi)
- **Decay rate: 0.99 - 0.999**
- **Minimum: 0.01**

## 🎯 When to Use Q-Learning?

### ✅ Good for:
- **Discrete state dan action spaces**
  - Grid worlds, game sederhana
- **Model-free learning**
  - Tidak perlu tahu dinamika lingkungan
- **Off-policy learning**
  - Bisa belajar dari pengalaman orang lain
- **Tabular RL problems**
  - State space tidak terlalu besar
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

Environment yang digunakan dalam implementasi:

```
Grid 5×5 dengan:
- S: Start (posisi awal SPG)
- O: Outlet (tujuan, reward +10)
- X: Obstacle (hambatan, penalty -10)
- SPG: Posisi agent saat ini
```

**Contoh Grid:**
```
S  .  .  O  .
.  X  .  .  .
.  .  O  X  .
O  .  .  .  X
.  .  .  .  .
```

**Tujuan:** SPG harus mengunjungi semua outlet (O) sambil menghindari hambatan (X) dengan jalur seefisien mungkin.

### Hasil Training

Setelah training 1000 episode:
- **Q-table konvergen** menunjukkan tindakan optimal
- **Reward meningkat** dari random ke optimal
- **Steps menurun** seiring agent belajar jalur efisien
- **Success rate tinggi** (>90%)

**Learning Curve menunjukkan:**
1. **Fase Eksplorasi** (episode 0-200):
   - Reward volatile
   - Epsilon tinggi
   - Agent mencoba berbagai strategi

2. **Fase Transisi** (episode 200-500):
   - Reward mulai stabil
   - Epsilon menurun
   - Agent menemukan strategi yang baik

3. **Fase Eksploitasi** (episode 500-1000):
   - Reward stabil dan tinggi
   - Epsilon rendah
   - Agent konsisten menggunakan strategi optimal

### Perbandingan dengan Random Policy

| Metric | Q-Learning | Random | Improvement |
|--------|-----------|--------|-------------|
| Mean Reward | ~35-45 | ~-50 | +170% |
| Mean Steps | ~15-20 | ~50-100 | -70% |
| Success Rate | 90-100% | <10% | +90 pts |

## 🔬 Eksperimen dan Variasi

### 1. Grid Size
- **Small (3×3)**: Belajar cepat, tapi terbatas
- **Medium (5×5)**: Balance antara kompleksitas dan speed
- **Large (10×10)**: Lebih challenging, training lebih lama

### 2. Number of Outlets/Obstacles
- Lebih banyak outlet: task lebih sulit, reward lebih besar
- Lebih banyak obstacle: perlu strategi lebih hati-hati

### 3. Reward Shaping
Coba variasi reward:
```python
# Dense rewards (lebih banyak feedback)
- Reward kecil untuk mendekati outlet
- Penalty bertingkat untuk obstacle

# Sparse rewards (lebih realistis)
- Hanya reward saat mencapai goal
- Lebih sulit, tapi lebih general
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
- Jumlah langkah hingga selesai
- Mengukur efisiensi
- Target: minimalisasi steps

### 3. Success Rate
- Persentase episode yang berhasil
- Mengukur reliability
- Target: mendekati 100%

### 4. Convergence Speed
- Episode hingga konvergen
- Mengukur learning efficiency
- Lebih cepat = lebih baik

### 5. Q-Value Stability
- Variance Q-values over time
- Mengukur stabilitas learning
- Rendah = stabil

## 🔗 Hubungan dengan Algoritma Lain

### Q-Learning vs SARSA
- **Q-Learning**: Off-policy (belajar optimal policy)
- **SARSA**: On-policy (belajar policy yang digunakan)
- Q-Learning lebih agresif, SARSA lebih aman

### Q-Learning vs Deep Q-Networks (DQN)
- **Q-Learning**: Tabular (Q-table)
- **DQN**: Neural network approximation
- DQN untuk state space besar

### Q-Learning vs Policy Gradient
- **Q-Learning**: Value-based (belajar Q-function)
- **Policy Gradient**: Policy-based (belajar policy langsung)
- Q-Learning lebih sample-efficient untuk discrete actions

## 📚 Learning Resources

- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [OpenAI Spinning Up - Q-Learning](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [UC Berkeley CS188 - Reinforcement Learning](https://inst.eecs.berkeley.edu/~cs188/)

## 🎓 Exercises

See [`exercise.ipynb`](./exercise.ipynb) for hands-on practice:
1. Eksperimen dengan hyperparameters
2. Modifikasi reward structure
3. Buat environment custom
4. Visualisasi learned policy
5. Bandingkan dengan algoritma lain

## 💻 Run Kode

```bash
# Run implementasi dari nol
python3 implementation.py

# Buka notebook latihan
jupyter lab exercise.ipynb
```

---

**Happy Learning Q-Learning! 🎮🤖**