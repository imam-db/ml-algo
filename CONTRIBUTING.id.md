# Panduan Kontribusi

**[🇬🇧 English](CONTRIBUTING.md) | 🇮🇩 Bahasa Indonesia**

Terima kasih atas minat Anda untuk berkontribusi pada Machine Learning Algorithms Learning Repository! 🎉

## 📋 Daftar Isi

- [Code of Conduct](#code-of-conduct)
- [Cara Berkontribusi](#cara-berkontribusi)
- [Proses Development](#proses-development)
- [Standar Kode](#standar-kode)
- [Dokumentasi Bilingual](#dokumentasi-bilingual)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Menambahkan Algoritma Baru](#menambahkan-algoritma-baru)

## Code of Conduct

Proyek ini mengikuti [Code of Conduct](CODE_OF_CONDUCT.id.md). Dengan berpartisipasi, Anda diharapkan menjunjung kode etik ini.

## Cara Berkontribusi

Ada banyak cara untuk berkontribusi:

### 🐛 Melaporkan Bug

- Gunakan GitHub Issues untuk melaporkan bug
- Jelaskan langkah-langkah untuk mereproduksi bug
- Sertakan environment details (Python version, OS, dll)
- Lampirkan screenshot jika relevan

### 💡 Mengusulkan Fitur Baru

- Buka issue dengan label `enhancement`
- Jelaskan use case dan manfaat fitur
- Diskusikan dengan maintainer sebelum implementasi besar

### 📝 Memperbaiki Dokumentasi

- Typo, kesalahan, atau penjelasan yang kurang jelas
- Menambahkan contoh atau tutorial
- Menerjemahkan dokumentasi

### 🎨 Menambahkan Algoritma

Lihat bagian [Menambahkan Algoritma Baru](#menambahkan-algoritma-baru) di bawah.

## Proses Development

### 1. Fork dan Clone Repository

```bash
# Fork repository di GitHub
# Clone fork Anda
git clone https://github.com/YOUR_USERNAME/ml-algo.git
cd ml-algo
```

### 2. Setup Development Environment

```bash
# Install dependencies, termasuk dev tools (uv direkomendasikan)
uv sync --extra dev
```

### 3. Buat Branch Baru

```bash
git checkout -b feature/nama-fitur-anda
```

### 4. Buat Perubahan

- Tulis kode yang jelas dan documented
- Tambahkan tests untuk fitur baru
- Pastikan semua tests pass

### 5. Run Tests

```bash
# Jalankan semua tests
uv run pytest

# Jalankan file test tertentu
uv run pytest tests/test_linear_regression.py -v

# Jalankan dengan coverage
uv run pytest --cov=. --cov-report=html
```

### 6. Commit dan Push

```bash
git add .
git commit -m "feat: deskripsi singkat perubahan"
git push origin feature/nama-fitur-anda
```

### 7. Buat Pull Request

- Buka Pull Request di GitHub
- Isi template PR dengan lengkap
- Link ke issue terkait jika ada

## Standar Kode

### Python Style Guide

Kami mengikuti [PEP 8](https://pep8.org/) dan menggunakan [Ruff](https://github.com/astral-sh/ruff) untuk linting dan formatting. Pemeriksaan ini bersifat opsional/direkomendasikan untuk pengembangan lokal dan **tidak dijalankan sebagai gate di CI**.

```bash
# Cek style dan lint (opsional/direkomendasikan)
uv run ruff check .

# Cek formatting (opsional/direkomendasikan)
uv run ruff format --check

# Perbaiki otomatis masalah lint dan format kode (opsional)
uv run ruff check --fix .
uv run ruff format .
```

### Contoh:

```python
# Good ✅
def train_model(X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01) -> Model:
    """
    Train a machine learning model.
    
    Args:
        X: Training features, shape (n_samples, n_features)
        y: Training labels, shape (n_samples,)
        learning_rate: Learning rate for gradient descent
        
    Returns:
        Trained model instance
    """
    model = Model(learning_rate=learning_rate)
    model.fit(X, y)
    return model

# Bad ❌
def trainModel(x,y,lr=0.01):
    m=Model(lr=lr)
    m.fit(x,y)
    return m
```

### Dokumentasi

- **Docstrings**: Gunakan Google style docstrings
- **Type hints**: Tambahkan type hints untuk fungsi publik
- **Comments**: Jelaskan "why", bukan "what"

### Testing

```python
# Good test ✅
def test_linear_regression_predictions():
    """Test that linear regression produces reasonable predictions."""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(np.array([[5], [6]]))
    expected = np.array([10, 12])
    
    np.testing.assert_allclose(predictions, expected, rtol=0.01)
```

**Test Requirements:**
- Setiap algoritma harus memiliki tests
- Tests harus deterministic (gunakan `random_state` atau `seed`)
- Tests harus cepat (<1 detik per test jika memungkinkan)
- Gunakan fixtures untuk setup yang reusable

## Dokumentasi Bilingual

**🌐 Semua modul baru WAJIB menyertakan dokumentasi dalam Bahasa Inggris dan Bahasa Indonesia.**

### Persyaratan:

1. **README.md** (English)
   - Teori dan penjelasan algoritma secara lengkap
   - Contoh penggunaan dengan kode
   - Rumus dan konsep matematika
   - Bahasa yang mudah dipahami pembelajar

2. **README.id.md** (Indonesian)
   - Terjemahan setia dari versi bahasa Inggris
   - Pertahankan bahasa yang sederhana dan fokus belajar
   - Pertahankan contoh bertema SPG/retail
   - Gunakan Bahasa Indonesia yang natural (hindari terjemahan literal)

3. **Language Switcher**
   - Tambahkan di bagian atas setiap README: `**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**`
   - Tautkan ke versi bahasa yang sesuai

4. **Internal Links**
   - Perbarui semua tautan relatif agar bekerja di kedua bahasa
   - Uji tautan setelah membuat versi bilingual

### Tips Terjemahan:

- **Bahasa Sederhana**: Kedua versi harus ramah pemula
- **Konsistensi**: Gunakan terminologi yang konsisten antar file
- **Contoh**: Pertahankan contoh praktis (konteks SPG/retail tidak masalah)
- **Rumus**: Notasi matematika tetap sama
- **Kode**: Pertahankan contoh kode identik di kedua bahasa

## Commit Messages

Kami menggunakan [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: Fitur baru
- `fix`: Bug fix
- `docs`: Perubahan dokumentasi
- `style`: Formatting, missing semicolons, etc
- `refactor`: Refactoring kode
- `test`: Menambah atau memperbaiki tests
- `chore`: Maintenance tasks

**Contoh:**

```bash
feat(kmeans): implement K-Means++ initialization

Add K-Means++ initialization method to improve convergence speed
and final cluster quality compared to random initialization.

Closes #42
```

```bash
fix(linear-regression): correct gradient calculation

Fixed sign error in gradient descent update that caused
divergence with high learning rates.
```

```bash
docs(readme): add badges and quick start guide

- Add CI/CD status badge
- Add test coverage badge
- Improve quick start instructions
```

## Pull Request Process

### PR Template

Saat membuka PR, isi template berikut:

```markdown
## Description
[Jelaskan perubahan Anda]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Other (explain)

## Checklist
- [ ] Tests pass locally
- [ ] Added/updated tests for changes
- [ ] Updated documentation
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented hard-to-understand areas

## Related Issues
Closes #[issue number]

## Screenshots (if applicable)
[Add screenshots]
```

### Review Process

1. **Automated Checks**: CI/CD akan run tests otomatis
2. **Code Review**: Maintainer akan review kode Anda
3. **Discussion**: Diskusi dan iterasi jika diperlukan
4. **Merge**: PR akan di-merge setelah approved

### Tips untuk PR yang Baik

✅ **Do:**
- Keep PR focused dan atomic
- Write clear description
- Include tests
- Update documentation
- Respond to feedback

❌ **Don't:**
- Mix unrelated changes
- Submit PR tanpa testing
- Ignore feedback
- Force push setelah review started

## Menambahkan Algoritma Baru

### Struktur Folder

Setiap algoritma harus mengikuti struktur ini:

```
01_supervised_learning/
└── XX_nama_algoritma/
    ├── README.md              # Teori dan penjelasan (English)
    ├── README.id.md           # Teori dan penjelasan (Indonesian)
    ├── implementation.py      # Implementasi dari nol
    ├── sklearn_example.py     # Contoh menggunakan scikit-learn (jika applicable)
    ├── exercise.ipynb         # Latihan interaktif
    ├── datasets/              # Dataset khusus (jika ada)
    │   └── README.md          # Dokumentasi dataset
    └── tests/                 # Tests (atau di tests/ root)
```

**Catatan:** Setiap algoritma wajib memiliki `README.md` (English) dan `README.id.md` (Indonesian).

### Template README.md (English)

`README.id.md` adalah versi Bahasa Indonesia dan wajib disertakan.

```markdown
# Algorithm Name

**🏠 [Back to Main Menu](../../README.md)** | **📁 [Category](../README.md)**

## 🗺️ Quick Access
[Quick access table]

## 📖 Theory and Concepts
[Concept explanation in English]

## 🎯 When to Use?
[Use cases]

## 💻 Implementation
[Code examples]

## 📊 Evaluation
[Metrics and evaluation]

## 📚 Learning Resources
[References]
```

### Template implementation.py

```python
"""
Nama Algoritma - Implementation from Scratch

Brief description in English.
"""

import numpy as np
from typing import Optional

class NamaAlgoritma:
    """
    Brief description.
    
    Parameters:
        param1: Description
        param2: Description
    """
    
    def __init__(self, param1: float = 0.01, param2: int = 100):
        self.param1 = param1
        self.param2 = param2
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NamaAlgoritma':
        """
        Train the model.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
            
        Returns:
            self: Trained model instance
        """
        # Implementation
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features, shape (n_samples, n_features)
            
        Returns:
            Predictions, shape (n_samples,)
        """
        # Implementation
        pass

if __name__ == "__main__":
    # Demo
    pass
```

### Checklist untuk Algoritma Baru

- [ ] README.md (English) lengkap
- [ ] README.id.md (Indonesian) lengkap
- [ ] implementation.py dengan docstrings
- [ ] exercise.ipynb dengan latihan interaktif
- [ ] Tests di `tests/test_nama_algoritma.py`
- [ ] Contoh dataset (jika diperlukan)
- [ ] Update README.md (English) dan README.id.md (Indonesian) di root
- [ ] Semua tests pass
- [ ] Code reviewed

## Pertanyaan?

Jika ada pertanyaan:
1. Cek [Issues](https://github.com/YOUR_USERNAME/ml-algo/issues) yang sudah ada
2. Buka issue baru dengan label `question`
3. Join diskusi di GitHub Discussions (jika tersedia)

---

**Terima kasih atas kontribusi Anda! 🙏**
