# Panduan Kontribusi

**[🇬🇧 English](CONTRIBUTING.md) | 🇮🇩 Bahasa Indonesia**

Terima kasih atas minat Anda untuk berkontribusi pada Machine Learning Algorithms Learning Repository! 🎉

## 📋 Daftar Isi

- [Code of Conduct](#code-of-conduct)
- [Cara Berkontribusi](#cara-berkontribusi)
- [Proses Development](#proses-development)
- [Standar Kode](#standar-kode)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Menambahkan Algoritma Baru](#menambahkan-algoritma-baru)

## Code of Conduct

Proyek ini mengikuti [Code of Conduct](CODE_OF_CONDUCT.md). Dengan berpartisipasi, Anda diharapkan menjunjung kode etik ini.

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
git clone https://github.com/YOUR_USERNAME/ml-algorithms-learning.git
cd ml-algorithms-learning
```

### 2. Setup Development Environment

```bash
# Install dependencies
pip install -e ".[dev]"

# Atau dengan uv (recommended)
uv sync
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
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_linear_regression.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
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

Kami mengikuti [PEP 8](https://pep8.org/) dengan beberapa penyesuaian:

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
    ├── README.md              # Teori dan penjelasan (Indonesian)
    ├── implementation.py      # Implementasi dari nol
    ├── sklearn_example.py     # Contoh menggunakan scikit-learn (jika applicable)
    ├── exercise.ipynb         # Latihan interaktif
    ├── datasets/              # Dataset khusus (jika ada)
    │   └── README.md
    └── tests/                 # Tests (atau di tests/ root)
```

### Template README.md

```markdown
# Nama Algoritma

**🏠 [Kembali ke Menu Utama](../../README.md)** | **📁 [Kategori](../README.md)**

## 🗺️ Akses Cepat
[Quick access table]

## 📖 Teori dan Konsep
[Penjelasan konsep dalam Bahasa Indonesia]

## 🎯 Kapan Menggunakan?
[Use cases]

## 💻 Implementasi
[Code examples]

## 📊 Evaluasi
[Metrics and evaluation]

## 📚 Sumber Belajar
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

- [ ] README.md lengkap (Indonesian)
- [ ] implementation.py dengan docstrings
- [ ] exercise.ipynb dengan latihan interaktif
- [ ] Tests di `tests/test_nama_algoritma.py`
- [ ] Contoh dataset (jika diperlukan)
- [ ] Update main README.md
- [ ] All tests pass
- [ ] Code reviewed

## Pertanyaan?

Jika ada pertanyaan:
1. Cek [Issues](https://github.com/YOUR_USERNAME/ml-algorithms-learning/issues) yang sudah ada
2. Buka issue baru dengan label `question`
3. Join diskusi di GitHub Discussions (jika tersedia)

---

**Terima kasih atas kontribusi Anda! 🙏**
