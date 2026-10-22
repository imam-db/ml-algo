# Contributing Guidelines

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](CONTRIBUTING.id.md)**

Thank you for your interest in contributing to the Machine Learning Algorithms Learning Repository! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Process](#development-process)
- [Code Standards](#code-standards)
- [Bilingual Documentation](#bilingual-documentation)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Adding New Algorithms](#adding-new-algorithms)

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

There are many ways to contribute:

### 🐛 Reporting Bugs

- Use GitHub Issues to report bugs
- Describe the steps to reproduce the bug
- Include environment details (Python version, OS, etc.)
- Attach screenshots if relevant

### 💡 Proposing New Features

- Open an issue with the `enhancement` label
- Explain the use case and benefits
- Discuss with maintainers before major implementations

### 📝 Improving Documentation

- Fix typos, errors, or unclear explanations
- Add examples or tutorials
- Translate documentation (both English and Indonesian required)

### 🎨 Adding Algorithms

See the [Adding New Algorithms](#adding-new-algorithms) section below.

## Development Process

### 1. Fork and Clone Repository

```bash
# Fork repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ml-algo.git
cd ml-algo
```

### 2. Setup Development Environment

```bash
# Install dependencies
pip install -e ".[dev]"

# Or with uv (recommended)
uv sync
```

### 3. Create New Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes

- Write clear and documented code
- Add tests for new features
- Ensure all tests pass

### 5. Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_linear_regression.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### 6. Commit and Push

```bash
git add .
git commit -m "feat: brief description of changes"
git push origin feature/your-feature-name
```

### 7. Create Pull Request

- Go to GitHub and create a Pull Request
- Fill out the PR template completely
- Link related issues
- Wait for review

## Code Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with these tools:

```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .
```

### Code Quality

- **Clear variable names**: Use descriptive names
- **Comments**: Explain non-obvious logic
- **Docstrings**: Document all functions and classes
- **Type hints**: Use type annotations where possible

### Example:

```python
def train_test_split(X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2,
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set (0.0-1.0)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Implementation
    pass
```

## Bilingual Documentation

**🌐 All new modules MUST include both English and Indonesian documentation.**

### Requirements:

1. **README.md** (English)
   - Complete algorithm theory and explanations
   - Usage examples with code
   - Mathematical formulas and concepts
   - Plain language suitable for learners

2. **README.id.md** (Indonesian)  
   - Faithful translation of English version
   - Maintain plain language and learning focus
   - Keep SPG/retail themed examples
   - Use natural Indonesian (avoid literal translations)

3. **Language Switcher**
   - Add at the top of each README: `**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**`
   - Link to the corresponding language version

4. **Internal Links**
   - Update all relative links to work in both languages
   - Test links after creating bilingual versions

### Translation Tips:

- **Plain Language**: Both versions should be beginner-friendly
- **Consistency**: Use consistent terminology across files
- **Examples**: Keep practical examples (SPG/retail context is fine)
- **Formulas**: Mathematical notation stays the same
- **Code**: Keep code examples identical in both languages

### Example Structure:

```
01_supervised_learning/
├── 07_knn/
│   ├── README.md          # English version
│   ├── README.id.md       # Indonesian version
│   ├── implementation.py
│   ├── sklearn_example.py
│   └── exercise.ipynb
```

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semicolons, etc
- `refactor`: Code refactoring
- `test`: Adding or fixing tests
- `chore`: Maintenance tasks

**Examples:**

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
docs(sarsa): add bilingual README

- Created README.md (English)
- Created README.id.md (Indonesian)
- Added language switcher links
```

## Pull Request Process

### PR Template

When opening a PR, fill out this template:

```markdown
## Description
[Describe your changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Other (explain)

## Bilingual Documentation (if applicable)
- [ ] Created/updated README.md (English)
- [ ] Created/updated README.id.md (Indonesian)
- [ ] Added language switcher links
- [ ] Tested all internal links

## Checklist
- [ ] Tests pass locally
- [ ] Added/updated tests for changes
- [ ] Updated documentation (both languages if applicable)
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented hard-to-understand areas

## Related Issues
Closes #[issue number]

## Screenshots (if applicable)
[Add screenshots]
```

### Review Process

1. **Automated Checks**: CI/CD will run tests automatically
2. **Code Review**: Maintainers will review your code
3. **Discussion**: Be open to feedback and questions
4. **Updates**: Make requested changes
5. **Approval**: After approval, your PR will be merged

## Adding New Algorithms

### Required Structure

Each algorithm must include:

```
algorithm_name/
├── README.md              # English documentation
├── README.id.md           # Indonesian documentation  
├── implementation.py      # From-scratch implementation
├── sklearn_example.py     # Comparison with sklearn
├── exercise.ipynb         # Interactive exercises
└── datasets/              # (Optional) specific datasets
```

### Documentation Requirements

Both **README.md** and **README.id.md** must include:

1. **Introduction**
   - What is the algorithm?
   - Plain language explanation
   - SPG/retail analogy (if applicable)

2. **Key Concepts**
   - Core concepts explained simply
   - Diagrams or visualizations (if helpful)
   - When to use this algorithm

3. **Mathematical Foundation**
   - Formulas with explanations
   - Step-by-step algorithm
   - Complexity analysis

4. **Implementation Details**
   - Code examples
   - Parameter tuning guide
   - Common pitfalls

5. **Comparison**
   - vs similar algorithms
   - Pros and cons
   - Use cases

6. **References**
   - Papers, books, or tutorials
   - Sklearn documentation links

### Testing Requirements

- Each algorithm must have comprehensive tests
- Tests must be deterministic (use `random_state` or `seed`)
- Tests should be fast (<1 second per test if possible)
- Use fixtures for reusable setup

### Example Test Structure:

```python
import pytest
import numpy as np
from implementation import YourAlgorithm

class TestYourAlgorithm:
    def test_initialization(self):
        algo = YourAlgorithm()
        assert algo is not None
    
    def test_fit_predict(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        algo = YourAlgorithm()
        algo.fit(X, y)
        predictions = algo.predict(X)
        
        assert predictions.shape == y.shape
```

## Questions?

- 💬 Open a [Discussion](https://github.com/imam-db/ml-algo/discussions)
- 🐛 Report an [Issue](https://github.com/imam-db/ml-algo/issues)
- 📧 Contact maintainers

---

**Thank you for contributing! 🙏**

*Together we make learning machine learning accessible to everyone.*
