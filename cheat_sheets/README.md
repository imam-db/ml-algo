# ML Cheat Sheets Collection

A comprehensive collection of machine learning cheat sheets and an interactive generator tool for creating customized reference materials.

## 📚 Available Cheat Sheets

### Core ML Topics
- **Algorithms**: [Algorithm Selection Guide](algorithms/algorithm_selection_flowchart.md) | [Quick References](algorithms/)
- **Python/ML Tools**: [Scikit-learn](python_sklearn/scikit_learn_quick_reference.md) | [Pandas & NumPy](python_sklearn/pandas_numpy_essentials.md)
- **Model Evaluation**: [Classification Metrics](model_evaluation/classification_metrics.md) | [Regression Metrics](model_evaluation/regression_metrics.md)
- **Math & Statistics**: [Linear Algebra](math_statistics/linear_algebra_essentials.md) | [Statistics](math_statistics/statistics_essentials.md)
- **Data Preprocessing**: [Pipeline Guide](templates/data_preprocessing_pipeline.md)
- **Visualization**: [Matplotlib](visualization/matplotlib_quick_reference.md) | [Seaborn](visualization/seaborn_quick_reference.md) | [Plotly](visualization/plotly_quick_reference.md)
- **Troubleshooting**: [Common Issues & Solutions](troubleshooting/troubleshooting_guide.md)

## 🚀 Interactive Cheat Sheet Generator

Generate customized cheat sheets tailored to your specific needs!

### Features
- **Topic Selection**: Choose from 40+ ML topics across 6 categories
- **Custom Combinations**: Combine multiple topics into a single cheat sheet
- **Template-Based**: Pre-built templates with examples and best practices
- **File Management**: View, open, and manage generated cheat sheets
- **Cross-Platform**: Works on Windows, macOS, and Linux

### Quick Start

```bash
# Navigate to cheat sheets directory
cd cheat_sheets/

# Run the interactive generator
python interactive_generator.py
```

### Usage Options

1. **Category-Based Generation**
   - Select a category (Algorithms, Preprocessing, etc.)
   - Choose specific topics within that category
   - Generate focused cheat sheets

2. **Custom Cheat Sheet**
   - Mix and match topics from any category
   - Create personalized reference guides
   - Perfect for specific projects or learning paths

3. **File Management**
   - View all generated cheat sheets
   - Open files directly from the tool
   - Delete outdated reference materials

### Available Categories

| Category | Topics | Description |
|----------|--------|-------------|
| **Algorithms** | 10 topics | Linear/Logistic Regression, Trees, SVM, Neural Networks, etc. |
| **Preprocessing** | 6 topics | Data Cleaning, Feature Engineering, Scaling, Missing Data, etc. |
| **Evaluation** | 5 topics | Metrics, Cross Validation, Model Selection, Hyperparameter Tuning |
| **Visualization** | 4 topics | Matplotlib, Seaborn, Plotly, Exploratory Data Analysis |
| **Tools** | 4 topics | Scikit-learn, Pandas, NumPy, Jupyter Notebooks |
| **Math** | 4 topics | Linear Algebra, Statistics, Probability, Calculus |

### Example Generated Content

Each cheat sheet includes:
- **Quick Reference**: Key concepts and formulas
- **Code Examples**: Ready-to-use implementations
- **Best Practices**: Tips and common pitfalls
- **Additional Resources**: Links to documentation and tutorials

```markdown
# Linear Regression Cheat Sheet

> Quick reference for linear regression implementation and theory

## Key Concepts
- **Linear Regression Equation**: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
- **Assumptions**: Linearity, Independence, Homoscedasticity, Normality
- **Cost Function**: MSE = (1/n) Σ(yᵢ - ŷᵢ)²

## Implementation
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
```

## 📁 Directory Structure

```
cheat_sheets/
├── algorithms/                 # Algorithm-specific cheat sheets
├── python_sklearn/            # Python tools and libraries
├── model_evaluation/          # Evaluation metrics and techniques  
├── math_statistics/           # Mathematical foundations
├── visualization/             # Plotting and visualization
├── troubleshooting/           # Common issues and solutions
├── templates/                 # Template files and workflows
├── generated/                 # Generated custom cheat sheets
├── interactive_generator.py   # Main generator tool
└── README.md                  # This file
```

## 🛠️ Requirements

- Python 3.6+
- Standard library modules only (no additional dependencies)

## 🎯 Use Cases

### For Students
- **Study Guides**: Generate focused cheat sheets for exam preparation
- **Assignment Help**: Quick reference for specific algorithms or techniques
- **Concept Review**: Combine related topics for comprehensive understanding

### For Practitioners  
- **Project Reference**: Create cheat sheets tailored to your current project
- **Team Onboarding**: Generate guides for new team members
- **Interview Prep**: Focused review materials for technical interviews

### For Educators
- **Teaching Materials**: Generate handouts for specific lessons
- **Workshop Guides**: Create reference materials for training sessions
- **Curriculum Support**: Supplementary materials aligned with course content

## 📝 Customization

### Adding New Templates
1. Edit `interactive_generator.py`
2. Add new topics to the `categories` dictionary
3. Create corresponding templates in the `templates` dictionary
4. Include sections with content arrays

### Template Structure
```python
"topic_key": {
    "title": "Topic Cheat Sheet",
    "description": "Brief description of the topic",
    "sections": [
        {
            "name": "Section Name",
            "content": [
                "Line 1 of content",
                "Line 2 of content",
                "```python",
                "code_example()",
                "```"
            ]
        }
    ]
}
```

## 🤝 Contributing

We welcome contributions to expand the cheat sheet collection!

### How to Contribute
1. **Add New Topics**: Extend the template collection
2. **Improve Existing Content**: Enhance current cheat sheets
3. **Report Issues**: Help us identify and fix problems
4. **Suggest Features**: Propose new generator capabilities

### Guidelines
- Follow the existing template structure
- Include practical code examples
- Add clear explanations and best practices
- Test generated output for accuracy

## 📄 License

This project is part of the ML Algorithms Learning repository. See the main repository for license information.

## 🔗 Related Resources

- [Main ML Repository](../README.md)
- [Interactive Playground Tools](../playground/)
- [Algorithm Implementations](../algorithms/)
- [Learning Resources](../resources/)

---

*Happy Learning! 🎓*