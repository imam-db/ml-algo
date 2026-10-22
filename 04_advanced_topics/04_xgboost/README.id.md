# XGBoost (eXtreme Gradient Boosting)

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

**🏠 [Back to Main](../../README.md)** | **📁 [Advanced Topics](../README.md)** | **⬅️ [Previous: Model Evaluation](../03_model_evaluation/README.md)**

## 🔑 Key Terms
**Quick definitions for this algorithm** - 📚 [Full Glossary](../../GLOSSARY.md)

- **[XGBoost](../../GLOSSARY.md#boosting)** - eXtreme Gradient Boosting, advanced ensemble method
- **[Ensemble Method](../../GLOSSARY.md#ensemble-method)** - Combining multiple models for better performance
- **[Boosting](../../GLOSSARY.md#boosting)** - Sequential training where each model corrects previous errors
- **[Hyperparameter Tuning](../../GLOSSARY.md#hyperparameter-tuning)** - Finding optimal algorithm configuration settings
- **[Cross-Validation](../../GLOSSARY.md#cross-validation-cv)** - Technique to assess model performance robustly
- **[Feature Importance](../../GLOSSARY.md#feature-importance)** - Measure of each feature's contribution to predictions
- **[SHAP](../../GLOSSARY.md#shap-shapley-additive-explanations)** - Method to explain individual predictions

## 🗺️ Quick Access

| File | Description | Quick Run |
|------|-------------|----------|
| [`implementation.py`](./implementation.py) | Basic XGBoost usage | `uv run python implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Comprehensive examples | `uv run python sklearn_example.py` |
| `hyperparameter_tuning.py` (direncanakan) | Advanced tuning | — |
| [`exercise.ipynb`](./exercise.ipynb) | Interactive exercises | `uv run jupyter lab exercise.ipynb` |

## 📖 Theory and Concepts

### What is XGBoost?
[**XGBoost**](../../GLOSSARY.md#boosting) (eXtreme Gradient Boosting) is an advanced implementation of [**gradient boosting**](../../GLOSSARY.md#boosting) optimized for speed and performance. XGBoost is one of the most popular and successful [**algorithms**](../../GLOSSARY.md#algorithm) in machine learning competitions, especially for tabular data.

### Gradient Boosting Concept
[**Gradient Boosting**](../../GLOSSARY.md#boosting) is an [**ensemble method**](../../GLOSSARY.md#ensemble-method) that:
1. Builds [**models**](../../GLOSSARY.md#model) sequentially
2. Each new model tries to correct the errors of previous models
3. Uses [**gradient descent**](../../GLOSSARY.md#gradient-descent) to minimize the [**loss function**](../../GLOSSARY.md#cost-function-loss-function)

### Mathematical Formula
**Objective Function:**
```
Obj(θ) = Σ L(yi, ŷi) + Σ Ω(fk)
```

Where:
- `L(yi, ŷi)` = Loss function (training loss)
- `Ω(fk)` = Regularization term
- `ŷi = Σ fk(xi)` = Prediction (sum of trees)

**XGBoost Innovation:**
- **Second-order approximation** of loss function
- **Regularization** terms in objective
- **Optimized tree construction** algorithm

## 🎯 Kapan Menggunakan XGBoost?

### ✅ Sangat Cocok untuk:
- **Data tabular/structured data** 
- **Kompetisi machine learning** (Kaggle, etc.)
- **Data dengan missing values**
- **Mixed data types** (numerical + categorical)
- **Classification dan Regression tasks**
- **Feature importance** analysis
- **Dataset berukuran menengah-besar**

### ❌ Tidak ideal untuk:
- **Computer vision** (CNN lebih baik)
- **Natural Language Processing** (Transformer lebih baik)  
- **Dataset yang sangat kecil** (< 1000 samples)
- **Real-time prediction** (latency tinggi)
- **Data dengan struktur temporal kompleks**

## 🔍 Kelebihan dan Kekurangan

### Kelebihan:
- ✅ **Performa excellent** pada data tabular
- ✅ **Handle missing values** secara native
- ✅ **Built-in regularization** (L1, L2)
- ✅ **Feature importance** yang akurat
- ✅ **Parallel processing** dan GPU support
- ✅ **Cross-validation** built-in
- ✅ **Early stopping** untuk mencegah overfitting
- ✅ **Robust terhadap outliers**

### Kekurangan:
- ❌ **Hyperparameter tuning** yang kompleks
- ❌ **Memory intensive** untuk dataset besar
- ❌ **Prone to overfitting** jika tidak di-tune dengan baik
- ❌ **Black box model** (sulit interpretasi)
- ❌ **Training time** relatif lama
- ❌ **Sensitive to noisy data**

## ⚙️ Hyperparameters Utama

### Tree-specific Parameters:
- **n_estimators**: Jumlah trees (default: 100)
- **max_depth**: Kedalaman maksimum tree (default: 6)
- **min_child_weight**: Minimum sum of weights di leaf (default: 1)
- **subsample**: Fraction of samples untuk setiap tree (default: 1)
- **colsample_bytree**: Fraction of features untuk setiap tree (default: 1)

### Learning Parameters:
- **learning_rate/eta**: Step size shrinkage (default: 0.3)
- **objective**: Loss function (reg:squarederror, binary:logistic, etc.)
- **eval_metric**: Evaluation metric (rmse, logloss, auc, etc.)

### Regularization Parameters:
- **reg_alpha**: L1 regularization (default: 0)
- **reg_lambda**: L2 regularization (default: 1)
- **gamma**: Minimum loss reduction untuk split (default: 0)

## 📊 XGBoost vs Other Algorithms

| Aspek | XGBoost | Random Forest | Gradient Boosting | Linear Models |
|-------|---------|---------------|-------------------|---------------|
| **Accuracy** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Speed** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Interpretability** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Overfitting Risk** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Hyperparameter Tuning** | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🛠️ Instalasi dan Setup

### Instalasi XGBoost:
```bash
# Via pip
pip install xgboost

# Via conda
conda install -c conda-forge xgboost

# Via UV (recommended untuk project ini)
uv add xgboost
```

### GPU Support (Optional):
```bash
# For CUDA support
pip install xgboost[gpu]
```

## 📈 Evaluation dan Model Selection

### Classification Metrics:
- **Accuracy**: Overall correct predictions
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Harmonic mean of precision/recall
- **AUC-ROC**: Area under ROC curve
- **Log Loss**: Probabilistic loss

### Regression Metrics:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

### Cross-Validation:
XGBoost memiliki built-in CV:
```python
cv_results = xgb.cv(
    params, dtrain, num_boost_round=100,
    nfold=5, early_stopping_rounds=10
)
```

## 🎓 Best Practices

### 1. **Data Preparation**:
- Handle missing values (XGBoost can handle, but preprocessing might help)
- Encode categorical variables (LabelEncoder atau One-hot)
- Feature scaling tidak wajib, tapi bisa membantu

### 2. **Hyperparameter Tuning**:
- Start dengan default parameters
- Tune **n_estimators** dengan early stopping
- Adjust **learning_rate** dan **max_depth**
- Use **RandomizedSearchCV** atau **Optuna**

### 3. **Overfitting Prevention**:
- Use **early_stopping_rounds**
- Apply **regularization** (reg_alpha, reg_lambda)
- Reduce **learning_rate** dan increase **n_estimators**
- Use **cross-validation** untuk monitoring

### 4. **Feature Engineering**:
- Create interaction features
- Polynomial features untuk non-linear relationships
- Domain-specific feature engineering

### 5. **Model Interpretation**:
- Plot **feature importance**
- Use **SHAP** values untuk detailed interpretation
- Analyze **learning curves**

## 📊 Workflow Tipikal

1. **Data Preparation** → Clean, encode, split
2. **Baseline Model** → Default parameters
3. **Hyperparameter Tuning** → GridSearch/RandomSearch
4. **Cross-Validation** → Robust performance estimate  
5. **Feature Importance** → Understanding model
6. **Model Interpretation** → SHAP, partial dependence
7. **Final Evaluation** → Test set performance

## 🎯 Real-World Scenarios

### 🏆 Scenario 1: Kaggle Competition Style
**Dataset**: Tabular competition data (Titanic, House Prices)
**Goal**: Maximize leaderboard score
**Run**: `uv run python sklearn_example.py --scenario kaggle`

**What you'll master:**
- Feature engineering techniques
- Cross-validation strategies
- Ensemble methods with multiple models
- Hyperparameter optimization

### 💳 Scenario 2: Credit Risk Assessment
**Dataset**: Customer financial data
**Goal**: Predict loan default probability
**Run**: `uv run python sklearn_example.py --scenario credit_risk`

**What you'll master:**
- Handling imbalanced datasets
- Feature importance analysis
- Business-oriented metrics (precision/recall)
- Model interpretability with SHAP

### 🏭 Scenario 3: E-commerce Sales Forecasting
**Dataset**: Historical sales, seasonal data
**Goal**: Predict monthly sales volume
**Run**: `uv run python sklearn_example.py --scenario sales_forecast`

**What you'll master:**
- Time series feature engineering
- Seasonal decomposition
- Multi-step forecasting
- Business impact evaluation

### 🎭 Scenario 4: Customer Churn Prediction
**Dataset**: Customer behavior, usage patterns
**Goal**: Identify customers likely to churn
**Run**: `uv run python sklearn_example.py --scenario churn`

**What you'll master:**
- Behavioral feature engineering
- Cost-sensitive learning
- Actionable insights generation
- Model monitoring strategies

### 🏏 Scenario 5: Medical Diagnosis Support
**Dataset**: Patient symptoms, medical history
**Goal**: Assist in disease diagnosis
**Run**: `uv run python sklearn_example.py --scenario medical`

**What you'll master:**
- High-stakes prediction scenarios
- Confidence interval estimation
- False positive/negative trade-offs
- Regulatory compliance considerations

### 🔍 Hyperparameter Tuning Scenarios

> 🚧 Skrip `hyperparameter_tuning.py` khusus masih direncanakan. Alur tuning di bawah belum dapat dijalankan; gunakan [`sklearn_example.py`](./sklearn_example.py) untuk contoh yang tersedia saat ini.

#### Quick Tuning (5-10 minutes)
Random Search dengan budget kecil.

#### Production Tuning (30-60 minutes)
Pencarian berbasis Optuna dengan budget sedang.

#### Competition Tuning (2-4 hours)
Optimasi Bayesian dengan budget besar.

### [`implementation.py`](./implementation.py)
- **Basic XGBoost workflows** for beginners
- **Quick start examples** for each scenario
- **Baseline model setup**
- **Performance comparison** with other algorithms

### [`sklearn_example.py`](./sklearn_example.py)
- **Production-ready implementations** for all 5 scenarios
- **Advanced preprocessing** pipelines
- **Feature engineering** automation
- **Model evaluation** and **business metrics**
- **SHAP interpretability** integration

### `hyperparameter_tuning.py` (direncanakan)
- **Strategi tuning yang direncanakan**: Random Search, Bayesian, Optuna
- **Budget-aware optimization**
- **Parallel processing** for faster tuning
- **Competition-grade** parameter spaces

### [`exercise.ipynb`](./exercise.ipynb)
- **Interactive step-by-step** tutorials
- **Real dataset exploration** and **EDA**
- **Hyperparameter sensitivity** analysis
- **Model interpretation** exercises

## 🎓 Learning Progression

### Prerequisites
- **Machine Learning Basics**: [Linear Regression](../../01_supervised_learning/01_linear_regression/), [Decision Trees](../../01_supervised_learning/03_decision_trees/)
- **Ensemble Methods**: [Random Forest](../../01_supervised_learning/04_random_forest/)
- **Model Evaluation**: [Evaluation Techniques](../03_model_evaluation/)

### Recommended Learning Path
1. **Start** with [`implementation.py`](./implementation.py) - Basic concepts
2. **Practice** with [`exercise.ipynb`](./exercise.ipynb) - Interactive learning
3. **Explore** [`sklearn_example.py`](./sklearn_example.py) - Real scenarios
4. **Master** `hyperparameter_tuning.py` (direncanakan) - Advanced techniques

## 🏆 Competition Tips

1. **Feature Engineering** is king
2. **Ensemble** with different seeds
3. **Blend** XGBoost with LightGBM/CatBoost
4. **Pseudo-labeling** for semi-supervised learning
5. **Target encoding** for high-cardinality categorical
6. **Stratified K-Fold** for robust CV

### Related Advanced Algorithms

| Algorithm | Comparison | Use Case |
|-----------|------------|----------|
| **LightGBM** (direncanakan) | Faster training | Large datasets |
| **CatBoost** (direncanakan) | Better categorical handling | Mixed data types |
| [**Random Forest**](../../01_supervised_learning/04_random_forest/) | More interpretable | Feature importance |
| [**Neural Networks**](../../01_supervised_learning/08_neural_networks/) | More flexible | Complex patterns |

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [XGBoost Tutorials](https://xgboost.readthedocs.io/en/stable/tutorials/index.html)
- [Interpretable ML - XGBoost](https://christophm.github.io/interpretable-ml-book/)

---
**➡️ Next Step**: After mastering XGBoost, explore **LightGBM** and **CatBoost** for even more powerful ensemble methods!
