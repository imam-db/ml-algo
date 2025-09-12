# XGBoost (eXtreme Gradient Boosting)

## 📖 Teori dan Konsep

### Apa itu XGBoost?
XGBoost (eXtreme Gradient Boosting) adalah implementasi advanced dari gradient boosting yang dioptimalkan untuk kecepatan dan performa. XGBoost adalah salah satu algoritma yang paling populer dan sukses dalam kompetisi machine learning, terutama untuk data tabular.

### Konsep Gradient Boosting
**Gradient Boosting** adalah ensemble method yang:
1. Membangun model secara sekuential
2. Setiap model baru mencoba memperbaiki kesalahan model sebelumnya
3. Menggunakan gradient descent untuk meminimalkan loss function

### Rumus Matematika
**Objective Function:**
```
Obj(θ) = Σ L(yi, ŷi) + Σ Ω(fk)
```

Dimana:
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

## 📝 Implementasi

Dalam folder ini, Anda akan menemukan:
- `implementation.py` - Basic XGBoost implementation dan tuning
- `sklearn_example.py` - Comprehensive examples dengan berbagai datasets
- `exercise.ipynb` - Hands-on exercises dengan real data
- `hyperparameter_tuning.py` - Advanced tuning techniques

## 🏆 Tips untuk Kompetisi

1. **Feature Engineering** is king
2. **Ensemble** dengan different seeds
3. **Blend** XGBoost dengan LightGBM/CatBoost
4. **Pseudo-labeling** untuk semi-supervised learning
5. **Target encoding** untuk high-cardinality categorical
6. **Stratified K-Fold** untuk robust CV

## 📚 Referensi

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [XGBoost Tutorials](https://xgboost.readthedocs.io/en/stable/tutorials/index.html)
- [Interpretable ML - XGBoost](https://christophm.github.io/interpretable-ml-book/)

---
**Next Step**: Setelah menguasai XGBoost, explore **LightGBM** dan **CatBoost** untuk ensemble methods yang lebih powerful!