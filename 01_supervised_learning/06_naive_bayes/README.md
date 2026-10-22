# Naive Bayes

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## Overview

Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem with the "naive" assumption of feature independence.

## Bayes' Theorem

$$P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$$

## Variants

### Gaussian Naive Bayes
- For continuous features
- Assumes normal distribution
- Good for real-valued data

### Multinomial Naive Bayes
- For count/frequency features
- Common in text classification
- Laplace smoothing (α parameter)

## Implementation

```python
from implementation import GaussianNaiveBayes, MultinomialNaiveBayes

# Gaussian (continuous features)
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)

# Multinomial (count data)
mnb = MultinomialNaiveBayes(alpha=1.0)
mnb.fit(X_train, y_train)
predictions = mnb.predict(X_test)
```

## Advantages

- Fast training and prediction
- Works well with small datasets
- Handles high dimensions well
- Provides probability estimates

## Related Files

- `implementation.py` - Naive Bayes from scratch
- `sklearn_example.py` - Comparison with sklearn
- `exercise.ipynb` - Interactive exercises

**Happy Learning! 🎓** | *[🇮🇩 Bahasa Indonesia](README.id.md)*
