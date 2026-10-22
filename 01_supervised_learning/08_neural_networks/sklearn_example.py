"""
Neural Networks - Comparison with sklearn MLPClassifier
"""

import numpy as np
from sklearn.neural_network import MLPClassifier as SklearnMLP
from sklearn.metrics import accuracy_score, classification_report
from implementation import MLPClassifier, generate_spg_outlet_data

def main():
    print("=" * 60)
    print("Neural Networks - sklearn comparison")
    print("=" * 60)
    
    # Generate data
    X, y = generate_spg_outlet_data(n_samples=500, random_state=42)
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Our implementation
    print("\n1. Our from-scratch MLP:")
    our_mlp = MLPClassifier(
        hidden_layers=[32, 16],
        activation='relu',
        learning_rate=0.01,
        max_iter=500,
        random_state=42
    )
    our_mlp.fit(X_train, y_train)
    our_pred = our_mlp.predict(X_test)
    our_acc = accuracy_score(y_test, our_pred)
    print(f"   Test accuracy: {our_acc:.4f}")
    
    # Sklearn implementation
    print("\n2. sklearn MLPClassifier:")
    sklearn_mlp = SklearnMLP(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42
    )
    sklearn_mlp.fit(X_train, y_train)
    sklearn_pred = sklearn_mlp.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    print(f"   Test accuracy: {sklearn_acc:.4f}")
    
    print("\n" + "=" * 60)
    print("Classification Report (Our implementation):")
    print("=" * 60)
    print(classification_report(y_test, our_pred,
                                target_names=['Low Risk', 'Med Risk', 'High Risk']))

if __name__ == "__main__":
    main()
