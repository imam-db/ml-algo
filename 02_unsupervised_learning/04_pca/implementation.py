"""
Principal Component Analysis (PCA) Implementation from Scratch
==============================================================

PCA via Singular Value Decomposition (SVD) on centered data.
Handles sign ambiguity when comparing with sklearn.
"""

import numpy as np
from typing import Optional

class PCAFromScratch:
    """
    Principal Component Analysis implementation from scratch using NumPy.
    
    Uses SVD on centered data: X_centered = U @ Σ @ V^T
    Principal components are the right singular vectors (V).
    
    Parameters:
    -----------
    n_components : int or None, default=None
        Number of components to keep. If None, keep all components.
        
    Attributes:
    -----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space (directions of maximum variance)
    explained_variance_ : ndarray of shape (n_components,)
        Amount of variance explained by each component
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each component
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set
    n_components_ : int
        The estimated number of components
    n_features_ : int
        Number of features in the training data
    n_samples_ : int
        Number of samples in the training data
    """
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        
        # Attributes set during fit
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_features_ = None
        self.n_samples_ = None
    
    def fit(self, X: np.ndarray) -> 'PCAFromScratch':
        """
        Fit the model with X.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_samples_, self.n_features_ = X.shape
        
        # Determine number of components to keep
        if self.n_components is None:
            self.n_components_ = min(self.n_samples_, self.n_features_)
        else:
            self.n_components_ = min(self.n_components, self.n_samples_, self.n_features_)
        
        # Center the data (subtract mean)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        # Perform SVD: X_centered = U @ Σ @ V^T
        # U: (n_samples, n_samples), left singular vectors
        # S: (min(n_samples, n_features),), singular values
        # Vt: (n_features, n_features), right singular vectors (transposed)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Principal components are the rows of Vt (columns of V)
        # Shape: (n_components, n_features)
        self.components_ = Vt[:self.n_components_]
        
        # Explained variance = (singular values)^2 / (n_samples - 1)
        # This matches sklearn's calculation
        self.explained_variance_ = (S[:self.n_components_] ** 2) / (self.n_samples_ - 1)
        
        # Total variance (sum of all squared singular values)
        total_variance = (S ** 2).sum() / (self.n_samples_ - 1)
        
        # Explained variance ratio
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data
        """
        if self.components_ is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Center the data
        X_centered = X - self.mean_
        
        # Project onto principal components: X_transformed = X_centered @ components.T
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model with X and apply dimensionality reduction.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed training data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Parameters:
        -----------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data
            
        Returns:
        --------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Data transformed back to original space
        """
        if self.components_ is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        X_transformed = np.asarray(X_transformed, dtype=np.float64)
        
        # Reconstruct: X_centered_reconstructed = X_transformed @ components
        X_reconstructed = X_transformed @ self.components_ + self.mean_
        
        return X_reconstructed
    
    def get_covariance(self) -> np.ndarray:
        """
        Compute data covariance with the fitted model.
        
        Returns:
        --------
        cov : ndarray of shape (n_features, n_features)
            Estimated covariance of the data
        """
        if self.components_ is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        # Covariance = components.T @ diag(explained_variance) @ components
        return (self.components_.T @ 
                np.diag(self.explained_variance_) @ 
                self.components_)


def compare_pca_sign_invariance(X: np.ndarray, pca1: PCAFromScratch, 
                                pca2: PCAFromScratch) -> np.ndarray:
    """
    Compare two PCA results accounting for sign ambiguity.
    
    PCA components are only defined up to sign (can be flipped).
    This function finds the sign correction to maximize agreement.
    
    Parameters:
    -----------
    X : ndarray
        Original data
    pca1, pca2 : PCAFromScratch
        Two fitted PCA models to compare
        
    Returns:
    --------
    signs : ndarray of shape (n_components,)
        Sign correction for each component (+1 or -1)
    """
    # Transform data with both models
    X_transformed1 = pca1.transform(X)
    X_transformed2 = pca2.transform(X)
    
    # For each component, find the sign that maximizes correlation
    n_components = min(pca1.n_components_, pca2.n_components_)
    signs = np.zeros(n_components)
    
    for i in range(n_components):
        # Correlation with positive sign
        corr_pos = np.corrcoef(X_transformed1[:, i], X_transformed2[:, i])[0, 1]
        # Correlation with negative sign
        corr_neg = np.corrcoef(X_transformed1[:, i], -X_transformed2[:, i])[0, 1]
        
        signs[i] = 1.0 if abs(corr_pos) > abs(corr_neg) else -1.0
    
    return signs


if __name__ == "__main__":
    # Simple demonstration
    print("PCA from Scratch - Demo")
    print("=" * 40)
    
    # Generate sample data with correlation
    rng = np.random.default_rng(42)
    n_samples = 200
    
    # Create correlated features
    x1 = rng.normal(0, 1, n_samples)
    x2 = 2 * x1 + rng.normal(0, 0.5, n_samples)  # Correlated with x1
    x3 = rng.normal(0, 0.3, n_samples)  # Independent noise
    
    X = np.column_stack([x1, x2, x3])
    
    print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Fit PCA
    pca = PCAFromScratch(n_components=2)
    X_transformed = pca.fit_transform(X)
    
    print(f"\nPCA Results:")
    print(f"  Components shape: {pca.components_.shape}")
    print(f"  Explained variance: {pca.explained_variance_}")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Cumulative variance: {pca.explained_variance_ratio_.cumsum()}")
    
    # Reconstruction
    X_reconstructed = pca.inverse_transform(X_transformed)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"\n  Reconstruction error (2 components): {reconstruction_error:.6f}")
    
    # Full reconstruction (all components)
    pca_full = PCAFromScratch(n_components=3)
    X_transformed_full = pca_full.fit_transform(X)
    X_reconstructed_full = pca_full.inverse_transform(X_transformed_full)
    reconstruction_error_full = np.mean((X - X_reconstructed_full) ** 2)
    print(f"  Reconstruction error (all components): {reconstruction_error_full:.10f}")
    
    print("\nInterpretation:")
    print(f"  First 2 components capture {pca.explained_variance_ratio_.sum()*100:.1f}% of variance")
    print(f"  Perfect reconstruction with all {pca_full.n_components_} components")
