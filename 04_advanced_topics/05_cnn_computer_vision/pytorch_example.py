"""
Optional PyTorch/Keras comparison for CNN
This file is optional and tests don't require it
"""

def pytorch_comparison():
    """Compare with PyTorch CNN (if available)"""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from implementation import generate_product_images
        import numpy as np
        
        print("=" * 60)
        print("PyTorch CNN Comparison")
        print("=" * 60)
        
        # Generate data
        X_train, y_train = generate_product_images(n_samples=50, random_state=42)
        X_train_tensor = torch.FloatTensor(np.array(X_train)).unsqueeze(1)  # Add channel dim
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        # Simple PyTorch CNN
        class SimplePyTorchCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 8, kernel_size=3)
                self.pool = nn.MaxPool2d(2)
                self.fc = nn.Linear(8 * 13 * 13, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.pool(torch.relu(self.conv(x)))
                x = x.view(x.size(0), -1)
                x = self.sigmoid(self.fc(x))
                return x
        
        model = SimplePyTorchCNN()
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Train
        print("\nTraining PyTorch CNN...")
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
        
        # Evaluate
        with torch.no_grad():
            predictions = (model(X_train_tensor) > 0.5).float()
            accuracy = (predictions == y_train_tensor).float().mean()
            print(f"\nPyTorch CNN Accuracy: {accuracy:.3f}")
        
        return True
        
    except ImportError:
        print("PyTorch not available (optional dependency)")
        return False


def keras_comparison():
    """Compare with Keras CNN (if available)"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from implementation import generate_product_images
        import numpy as np
        
        print("\n" + "=" * 60)
        print("Keras CNN Comparison")
        print("=" * 60)
        
        # Generate data
        X_train, y_train = generate_product_images(n_samples=50, random_state=42)
        X_train_array = np.array(X_train).reshape(-1, 28, 28, 1)
        y_train_array = np.array(y_train)
        
        # Simple Keras CNN
        model = keras.Sequential([
            keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        
        print("\nTraining Keras CNN...")
        history = model.fit(X_train_array, y_train_array, epochs=100, verbose=0)
        
        print(f"Final loss: {history.history['loss'][-1]:.4f}")
        print(f"Final accuracy: {history.history['accuracy'][-1]:.3f}")
        
        return True
        
    except ImportError:
        print("TensorFlow/Keras not available (optional dependency)")
        return False


if __name__ == "__main__":
    pytorch_available = pytorch_comparison()
    keras_available = keras_comparison()
    
    if not (pytorch_available or keras_available):
        print("\n" + "=" * 60)
        print("Neither PyTorch nor Keras available")
        print("Install with: pip install torch  OR  pip install tensorflow")
        print("=" * 60)
