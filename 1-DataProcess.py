# 1

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load + normalize + one-hot encode + split (all in one flow)
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

X_train_full, X_test = X_train_full.astype("float32")/255.0, X_test.astype("float32")/255.0
y_train_full, y_test = to_categorical(y_train_full, 10), to_categorical(y_test, 10)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Output shapes
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Val:   {X_val.shape}, {y_val.shape}")
print(f"Test:  {X_test.shape}, {y_test.shape}")
