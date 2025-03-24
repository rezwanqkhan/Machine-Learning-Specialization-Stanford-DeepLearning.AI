import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Load MNIST dataset (handwritten digits)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2️⃣ Normalize the data (convert pixel values 0-255 to 0-1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# 3️⃣ Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Convert 28x28 images into a 1D vector
    Dense(128, activation='relu', name="Hidden_Layer_1"),  # 128 neurons
    Dense(64, activation='relu', name="Hidden_Layer_2"),  # 64 neurons
    Dense(10, activation='softmax', name="Output_Layer")  # 10 neurons for 10 classes (digits 0-9)
])

# 4️⃣ Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5️⃣ Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 6️⃣ Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 7️⃣ Make predictions on test images
predictions = model.predict(X_test)
# Get layer weights
weights, biases = model.get_layer("Hidden_Layer_1").get_weights()

print(f"Weights Shape: {weights.shape}")  # Should be (784, 128)
print(f"Biases Shape: {biases.shape}")    # Should be (128,)

print(f"First 5 Weights: {weights[:5]}")  # Print first 5 weight vectors
print(f"First 5 Biases: {biases[:5]}")    # Print first 5 bias values
# 8️⃣ Plot some predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i], cmap='gray')
    ax.set_title(f"Pred: {np.argmax(predictions[i])}")
    ax.axis('off')
plt.show()
