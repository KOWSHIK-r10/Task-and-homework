import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

inputs = Input(shape=(X_train.shape[1],))
hidden1 = Dense(8, activation='relu', name='hidden1')(inputs)
hidden2 = Dense(4, activation='relu', name='hidden2')(hidden1)
outputs = Dense(1, activation='sigmoid', name='output')(hidden2)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

activation_model = Model(inputs=model.input, outputs=[model.get_layer('hidden1').output,
                                                      model.get_layer('hidden2').output,
                                                      model.get_layer('output').output])
activations = activation_model.predict(X_test)
act_hidden1, act_hidden2, act_output = activations

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolor='k', s=50)
axes[0].set_title("Original Data")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

axes[1].scatter(act_hidden1[:, 0], act_hidden1[:, 1], c=y_test, cmap='coolwarm', edgecolor='k', s=50)
axes[1].set_title("Hidden Layer 1 Activations")
axes[1].set_xlabel("Neuron 1")
axes[1].set_ylabel("Neuron 2")

if act_hidden2.shape[1] >= 2:
    axes[2].scatter(act_hidden2[:, 0], act_hidden2[:, 1], c=y_test, cmap='coolwarm', edgecolor='k', s=50)
    axes[2].set_title("Hidden Layer 2 Activations")
    axes[2].set_xlabel("Neuron 1")
    axes[2].set_ylabel("Neuron 2")
else:
    axes[2].scatter(np.arange(len(act_hidden2)), act_hidden2[:, 0], c=y_test, cmap='coolwarm', edgecolor='k', s=50)
    axes[2].set_title("Hidden Layer 2 Activations")
    axes[2].set_xlabel("Index")
    axes[2].set_ylabel("Activation")

axes[3].scatter(np.arange(len(act_output)), act_output, c=y_test, cmap='coolwarm', edgecolor='k', s=50)
axes[3].set_title("Output Activations")
axes[3].set_xlabel("Index")
axes[3].set_ylabel("Probability")

plt.tight_layout()
plt.show()
