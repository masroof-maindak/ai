import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Load Heart Disease Dataset (Cleaned version from UCI)
def load_heart_data():
    data = pd.read_csv("processed.cleveland.data", header=None, na_values="?")

    column_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    data.columns = column_names

    # Preprocessing:
    # 1. Convert target to binary (0 = no disease, 1 = disease)
    data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)

    # 2. Handle missing values (drop rows with NaN)
    data.dropna(inplace=True)

    # 3. Separate features (X) and labels (y)
    X = data.drop("target", axis=1).values
    y = data["target"].values.reshape(-1, 1)

    # 4. Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


# Initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    return {
        "W1": np.random.randn(input_size, hidden_size) * 0.01,
        "b1": np.zeros((1, hidden_size)),
        "W2": np.random.randn(hidden_size, output_size) * 0.01,
        "b2": np.zeros((1, output_size)),
    }


# Forward propagation
def forward_pass(X, params, apply_dropout=False, keep_prob=0.8):
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    # --- HIDDEN LAYER ---
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    dropout_mask = None  # Initialize mask to None
    if apply_dropout:
        # Create random binary mask with probability 'keep_prob'
        # Values < keep_prob become 1 (keep), others 0 (drop)
        dropout_mask = (np.random.rand(*A1.shape) < keep_prob).astype(float)
        # Apply the mask to activations
        A1 *= dropout_mask
        # Scale activations to maintain expected value (Inverted Dropout)
        A1 /= keep_prob

    # --- OUTPUT LAYER ---
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)  # Final prediction (probability)

    cache = {
        "A1": A1,  # Activation of hidden layer (potentially after dropout)
        "A2": A2,
        "Z1": Z1,
        "Z2": Z2,
        "dropout_mask": dropout_mask,  # Store the mask used
        # Store A1 *before* dropout scaling/mask if needed for specific derivative calculations
        # "A1_pre_dropout": sigmoid(Z1) # Optional, depending on backprop implementation
    }
    return A2, cache


def bce_loss(y_true, y_pred):
    m = y_true.shape[0]
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return -(1 / m) * np.sum(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )


# Accuracy calculation
def compute_accuracy(y_true, y_pred):
    predictions = np.where(y_pred > 0.5, 1, 0)
    return np.mean(predictions == y_true)


# Backpropagation
def backward_pass(X, y, cache, params, learning_rate, keep_prob=1.0):
    # Retrieve parameters and cached values
    W1, W2 = params["W1"], params["W2"]
    A1, A2 = cache["A1"], cache["A2"]  # A1 here is potentially post-dropout
    Z1 = cache["Z1"]
    dropout_mask = cache["dropout_mask"]
    m = X.shape[0]  # Number of samples

    # --- OUTPUT LAYER ---
    # Gradient of Loss w.r.t Z2 (for BCE loss with sigmoid activation)
    # dL/dZ2 = dL/dA2 * dA2/dZ2 = (A2 - y)
    dZ2 = A2 - y  # Shape (m, output_size)

    # Gradient of Loss w.r.t W2
    # dL/dW2 = dL/dZ2 * dZ2/dW2 = dZ2 * A1^T
    # Average over the batch
    dW2 = (1 / m) * (A1.T @ dZ2)  # Shape (hidden_size, output_size)

    # Gradient of Loss w.r.t b2
    # dL/db2 = dL/dZ2 * dZ2/db2 = dZ2 * 1
    # Average over the batch by summing and dividing
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)  # Shape (1, output_size)

    # --- HIDDEN LAYER GRADIENTS ---
    # Gradient of Loss w.r.t A1 (post-dropout if applied)
    # dL/dA1 = dL/dZ2 * dZ2/dA1 = dZ2 @ W2^T
    dA1 = dZ2 @ W2.T  # Shape (m, hidden_size)

    # Apply inverse dropout mask and scaling if dropout was used in forward pass
    if dropout_mask is not None:
        # Apply same dropout mask as forward pass (zeros out gradients for dropped neurons)
        dA1 *= dropout_mask
        # Scale gradients to match forward pass scaling (inverse of inverted dropout)
        dA1 /= keep_prob

    # Gradient of Loss w.r.t Z1
    # dL/dZ1 = dL/dA1 * dA1/dZ1
    # dA1/dZ1 is the derivative of the hidden layer activation (sigmoid)
    # We need sigmoid_derivative(A1_pre_dropout), which is sigmoid_derivative(sigmoid(Z1))
    dZ1 = dA1 * sigmoid_derivative(sigmoid(Z1))  # Shape (m, hidden_size)

    # Gradient of Loss w.r.t W1
    # dL/dW1 = dL/dZ1 * dZ1/dW1 = dZ1 * X^T
    # Average over the batch
    dW1 = (1 / m) * (X.T @ dZ1)  # Shape (input_size, hidden_size)

    # Gradient of Loss w.r.t b1
    # dL/db1 = dL/dZ1 * dZ1/db1 = dZ1 * 1
    # Average over the batch
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)  # Shape (1, hidden_size)

    params["W2"] -= learning_rate * dW2
    params["b2"] -= learning_rate * db2
    params["W1"] -= learning_rate * dW1
    params["b1"] -= learning_rate * db1


def train(
    X, y, hidden_size=10, epochs=1000, learning_rate=0.1, keep_prob=0.8, patience=50
):
    m = X.shape[1]
    output_size = 1

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Stratify for class balance
    )
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Initialize parameters
    params = initialize_parameters(m, hidden_size, output_size)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_params = params.copy()  # Store best parameters based on validation loss
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # --- Training Step ---
        y_pred_train, cache = forward_pass(
            X_train, params, apply_dropout=True, keep_prob=keep_prob
        )

        loss = bce_loss(y_train, y_pred_train)
        acc = compute_accuracy(y_train, y_pred_train)

        backward_pass(X_train, y_train, cache, params, learning_rate, keep_prob)

        # --- Validation Step ---
        # Forward pass without dropout for validation
        y_pred_val, _ = forward_pass(X_val, params, apply_dropout=False)
        # Calculate validation loss
        val_loss = bce_loss(y_val, y_pred_val)
        # Calculate validation accuracy
        val_acc = compute_accuracy(y_val, y_pred_val)

        # Store metrics for plotting
        history["loss"].append(loss)
        history["acc"].append(acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print progress periodically
        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Loss: {loss:.4f}, Acc: {acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        # --- Early Stopping Check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {
                k: v.copy() for k, v in params.items()
            }  # Deep copy best params
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹ Early stopping triggered after {epoch + 1} epochs.")
                params = best_params  # Restore best parameters
                break

    print(f"\nTraining finished. Best Validation Loss: {best_val_loss:.4f}")
    # Calculate final accuracy on validation set using best params
    final_val_pred, _ = forward_pass(X_val, params, apply_dropout=False)
    final_val_acc = compute_accuracy(y_val, final_val_pred)
    print(f"Final Validation Accuracy (using best params): {final_val_acc:.4f}")

    return params, history  # Return the best parameters found


# Plotting function
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Binary Cross-Entropy)")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("output.png")


def main():
    X, y = load_heart_data()

    print("\nStarting training...")

    HIDDEN_LAYER_SIZE = 16
    LEARNING_RATE = 0.05
    EPOCHS = 2000
    KEEP_PROBABILITY = 0.7
    PATIENCE = 100

    _, training_history = train(
        X,
        y,
        hidden_size=HIDDEN_LAYER_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        keep_prob=KEEP_PROBABILITY,
        patience=PATIENCE,
    )

    plot_history(training_history)


if __name__ == "__main__":
    main()
