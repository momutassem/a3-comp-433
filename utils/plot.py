import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_batch_accs(train_batch_accs, val_batch_accs):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_batch_accs)
    plt.title("Batch Training Accuracies Across Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_batch_accs, color="orange")
    plt.title("Batch Validation Accuracies Across Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm):
    class_labels = [f"{_}" for _ in range(6)]
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm_normalized,
        annot=True,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_training_loss(training_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(training_losses, color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.grid()
    plt.show()


def plot_predictions(labels, predictions, scaler):
    """
    Plots the actual labels vs. predictions.
    """
    # Get the original temperatures back
    actual_labels = scaler.inverse_transform(np.array(labels).reshape(-1, 1)).flatten()
    actual_predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(actual_labels, label="Actual", color="tab:red", alpha=0.8)
    plt.plot(actual_predictions, label="Predicted", color="tab:blue", linestyle=":")
    plt.xlabel("Time Steps")
    plt.ylabel("Mean Temperature")
    plt.title("Actual vs Predicted Mean Temperatures")
    plt.legend()
    plt.grid()
    plt.show()
