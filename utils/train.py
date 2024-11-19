import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from utils.metrics import (
    init_test_metrics,
    init_train_metrics,
    update_metrics,
)


def create_data_loaders(
    config: dict, include_val_loader=True, include_test_loader=True
):
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        config (dict): Configuration dictionary containing:
            - "data_dir": dataset directory
            - "valid_ratio": validation split ratio
            - "batch_size": batch size
            - "seed": random seed
        include_val_loader (bool): Whether to create a validation DataLoader.
        include_test_loader (bool): Whether to create a test DataLoader.

    Returns:
       tuple: (train_loader, val_loader, test_loader)
    """
    dataset = config["dataset"]
    train_data = dataset(config, preprocess=True, set="train")
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)

    val_loader = None
    if include_val_loader:
        val_data = dataset(config, preprocess=True, set="val")
        val_loader = DataLoader(
            val_data, batch_size=config["batch_size"], shuffle=False
        )

    test_loader = None
    if include_test_loader:
        test_data = dataset(config, preprocess=True, set="test")
        test_loader = DataLoader(
            test_data, batch_size=config["batch_size"], shuffle=False
        )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """
    Trains the model for one epoch.
    """
    metrics = init_train_metrics()
    model.train()

    with tqdm(total=len(train_loader), desc="Training", leave=False) as pbar:
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = update_metrics(outputs, labels, loss, metrics, phase="train")

            pbar.update(1)

    metrics["loss"] /= len(train_loader)
    metrics["acc"] = (metrics["correct_preds"] / metrics["total_preds"]) * 100

    return metrics


def validate(model, val_loader, criterion, device):
    metrics = init_train_metrics()
    model.eval()

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc="Validation", leave=False) as pbar:
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)

                metrics = update_metrics(outputs, labels, loss, metrics, phase="train")

                pbar.update(1)

    metrics["loss"] /= len(val_loader)
    metrics["acc"] = (metrics["correct_preds"] / metrics["total_preds"]) * 100

    return metrics


def evaluate(model, test_loader, criterion, device, output_size):
    if output_size == 1:
        return evaluate_reg(model, test_loader, criterion, device)
    metrics = init_test_metrics()
    model.eval()

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Evaluation", leave=False) as pbar:
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)

                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, labels)

                metrics = update_metrics(outputs, labels, loss, metrics, phase="eval")

                pbar.update(1)

    # Final metrics
    metrics["loss"] /= len(test_loader)
    metrics["predictions"] = np.concatenate(metrics["predictions"])
    metrics["labels"] = np.concatenate(metrics["labels"])

    return metrics


def evaluate_reg(model, test_loader, criterion, device):
    model.eval()
    metrics = {"predictions": [], "labels": []}
    test_loss = 0.0

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, label) 
            test_loss += loss

            metrics["predictions"].extend(outputs.cpu().numpy().flatten())  
            metrics["labels"].extend(label.cpu().numpy().flatten())        

    metrics["loss"] = test_loss / len(test_loader)

    return metrics
