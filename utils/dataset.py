import os
import re
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from utils.general import get_transforms


class FingersDataset(Dataset):
    def __init__(self, config, preprocess=True, set="train"):
        self.data_dir = os.path.join(
            config["data_dir"], "test" if (set == "test") else "train"
        )
        if preprocess:
            self.preprocess()

        self.transforms = get_transforms(config, set)
        self.data = datasets.ImageFolder(self.data_dir)

        if set != "test":
            self.data = self.train_val_split(config, set)

    def preprocess(self):
        targets = ["0", "1", "2", "3", "4", "5"]

        def _get_label(file):
            match = re.search(r"_(\d)[LR].png$", file)
            return None if match is None else match.group(1)

        # Create folders for each unique label
        try:
            for target in targets:
                new_dir = os.path.join(self.data_dir, target)
                os.makedirs(new_dir)
        except OSError:
            return

        # Move data files into their corresponding label folders
        for file in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file)
            if os.path.isdir(file_path):
                continue

            label = _get_label(file)
            if label is None:
                continue
            file_dest = os.path.join(self.data_dir, label)
            shutil.move(file_path, file_dest)

    def train_val_split(self, config, set):
        labels = [label for _, label in self.data.imgs]

        train_indices, val_indices = train_test_split(
            range(len(labels)),
            test_size=config["valid_ratio"],
            stratify=labels,
            random_state=config["seed"],
        )

        if set == "train":
            return Subset(self.data, train_indices)
        else:
            return Subset(self.data, val_indices)

    def __getitem__(self, idx):
        sample, label = self.data[idx]
        sample = self.transforms(sample)
        label = torch.tensor(label)

        return sample, label

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)


class ClimateDataset(Dataset):
    def __init__(self, config, preprocess=True, set="train"):
        self.data_path = os.path.join(
            config["data_dir"], "test.csv" if (set == "test") else "train.csv"
        )
        self.window_size = config["window_size"]
        self.scaler = config["scaler"]
        self.data = pd.read_csv(self.data_path)["meantemp"]

        if preprocess:
            self.sequences, self.targets = self.preprocess()

    def preprocess(self):
        """
        Preprocess the data by creating sequences and normalizing them.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Normalized sequences of shape (num_samples, window_size).
                - Normalized target values of shape (num_samples,).
        """
        
        sequences, targets = self.create_sequences()
        sequences, targets = self.normalize_data(sequences, targets)

        return sequences, targets

    def create_sequences(self):
        """
        Create sequences and targets
        """
        sequences, targets = [], []

        for i in range(len(self.data) - self.window_size):
            sequence = self.data[i : i + self.window_size]
            target = self.data[i + self.window_size]
            sequences.append(sequence)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def normalize_data(self, sequences, targets):
        """
        Normalize the sequences and targets.
        """
        # Flatten sequences 
        sequences_reshaped = sequences.reshape(-1, 1)

        self.scaler.fit(sequences_reshaped)

        # Normalize sequences and targets
        sequences = self.scaler.transform(sequences_reshaped).reshape(sequences.shape)
        targets = self.scaler.transform(targets.reshape(-1, 1)).flatten()

        return sequences, targets

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32).unsqueeze(-1)
        target = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(-1)
        return sequence, target

    def __len__(self):
        return len(self.sequences)
