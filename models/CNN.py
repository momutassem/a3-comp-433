import math

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        activation_type,
        input_shape=None,
        output_size=6,
        depth=3,
        kernel_sizes=[5, 7, 9],
        in_chans=3,  
        out_chans=4,
    ):
        super(CNN, self).__init__()
        self.act = activation_type
        self.conv_module = nn.Sequential()

        # CONVOLUTIONAL MODULE
        for layer in range(depth):
            k_size = kernel_sizes[layer]

            self.conv_module.add_module(
                f"conv{layer}",
                nn.Conv2d(
                    in_channels=in_chans, 
                    out_channels=out_chans,  
                    kernel_size=k_size,
                    stride=1,
                    padding=k_size // 2, 
                ),
            )
            self.conv_module.add_module(
                f"bnorm{layer}",
                nn.BatchNorm2d(num_features=out_chans),
            )
            self.conv_module.add_module(f"act_{layer}", self.act)

            in_chans = out_chans
            out_chans = math.floor(out_chans * 1.5)

        # Shape of intermediate Feature maps
        out = torch.ones((1,) + (in_chans,) + tuple(input_shape[2:]))
        dense_input_size = self._num_flat_features(out)

        # DENSE MODULE
        self.dense_module = nn.Sequential()
        self.dense_module.add_module("flatten", nn.Flatten())
        self.dense_module.add_module(
            "fc_out",
            nn.Linear(
                in_features=dense_input_size,
                out_features=output_size,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def _num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
    
    def _initialize_weights(self):
        for layer in self.conv_module:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(layer.bias)

        for layer in self.dense_module:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.conv_module(x)
        x = self.dense_module(x)

        return x
