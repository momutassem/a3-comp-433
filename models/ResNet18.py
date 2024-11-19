import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, config, pre_trained=False, freeze_backbone=True):
        super(ResNet18, self).__init__()

        if pre_trained:
            self.model = models.resnet18(weights=config["resnet_weights"])
            if freeze_backbone:
                self._freeze_backbone()
        else:
            self.model = models.resnet18()
        
        # Replace ResNet18's FC layer
        self.model.fc = nn.Linear(self.model.fc.in_features, config["output_size"])

    def _freeze_backbone(self):
        """Freeze all layers except the final fully connected layer."""
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)
