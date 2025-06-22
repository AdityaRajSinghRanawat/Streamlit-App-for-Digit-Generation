import torch
import torch.nn as nn

class DigitGenerator(nn.Module):
    def __init__(self, num_classes=10, noise_dim=100):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_embed(labels)
        x = torch.cat((noise, label_embedding), dim=1)
        return self.model(x).view(-1, 1, 28, 28)
