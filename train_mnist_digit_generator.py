import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitGenerator().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Train (simple loss to replicate digits)
for epoch in range(10):
    for images, labels in loader:
        noise = torch.randn(images.size(0), 100).to(device)
        labels = labels.to(device)
        generated = model(noise, labels)
        loss = loss_fn(generated, images.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "digit_generator.pth")
