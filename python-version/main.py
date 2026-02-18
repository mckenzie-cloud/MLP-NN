import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------
# Hyperparameters
# -----------------------
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Dataset (KMNIST)
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),              # [0,1]
    transforms.Normalize((0.5,), (0.5,)) # normalize
])

train_dataset = datasets.KMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.KMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# -----------------------
# MLP Model
# -----------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(DEVICE)

# -----------------------
# Optimizer & Loss
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# -----------------------
# Training Loop
# -----------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss:.4f}")

# -----------------------
# Evaluation
# -----------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")
