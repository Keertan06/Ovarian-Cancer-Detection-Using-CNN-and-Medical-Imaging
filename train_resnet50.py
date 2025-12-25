import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


DATA_DIR = "dataset"
BATCH_SIZE = 8
EPOCHS = 5
IMG_SIZE = 224
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model Setup

model = models.resnet50(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

model = model.to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# Training Loop
print(f"Training on {DEVICE.upper()} for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

print("Training complete")
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/ovarian_cancer_resnet50_torch.pth")
print("Model saved to models/ovarian_cancer_resnet50_torch.pth")
