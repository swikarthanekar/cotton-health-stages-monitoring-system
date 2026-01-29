import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Paths
DATA_DIR = "../data"
BATCH_SIZE = 8
EPOCHS = 15
LR = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_data = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_data.classes)

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

train_losses = []
val_accs = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    val_accs.append(acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_losses[-1]:.4f} | Val Acc: {acc:.3f}")

torch.save(model.state_dict(), "cotton_stage_model.pth")

plt.plot(train_losses, label="Train Loss")
plt.plot(val_accs, label="Val Accuracy")
plt.legend()
plt.title("Training Progress")
plt.savefig("training_curve.png")
plt.show()
