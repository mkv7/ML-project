# Moaeez Muhammad — MNIST Digit Classifier (PyTorch)
# Original code, no external license.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loss_hist, test_acc_hist = [], []

    epochs = 3
    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        avg_loss = running / len(train_loader.dataset)
        train_loss_hist.append(avg_loss)

        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / total
        test_acc_hist.append(acc)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - test acc: {acc:.4f}")

    # Save model
    os.makedirs('artifacts', exist_ok=True)
    torch.save(model.state_dict(), 'artifacts/mnist_cnn.pt')
    print("Saved model to artifacts/mnist_cnn.pt")

    # Plot curves
    plt.figure()
    plt.plot(range(1, epochs+1), train_loss_hist, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss')
    plt.savefig('artifacts/train_loss.png', dpi=150, bbox_inches='tight')

    plt.figure()
    plt.plot(range(1, epochs+1), test_acc_hist, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy')
    plt.savefig('artifacts/test_accuracy.png', dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()