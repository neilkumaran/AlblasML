from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ColorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = Path("dataset")
MODEL_PATH = Path("red_blue_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    dataset = datasets.ImageFolder(
        DATASET_DIR,
        transform=transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        ),
    )
    print("Class labels:", dataset.class_to_idx)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ColorClassifier().to(DEVICE)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
