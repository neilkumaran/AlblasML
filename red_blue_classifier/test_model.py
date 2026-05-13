from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

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
MODEL_PATH = Path("red_blue_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test_model.py path/to/image.jpg")
        return

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    model = ColorClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)

    prediction = "RED" if output.item() < 0.5 else "BLUE"
    print(f"Prediction: {prediction} ({output.item():.2f})")


if __name__ == "__main__":
    main()
