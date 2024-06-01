# Standard library imports
import os
import json

# Related third party imports
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class Food101Dataset(Dataset):
    def __init__(self, data_root, meta_file, transform=None):
        self.data_root = Path(data_root) / "images"
        self.transform = transform
        with open(meta_file, 'r') as f:
            self.data = json.load(f)

        self.samples = []
        for label, items in self.data.items():
            for item in items:
                self.samples.append((item, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        while True:
            img_path, label = self.samples[idx]
            img_path = self.data_root / f"{img_path}.jpg"

            if not img_path.exists():
                print(f"Warning: File {img_path} does not exist. Skipping this file.")
                idx = (idx + 1) % len(self.samples)
                continue

            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label


def get_data_loaders(data_root, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = Food101Dataset(
        data_root=data_root, meta_file=os.path.join(data_root, 'meta', 'train.json'), transform=transform)
    test_dataset = Food101Dataset(
        data_root=data_root, meta_file=os.path.join(data_root, 'meta', 'test.json'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
