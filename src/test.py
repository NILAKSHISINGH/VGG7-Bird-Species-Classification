import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vgg7 import VGG7


# ==========================================
# 1. DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# 2. TEST DATA
# ==========================================

TEST_PATH = "dataset/test"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_dataset = datasets.ImageFolder(
    TEST_PATH,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 3. LOAD MODEL
# ==========================================

model = VGG7(
    num_classes=len(test_dataset.classes)
)

model.load_state_dict(
    torch.load(
        "models/best_vgg7_birds.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ==========================================
# 4. LOSS
# ==========================================

criterion = nn.CrossEntropyLoss()

test_loss = 0.0
correct = 0
total = 0


# ==========================================
# 5. TESTING
# ==========================================

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        test_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


# ==========================================
# 6. RESULTS
# ==========================================

test_loss = test_loss / len(test_loader)

test_accuracy = (
    100 * correct / total
)


print()
print("========================================")
print("TEST RESULTS")
print("========================================")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)

print(
    f"Correct Predictions: {correct}/{total}"
)
