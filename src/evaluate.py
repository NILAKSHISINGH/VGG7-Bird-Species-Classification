import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# ==============================
# 1. DEVICE
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==============================
# 2. PATHS
# ==============================

TEST_PATH = "dataset/test"
MODEL_PATH = "models/best_vgg7_birds.pth"


# ==============================
# 3. TRANSFORMS
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# 4. DATASET
# ==============================

test_dataset = datasets.ImageFolder(
    TEST_PATH,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

class_names = test_dataset.classes

print("\nNumber of classes:", len(class_names))

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, name)


# ==============================
# 5. VGG7 MODEL
# ==============================

class VGG7(nn.Module):

    def __init__(self, num_classes=20):

        super(VGG7, self).__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )


    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ==============================
# 6. LOAD MODEL
# ==============================

model = VGG7(num_classes=len(class_names))

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("\nModel loaded successfully.")


# ==============================
# 7. PREDICTIONS
# ==============================

all_predictions = []
all_labels = []

correct = 0
total = 0

criterion = nn.CrossEntropyLoss()

total_loss = 0.0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        _, predictions = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predictions == labels).sum().item()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ==============================
# 8. RESULTS
# ==============================

accuracy = 100 * correct / total
average_loss = total_loss / total

print("\n========================================")
print("EVALUATION RESULTS")
print("========================================")

print(f"Test Loss: {average_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Correct Predictions: {correct}/{total}")


# ==============================
# 9. CLASSIFICATION REPORT
# ==============================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# ==============================
# 10. CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ==============================
# 11. SAVE RESULTS
# ==============================

os.makedirs("results", exist_ok=True)

np.savetxt(
    "results/confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

print("\nConfusion matrix saved to:")
print("results/confusion_matrix.csv")
