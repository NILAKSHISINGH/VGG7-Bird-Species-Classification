import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vgg7 import VGG7


# ==========================================
# 1. DEVICE / GPU
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("========================================")
print("DEVICE INFORMATION")
print("========================================")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

print()


# ==========================================
# 2. DATASET PATHS
# ==========================================

TRAIN_PATH = "dataset/train"
VALID_PATH = "dataset/valid"
TEST_PATH = "dataset/test"


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


valid_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. LOAD DATASETS
# ==========================================

print("Loading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_PATH,
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    VALID_PATH,
    transform=valid_test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_PATH,
    transform=valid_test_transform
)


# ==========================================
# 5. CHECK CLASSES
# ==========================================

print()
print("========================================")
print("DATASET INFORMATION")
print("========================================")

print("Number of classes:", len(train_dataset.classes))

print("Training images:", len(train_dataset))
print("Validation images:", len(valid_dataset))
print("Test images:", len(test_dataset))

print()
print("Bird classes:")

for i, class_name in enumerate(train_dataset.classes):
    print(i, ":", class_name)


# ==========================================
# 6. DATA LOADERS
# ==========================================

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ==========================================
# 7. CREATE VGG7 MODEL
# ==========================================

print()
print("========================================")
print("CREATING VGG7 MODEL")
print("========================================")

num_classes = len(train_dataset.classes)

model = VGG7(num_classes=num_classes)

model = model.to(device)

print(model)


# ==========================================
# 8. LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# 9. OPTIMIZER
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ==========================================
# 10. TRAINING SETTINGS
# ==========================================

EPOCHS = 20

best_val_accuracy = 0.0


# ==========================================
# 11. CREATE MODELS FOLDER
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# 12. TRAINING
# ==========================================

print()
print("========================================")
print("STARTING TRAINING")
print("========================================")


for epoch in range(EPOCHS):

    # --------------------------------------
    # TRAINING
    # --------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_accuracy = (
        100 * correct / total
    )


    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    model.eval()

    val_loss = 0.0

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_loss /
        len(valid_loader)
    )

    val_accuracy = (
        100 * val_correct / val_total
    )


    # --------------------------------------
    # PRINT RESULTS
    # --------------------------------------

    print()
    print("----------------------------------------")
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


    # --------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "models/best_vgg7_birds.pth"
        )

        print(
            ">>> Best model saved!"
        )


# ==========================================
# 13. TRAINING FINISHED
# ==========================================

print()
print("========================================")
print("TRAINING COMPLETED")
print("========================================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print()
print(
    "Model saved at:"
)

print(
    "models/best_vgg7_birds.pth"
)