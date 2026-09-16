import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import random


# ========================================
# 1. DEVICE
# ========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ========================================
# 2. PATHS
# ========================================

TEST_PATH = "dataset/test"
MODEL_PATH = "models/best_vgg7_birds.pth"


# ========================================
# 3. TRANSFORM
# ========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ========================================
# 4. DATASET
# ========================================

test_dataset = datasets.ImageFolder(
    TEST_PATH,
    transform=transform
)

class_names = test_dataset.classes


# ========================================
# 5. VGG7 MODEL
# ========================================

class VGG7(nn.Module):

    def __init__(self, num_classes=20):

        super(VGG7, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
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


# ========================================
# 6. LOAD MODEL
# ========================================

model = VGG7(
    num_classes=len(class_names)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ========================================
# 7. SELECT RANDOM IMAGES
# ========================================

num_images = 9

indices = random.sample(
    range(len(test_dataset)),
    num_images
)


# ========================================
# 8. PREDICT
# ========================================

os.makedirs(
    "results",
    exist_ok=True
)

plt.figure(figsize=(12, 12))


for i, index in enumerate(indices):

    image, true_label = test_dataset[index]

    input_image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(input_image)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()


    # Convert image back for display

    display_image = image.permute(
        1, 2, 0
    ).numpy()

    # Unnormalize

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    display_image = (
        display_image * std + mean
    )

    display_image = display_image.clip(
        0,
        1
    )


    # ====================================
    # PLOT
    # ====================================

    plt.subplot(3, 3, i + 1)

    plt.imshow(display_image)

    plt.axis("off")

    plt.title(
        f"True: {class_names[true_label]}\n"
        f"Pred: {class_names[prediction]}"
    )


plt.tight_layout()


# ========================================
# 9. SAVE
# ========================================

output_path = (
    "results/sample_predictions.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n========================================")
print("PREDICTION VISUALIZATION COMPLETED")
print("========================================")

print(
    "Saved at:",
    output_path
)