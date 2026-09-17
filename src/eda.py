import os
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image


# ========================================
# 1. PATHS
# ========================================

DATASET_PATH = "dataset"

SPLITS = ["train", "validation", "test"]


# ========================================
# 2. FIND DATASET STRUCTURE
# ========================================

print("========================================")
print("VGG7 BIRD DATASET - EDA")
print("========================================")


for split in SPLITS:

    split_path = os.path.join(DATASET_PATH, split)

    if not os.path.exists(split_path):
        print(f"\n{split.upper()} folder not found:")
        print(split_path)
        continue

    classes = [
        folder for folder in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, folder))
    ]

    print(f"\n{split.upper()} DATASET")
    print("Number of classes:", len(classes))


    # ========================================
    # 3. COUNT IMAGES
    # ========================================

    class_counts = {}

    for class_name in sorted(classes):

        class_path = os.path.join(
            split_path,
            class_name
        )

        image_count = 0

        for file in os.listdir(class_path):

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            ):
                image_count += 1

        class_counts[class_name] = image_count


    total_images = sum(class_counts.values())

    print("Total images:", total_images)


    # ========================================
    # 4. PRINT CLASS DISTRIBUTION
    # ========================================

    print("\nClass distribution:")

    for class_name, count in class_counts.items():

        print(
            f"{class_name}: {count}"
        )


    # ========================================
    # 5. PLOT DISTRIBUTION
    # ========================================

    plt.figure(figsize=(14, 8))

    plt.bar(
        class_counts.keys(),
        class_counts.values()
    )

    plt.title(
        f"{split.capitalize()} Dataset Class Distribution"
    )

    plt.xlabel("Bird Species")
    plt.ylabel("Number of Images")

    plt.xticks(
        rotation=90
    )

    plt.tight_layout()


    # ========================================
    # 6. SAVE PLOT
    # ========================================

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = (
        f"results/{split}_class_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Plot saved: {output_path}"
    )


# ========================================
# 7. IMAGE SIZE ANALYSIS
# ========================================

print("\n========================================")
print("IMAGE SIZE ANALYSIS")
print("========================================")


image_sizes = []

train_path = os.path.join(
    DATASET_PATH,
    "train"
)


if os.path.exists(train_path):

    for class_name in os.listdir(train_path):

        class_path = os.path.join(
            train_path,
            class_name
        )

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            ):

                image_path = os.path.join(
                    class_path,
                    file
                )

                try:

                    with Image.open(image_path) as img:

                        image_sizes.append(
                            img.size
                        )

                except Exception:
                    pass


print(
    "Images analyzed:",
    len(image_sizes)
)


if image_sizes:

    size_counts = Counter(
        image_sizes
    )

    print("\nMost common image sizes:")

    for size, count in size_counts.most_common(10):

        print(
            f"{size}: {count} images"
        )


# ========================================
# 8. SUMMARY
# ========================================

print("\n========================================")
print("EDA COMPLETED")
print("========================================")

print("\nGenerated files:")

for split in SPLITS:

    file_path = (
        f"results/{split}_class_distribution.png"
    )

    if os.path.exists(file_path):

        print(file_path)