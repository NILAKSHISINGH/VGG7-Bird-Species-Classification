import os
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# DATASET PATH
# =========================

TRAIN_PATH = "../dataset/train"
VALID_PATH = "../dataset/valid"
TEST_PATH = "../dataset/test"


# =========================
# FUNCTION TO ANALYZE DATA
# =========================

def analyze_dataset(path, name):

    print("\n==============================")
    print(name)
    print("==============================")

    total_images = 0

    class_counts = {}

    corrupted_images = []

    # Get all classes
    classes = sorted(
        [
            folder
            for folder in os.listdir(path)
            if os.path.isdir(os.path.join(path, folder))
        ]
    )

    print("Number of classes:", len(classes))

    print("\nClasses:")

    for class_name in classes:
        print(class_name)

    # Count images
    for class_name in classes:

        class_path = os.path.join(
            path,
            class_name
        )

        count = 0

        for file_name in os.listdir(class_path):

            file_path = os.path.join(
                class_path,
                file_name
            )

            try:

                with Image.open(file_path) as image:

                    image.verify()

                count += 1

            except:

                corrupted_images.append(
                    file_path
                )

        class_counts[class_name] = count

        total_images += count

    # =========================
    # PRINT RESULTS
    # =========================

    print("\nImages per class:")

    for class_name, count in class_counts.items():

        print(
            f"{class_name}: {count}"
        )

    print("\nTotal images:", total_images)

    print(
        "Corrupted images:",
        len(corrupted_images)
    )

    if corrupted_images:

        print("\nCorrupted files:")

        for file in corrupted_images:

            print(file)

    # =========================
    # PLOT
    # =========================

    plt.figure(figsize=(14, 6))

    plt.bar(
        class_counts.keys(),
        class_counts.values()
    )

    plt.xlabel("Bird Species")

    plt.ylabel("Number of Images")

    plt.title(
        f"Class Distribution - {name}"
    )

    plt.xticks(
        rotation=90
    )

    plt.tight_layout()

    # Save graph
    os.makedirs(
        "../results",
        exist_ok=True
    )

    plt.savefig(
        f"../results/{name}_class_distribution.png"
    )

    plt.show()


# =========================
# RUN EDA
# =========================

analyze_dataset(
    TRAIN_PATH,
    "Train"
)

analyze_dataset(
    VALID_PATH,
    "Validation"
)

analyze_dataset(
    TEST_PATH,
    "Test"
)
