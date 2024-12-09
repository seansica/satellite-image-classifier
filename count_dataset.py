import os
from collections import defaultdict


def count_images_by_class(base_path):
    directories = {
        "test_rgb": os.path.join(base_path, "test_rgb"),
        "train_rgb": os.path.join(base_path, "train_rgb"),
        "val_rgb": os.path.join(base_path, "validate_rgb"),
    }

    counts = {}
    for split_name, split_path in directories.items():
        if split_name == "test_rgb":
            # Test dir has no subdirs, count total images
            counts[split_name] = {
                "total": sum(
                    1
                    for f in os.listdir(split_path)
                    if f.lower().endswith((".jpg", ".jpeg"))
                )
            }
            continue

        class_counts = defaultdict(int)
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                class_counts[class_name] = sum(
                    1
                    for f in os.listdir(class_path)
                    if f.lower().endswith((".jpg", ".jpeg"))
                )
        class_counts["total"] = sum(v for k, v in class_counts.items() if k != "total")
        counts[split_name] = dict(class_counts)

    return counts


if __name__ == "__main__":
    counts = count_images_by_class("../ICIP-2021")

    # Print summary
    print("\nOverall totals:")
    for split, count_dict in counts.items():
        print(f"{split}: {count_dict['total']}")

    print("\nClass distribution:")
    for split in ["train_rgb", "val_rgb"]:
        print(f"\n{split}:")
        for class_name, count in sorted(counts[split].items()):
            if class_name != "total":
                print(f"{class_name}: {count}")
